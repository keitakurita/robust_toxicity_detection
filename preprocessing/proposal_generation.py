"""
Perform OOV substitution proposals and scoring

Go through the texts and generate FastText embedding-based proposals based on the embedding space similarity. Scoring options

- Global embedding-based scoring: Choose the one with largest similarity
- Local embedding-based scoring: Choose the best Bert proposal
       Issue: there may be replacement proposals that are not in Bert dictionary.
       Just use cosine similarity backup for scoring in this case
       Attempt to re-score a candidate if split into parts using Bert splitter (“delete” --> “del” “##ete”)
        - how to combine probabilities of word pieces? Mean?
- Text-based scoring: Levenshtein distance to score proposals for replacement

"""

import os, sys
import torch
from collections import namedtuple
import argparse
import numpy as np
import json
from typing import *
from allennlp.data.vocabulary import Vocabulary
import time, gc
import pandas as pd
from scipy.special import softmax
import nmslib, time
from allennlp.data.vocabulary import Vocabulary

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForMaskedLM

torch_device=torch.device('cuda')

bert_model_mlm = BertForMaskedLM.from_pretrained('bert-base-uncased')
bert_model_mlm.eval()
bert_model_mlm.to(torch_device)

for param in bert_model_mlm.parameters():
    param.requires_grad = False

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

bert_id2tok = dict()
for tok, tok_id in bert_tokenizer.vocab.items():
    bert_id2tok[tok_id] = tok

M = 30
efC = 200

from basic_processing import MAX_SEQ_LEN


MAX_COSINE_DIST=0.3

VOCAB_SIZE=30000

num_threads=8
K=10

DEBUG_PRINT=False

BertPredProbs = namedtuple('BertPred', ['batch_sent_id', 'pos_oov', 'logits'])

UtterData = namedtuple('SentData', ['batch_sent_id', 'pos_oov', 'tok_ids', 'oov_token'])


def get_bert_masked_inputs(toks, bert_tokenizer):
  res = []

  oov_pos = []
  bert_vocab = bert_tokenizer.vocab

  for i in range(len(toks)):
    if toks[i] not in bert_vocab:
      oov_pos.append(i)

  for pos in oov_pos:
    res.append((pos, '[CLS] %s [MASK] %s [SEP]' %
                (' '.join(toks[0:pos]), ' '.join(toks[pos + 1:]))))

  return res

def get_batch_data(torch_device, tokenizer, bert_tokenizer, sent_list, max_len=MAX_SEQ_LEN):
  batch_data_raw = []
  batch_max_seq_qty = 0
  batch_sent_id = -1
  for sent in sent_list:
    batch_sent_id += 1
    sent_toks = tokenizer(sent)
    for sent_oov_pos, text in get_bert_masked_inputs(sent_toks, bert_tokenizer):
      # To accurately get what is the position of [MASK] according
      # to BERT tokenizer, we need to re-tokenize the sentence using
      # the BERT tokenizer
      all_bert_toks = bert_tokenizer.tokenize(text)
      bert_toks = all_bert_toks[0:max_len]  # 512 is the max. Bert seq. length

      tok_ids = bert_tokenizer.convert_tokens_to_ids(bert_toks)
      pos_oov = None
      for i in range(len(bert_toks)):
        if bert_toks[i] == '[MASK]':
          pos_oov = i
          break
      assert (pos_oov is not None or len(all_bert_toks) > max_len)
      if pos_oov is not None:
        tok_qty = len(tok_ids)
        batch_max_seq_qty = max(batch_max_seq_qty, tok_qty)
        batch_data_raw.append(
          UtterData(batch_sent_id=batch_sent_id,
                    pos_oov=sent_oov_pos,
                    tok_ids=tok_ids,
                    oov_token=sent_toks[sent_oov_pos]))

  batch_qty = len(batch_data_raw)
  tok_ids_batch = np.zeros((batch_qty, batch_max_seq_qty), dtype=np.int64)  # zero is a padding symbol
  for k in range(batch_qty):
    tok_ids = batch_data_raw[k].tok_ids
    tok_ids_batch[k, 0:len(tok_ids)] = tok_ids

  tok_ids_batch = torch.from_numpy(tok_ids_batch).to(device=torch_device)

  return batch_data_raw, tok_ids_batch

def get_bert_preds_for_words_batch(torch_device, bert_model_mlm,
                                   batch_data_raw, tok_ids_batch,  # comes from get_batch_data
                                   word_ids,  # a list of IDS for which we generate logits
                                   max_len=MAX_SEQ_LEN):
  seg_ids = torch.zeros_like(tok_ids_batch, device=torch_device)

  batch_qty = len(batch_data_raw)

  # Main BERT model see modeling.py in https://github.com/huggingface/pytorch-pretrained-BERT
  bert = bert_model_mlm.bert
  # cls is an instance of BertOnlyMLMHead (see https://github.com/huggingface/pytorch-pretrained-BERT)
  cls = bert_model_mlm.cls
  # predictions are of the type BertLMPredictionHead (see https://github.com/huggingface/pytorch-pretrained-BERT)
  predictions = cls.predictions
  transform = predictions.transform

  # We don't use the complete decoding matrix, but only selected rows
  word_ids = torch.from_numpy(np.array(word_ids, dtype=np.int64)).to(device=torch_device)

  weight = predictions.decoder.weight[word_ids, :]
  bias = predictions.bias[word_ids]

  # Transformations from the main BERT model
  sequence_output, _ = bert(tok_ids_batch, seg_ids, attention_mask=None, output_all_encoded_layers=False)
  # Transformations from the BertLMPredictionHead model with the restricted last layer
  hidden_states = transform(sequence_output)
  logits = torch.nn.functional.linear(hidden_states, weight) + bias

  logits = logits.detach().cpu().numpy()

  res = []

  for k in range(batch_qty):
    pos_oov = batch_data_raw[k].pos_oov
    res.append(BertPredProbs(batch_sent_id=batch_data_raw[k].batch_sent_id,
                             pos_oov=pos_oov,
                             logits=logits[k, pos_oov]
                             )
               )

  return res


def main(argv):

  parser = argparse.ArgumentParser(description='Proposal Generation')
  parser.add_argument('--datapath', type=str,
                      required=False,
                      default = '',
                      help = '../data/jigsaw/')
  parser.add_argument('--vocabname', type=str,
                      required = False,
                      default = 'voc_basic_toks',
                      help = 'Output vocabulary file')
  parser.add_argument('--proctrain', type=str,
                      required = False,
                      default = 'train_basic.jsonl',
                      help = 'Processed train file')
  parser.add_argument('--proctest', type=str,
                      required = False,
                      default = 'test_basic.jsonl',
                      help = 'Processed test file')
  parser.add_argument('--xproctrain', type=str,
                      required = True,
                      default = 'train_basic_extended.jsonl',
                      help = 'Extended train file')
  parser.add_argument('--xproctrain', type=str,
                      required = True,
                      default = 'train_basic_extended.jsonl',
                      help = 'Extended test file')

  args = parser.parse_args(argv)
  print(args)

  vocab = Vocabulary.from_files("../data/jigsaw/data_ft_vocab")

  repl_oov_files = [("../data/jigsaw/test_proced.csv", "../data/jigsaw/test_proced_no_oov1.csv"),
                    ("../data/jigsaw/train.csv", "../data/jigsaw/train_no_oov1.csv")]

  ft_compiled_path = "../data/jigsaw/ft_model_bert_basic_tok.npy"  # Embeddings generated from the vocabulary
  fasttext_embeds = np.load(ft_compiled_path)

  bert_vocab_term_glob_ids = []
  bert_vocab_term_bert_ids = []

  for tok, bert_tok_id in bert_tokenizer.vocab.items():
    glob_tok_id = vocab.get_token_index(tok)
    if glob_tok_id > 1:
      bert_vocab_term_glob_ids.append(glob_tok_id)
      bert_vocab_term_bert_ids.append(bert_tok_id)

  bert_vocab_term_glob_ids = np.array(bert_vocab_term_glob_ids)
  bert_vocab_term_bert_ids = np.array(bert_vocab_term_bert_ids)
  fasttext_embeds[bert_vocab_term_glob_ids].shape

  num_threads = 0
  index_time_params = {'M': M, 'indexThreadQty': num_threads, 'efConstruction': efC, 'post': 0}
  print('Index-time parameters', index_time_params)

  # Space name should correspond to the space name
  # used for brute-force search
  space_name = 'cosinesimil'

  # Intitialize the library, specify the space, the type of the vector and add data points
  index = nmslib.init(method='hnsw', space=space_name, data_type=nmslib.DataType.DENSE_VECTOR)
  index.addDataPointBatch(fasttext_embeds[bert_vocab_term_glob_ids], bert_vocab_term_bert_ids)

  # Create an index
  start = time.time()
  index_time_params = {'M': M, 'indexThreadQty': num_threads, 'efConstruction': efC}
  index.createIndex(index_time_params)
  end = time.time()
  print('Index-time parameters', index_time_params)
  print('Indexing time = %f' % (end - start))

  # Setting query-time parameters
  efS = 200
  K = 10
  query_time_params = {'efSearch': efS}
  print('Setting query-time parameters', query_time_params)
  index.setQueryTimeParams(query_time_params)



  for src_file, dst_file in repl_oov_files:
    src_data = pd.read_csv(src_file)

    t0 = time.time()
    preds = []

    # all_src_sents = [' '.join(t['tokens']) for t in val_ds]
    all_src_sents = list(src_data['comment_text'])
    all_dst_sents = []

    # print(s)
    # preds.append(get_bert_top_preds(tokenizer, bert_tokenizer, s, 2))
    # preds.append(get_bert_masked_inputs(tokenizer(s), bert_tokenizer, sent))

    batch_qty_step = 20

    for batch_start_sent_id in range(0, len(all_src_sents), batch_qty_step):
      print('Batch start', batch_start_sent_id)

      batch_qty = min(batch_qty_step, len(all_src_sents) - batch_start_sent_id)

      batch_sents = [all_src_sents[k] for k in range(batch_start_sent_id,
                                                     batch_start_sent_id + batch_qty)]

      replace_dict = {k: dict() for k in range(0, batch_qty)}

      # batch_data raw contains elements
      # UtterData = namedtuple('SentData', ['batch_sent_id', 'pos_oov', 'tok_ids', 'oov_token')
      # NOTE: pos_oov is OOV index with respect to the original (not BERT) tokenizer!!!
      #
      # tok_ids_batch is a Tensor with padded Bert-specific token IDs ready
      # to be fed into a BERT model
      batch_data_raw, tok_ids_batch = get_batch_data(torch_device,
                                                     tokenizer, bert_tokenizer,
                                                     batch_sents,
                                                     MAX_SEQ_LEN)

      query_arr = []
      query_tok_oov_id = []

      for e in batch_data_raw:
        w = e.oov_token
        wCompr = remove_extra_chars(w)
        wid = vocab.get_token_index(wCompr)
        if w != wCompr:
          if wid < 2:
            wid = vocab.get_token_index(w)

        query_arr.append(fasttext_embeds[wid])
        query_tok_oov_id.append(wid)

      query_arr = np.array(query_arr)
      query_matrix = np.array(query_arr)
      query_qty = query_matrix.shape[0]

      if DEBUG_PRINT: print('Query matrix shape:', query_matrix.shape)

      start = time.time()
      # nbrs is array of tuples (neighbor array, distance array)
      # For cosine, the distance is 1 - cosine similarity
      # k-NN search returns Bert-specific token IDs
      nbrs = index.knnQueryBatch(query_matrix, k=K, num_threads=num_threads)
      end = time.time()
      if DEBUG_PRINT:
        print('kNN time total=%f (sec), per query=%f (sec), per query adjusted for thread number=%f (sec)' %
              (end - start, float(end - start) / query_qty, num_threads * float(end - start) / query_qty))

      neighb_tok_ids = set()

      for qid in range(query_qty):
        if query_tok_oov_id[qid] > 1:
          nbrs_ids = nbrs[qid][0]
          nbrs_dist = nbrs[qid][1]

          nqty = len(nbrs_ids)
          for t in range(nqty):
            if nbrs_dist[t] < MAX_COSINE_DIST:
              assert (nbrs_ids[t] < VOCAB_SIZE)
              neighb_tok_ids.add(nbrs_ids[t])

      neighb_tok_ids = list(neighb_tok_ids)

      preds = get_bert_preds_for_words_batch(torch_device,
                                             bert_model_mlm,
                                             batch_data_raw, tok_ids_batch,
                                             neighb_tok_ids)

      assert (len(preds) == query_qty)
      for qid in range(query_qty):
        e = batch_data_raw[qid]
        glob_sent_id = batch_start_sent_id + e.batch_sent_id
        assert (batch_sents[e.batch_sent_id] == all_src_sents[glob_sent_id])
        if is_apost_token(e.oov_token) or e.oov_token == "n't":
          # Thing's like "I don't" or "You're" are tokenized as do "I do n't" or "You 're'"
          pass  # TODO fix this
        elif query_tok_oov_id[qid] > 1:
          # Let's map neighbor IDs from each queries to respective
          # logits from the prediction set
          logit_map = dict()  # from Bert-specific token IDs to predicted logits
          assert (len(preds[qid].logits) == len(neighb_tok_ids))
          for i in range(len(neighb_tok_ids)):
            logit_map[neighb_tok_ids[i]] = preds[qid].logits[i]

          e = batch_data_raw[qid]
          if DEBUG_PRINT:
            print(all_src_sents[glob_sent_id])
            print("### OOV ###", e.oov_token)
            print([bert_id2tok[bert_tok_id] for bert_tok_id in nbrs[qid][0]])

          nbrs_sel_logits = []
          nbrs_sel_toks = []
          nbrs_sel_dists = []

          nbrs_ids = nbrs[qid][0]
          nbrs_dist = nbrs[qid][1]

          # print('Logit map:', logit_map)
          # print('neighb_tok_ids', neighb_tok_ids)

          nqty = len(nbrs_ids)
          for t in range(nqty):
            bert_tok_id = nbrs_ids[t]
            # nid is Bert-speicifc token ID
            if not bert_tok_id in neighb_tok_ids:
              if DEBUG_PRINT:
                print('Missing %s distance %g '
                      % (bert_id2tok[bert_tok_id],
                         nbrs_dist[t]))
            else:
              if nbrs_dist[t] < MAX_COSINE_DIST:
                nbrs_sel_logits.append(logit_map[bert_tok_id])
                nbrs_sel_toks.append(bert_id2tok[bert_tok_id])
                nbrs_sel_dists.append(nbrs_dist[t])

          if nbrs_sel_logits:
            nbrs_softmax = softmax(np.array(nbrs_sel_logits))
            nbrs_simil = 1 - np.array(nbrs_sel_dists)
            nbrs_simil_adj = nbrs_softmax * nbrs_simil

            best_tok_id = np.argmax(nbrs_simil_adj)

            # print("batch sent id:",e.batch_sent_id, e.pos_oov, best_tok_id)
            # print(replace_dict[e.batch_sent_id])
            assert (not e.pos_oov in replace_dict[e.batch_sent_id])
            replace_dict[e.batch_sent_id][e.pos_oov] = nbrs_sel_toks[best_tok_id]

            if DEBUG_PRINT:
              print('Selected info, best_tok:', nbrs_sel_toks[best_tok_id])
              for k in range(len(nbrs_sel_logits)):
                print(nbrs_sel_toks[k], nbrs_softmax[k],
                      nbrs_sel_dists[k], nbrs_simil_adj[k])
          else:
            if DEBUG_PRINT: print('Nothing found!')

          # if DEBUG_PRINT: print(preds[qid])
          if DEBUG_PRINT:
            print("====================================================================")

      # gc.collect()
      # torch.cuda.empty_cache()
      for k in range(0, batch_qty):
        src_sent = batch_sents[k]
        rd = replace_dict[k]
        # print('Replacement dict:', rd)
        dst_sent = replace_by_patterns(tokenizer, src_sent, rd)
        all_dst_sents.append(dst_sent)
        if DEBUG_PRINT:
          print("====================================================================")
          print('Replacement dict:', rd)
          print(src_sent)
          print('------------')
          print(dst_sent)
          print("====================================================================")

      # break

    t1 = time.time()
    print('# of src sentences:', len(all_src_sents),
          "# of dst sentences:", len(all_dst_sents),
          ' time elapsed:', t1 - t0)
    src_data['comment_text'] = all_dst_sents
    src_data.to_csv(dst_file, index=False)



if __name__ == '__main__':
  main(sys.argv[1:])