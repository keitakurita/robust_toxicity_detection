#!/usr/bin/env python3
"""
Read data JSONL one line at a time
Apply TBD function
Augment JSON
Write enriched JSONL to another file
"""

import os, sys, tqdm

sys.path.append('.')
sys.path.append('..')

import argparse
import json
import numpy as np
import time
import spacy
from contextlib import contextmanager
from allennlp.common import Tqdm
from typing import *
from preprocessing.preprocessing_common import *


KNN_K=10

BATCH_QTY_STEP = 64
MINI_BATCH_QTY = 32
DEBUG_PRINT=True

DictList = List[Dict]

import torch

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForMaskedLM

torch_device = torch.device('cuda')
# torch_device=torch.device('cpu')

bert_model_mlm = BertForMaskedLM.from_pretrained('bert-base-uncased')
bert_model_mlm.eval()
bert_model_mlm.to(torch_device)

for param in bert_model_mlm.parameters():
  param.requires_grad = False

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')


def main(argv):

  parser = argparse.ArgumentParser(description='Data augmentation workflow')
  parser.add_argument('--datapath', type=str,
                      required=False,
                      default = '',
                      help = '../data/jigsaw/')
  parser.add_argument('--infile', type=str,
                      required = True,
                      default = 'test_basic.jsonl',
                      help = 'Input file that underwent basic processing')
  parser.add_argument('--outfile', type=str,
                      required = True,
                      default = 'test_basic_aug.jsonl',
                      help = 'Augmented output file')
  parser.add_argument('--ftmatname', type=str,
                      required = True,
                      default = 'ft_basic_toks.txt',
                      help = 'fastText embeddings file')

  args = parser.parse_args(argv)
  print(args)


  spacy_nlp = spacy.load(SPACY_MODEL_TYPE,
                         disable=['parser', 'ner', 'pos'])
  spacy_word_map = get_canon_case_map(spacy_nlp)

  # Read all embeddings
  ft_name = os.path.join(args.datapath, args.ftmatname)
  word_arr, embed_arr, word_to_embed_map = read_embeds_and_words(ft_name)
  print('Read %d words, %d embeddings, %d word-to-embed maps from the joint-vocab file %s' %
        (len(word_arr), len(embed_arr), len(word_to_embed_map), ft_name))

  # Select only embeddings with tokens from the Spacy vocabulary
  spacy_word_arr = []
  spacy_embed_arr = []

  from allennlp.data.vocabulary import Vocabulary

  vocab = Vocabulary.from_instances(get_spacy_vocab_instances(spacy_nlp))

  for idx, token in vocab.get_index_to_token_vocabulary().items():
    if idx > 1:
      spacy_word_arr.append(token)
      spacy_embed_arr.append(word_to_embed_map[token])



  embed_index = create_embed_index(spacy_embed_arr)
  word_index = create_word_index(spacy_word_arr)

  all_src_objs = []
  all_src_sents = []

  with open(os.path.join(args.datapath,args.infile),"r") as f:
    for line in f:
      obj = json.loads(line)
      all_src_objs.append(obj)
      all_src_sents.append(obj["tokens"])



  t0 = time.time()

  for batch_start_sent_id in tqdm.tqdm(range(0, len(all_src_sents), BATCH_QTY_STEP)):
    if DEBUG_PRINT:
      print('Batch start', batch_start_sent_id)

    batch_qty = min(BATCH_QTY_STEP, len(all_src_sents) - batch_start_sent_id)

    batch_sents = [toksToLower(all_src_sents[k]) for k in
                   range(batch_start_sent_id, batch_start_sent_id + batch_qty)]
    # toksToLower(all_src_sents[k]) for k in [batch_start_sent_id + 3]]

    oov_res_dict = [[] for k in range(batch_qty)]

    # batch_data raw contains elements
    # UtterData = namedtuple('SentData', ['batch_sent_id', 'sent_pos_oov', 'bert_pos_oov', 'tok_ids', 'oov_token'])
    #
    # tok_ids_batch is a Tensor with padded Bert-specific token IDs ready
    # to be fed into a BERT model
    batch_data_raw, tok_ids_batch, tok_mask_batch = get_batch_data(torch_device,
                                                                   spacy_word_map,
                                                                   bert_tokenizer,
                                                                   batch_sents,
                                                                   MAX_BERT_LEN)

    query_arr = [e.oov_token.lower() for e in batch_data_raw]

    oov_scores = get_pooled_neighbors(embed_index, word_index,
                                      spacy_word_arr, word_to_embed_map,
                                      KNN_K, 10 * KNN_K,
                                      query_arr)

    # for query_res in oov_scores:
    #    print(query_res)
    #    print("***************")

    add_bert_scores(bert_model_mlm, MINI_BATCH_QTY,
                    torch_device, bert_tokenizer,
                    oov_scores,
                    batch_data_raw, tok_ids_batch, tok_mask_batch)

    for qid in range(len(batch_data_raw)):
      e = batch_data_raw[qid]

      oov_scores_cased = {}
      for k, v in oov_scores[qid].items():
        for ks, vs in v.items():
          v[ks] = float(vs)
        oov_scores_cased[spacy_word_map[k]] = v
      oov_scores[qid] = oov_scores_cased

      oov_res_dict[e.batch_sent_id].append({'sent_pos_oov': e.sent_pos_oov,
                                            'oov_token': e.oov_token,
                                            'cand_scores': oov_scores[qid]})
      assert (batch_sents[e.batch_sent_id][e.sent_pos_oov] == e.oov_token)

      if DEBUG_PRINT:
        print('---------------')
        print(' '.join(batch_sents[e.batch_sent_id]))
        print('Batch sent id: ', e.batch_sent_id, ' query word:', query_arr[qid])
        print('---------------')
        print(' '.join(oov_scores[qid]))
        for k, v in oov_scores[qid].items():
          print(k, v)
        print('===============')

    # gc.collect()
    # torch.cuda.empty_cache()
    for k in range(0, batch_qty):
      src_objs = all_src_objs[batch_start_sent_id + k]

      src_objs["oov"] = oov_res_dict[k]

      if DEBUG_PRINT:
        print(src_objs)

    #break

  t1 = time.time()
  print('# of src objs:', len(all_src_objs),
        ' time elapsed:', t1 - t0)

  with open(os.path.join(args.datapath,args.outfile),"w") as f:
    for elem in all_src_objs:
      objStr = json.dumps(elem)
      f.write(objStr + '\n')

if __name__ == '__main__':
  main(sys.argv[1:])

