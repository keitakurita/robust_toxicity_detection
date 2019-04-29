#NOTE spaCy version 2.0.18
import nmslib
import time
import numpy as np
from allennlp.data.fields import TextField
from allennlp.data.token_indexers import TokenIndexer
from typing import *
from overrides import overrides
from allennlp.data import  Token, Instance
from allennlp.data.fields import  MetadataField
from allennlp.data.token_indexers import SingleIdTokenIndexer
import random
import torch


JIGSAW_LABEL_NAMES = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
JIGSAW_TEXT_NAME = "comment_text"
JIGSAW_TOXIC_COL = "toxic"

OFFENSEVAL_LABEL_NAMES = ["is_offensive","is_offensive_targeted","is_offensive_indiv_targeted","is_offensive_group_targeted","is_offensive_other_targeted"]
OFFENSEVAL_TEXT_NAME = "tweet"
OFFENSEVAL_TOXIC_COL = "is_offensive"

HATEVAL_LABEL_NAMES = ["HS","TR","AG","toxic"]
HATEVAL_TEXT_NAME = "text"
HATEVAL_TOXIC_COL = "toxic"

MAX_SEQ_LEN = 512

MAX_BERT_LEN=256

NON_WORDS = ['@@UNKNOWN@@', '@@PADDING@@']

SPACY_MODEL_TYPE = "en_core_web_sm"


HOMO_SUBS = {'-': '˗', '9': '৭', '8': 'Ȣ', '7': '𝟕', '6': 'б', '5': 'Ƽ', '4': 'Ꮞ', '3': 'Ʒ', '2': 'ᒿ', '1': 'l',
             '0': 'O', "'": '`', 'a': 'ɑ', 'b': 'Ь', 'c': 'ϲ', 'd': 'ԁ', 'e': 'е', 'f': '𝚏', 'g': 'ɡ', 'h': 'հ',
             'i': 'і', 'j': 'ϳ', 'k': '𝒌', 'l': 'ⅼ', 'm': 'ｍ', 'n': 'ո', 'o': 'о', 'p': 'р', 'q': 'ԛ', 'r': 'ⲅ',
             's': 'ѕ', 't': '𝚝', 'u': 'ս', 'v': 'ѵ', 'w': 'ԝ', 'x': '×', 'y': 'у', 'z': 'ᴢ'}

HOMO_SUBS_EXTEND = {'-': '˗', '9': '৭', '8': 'Ȣ', '7': '𝟕', '6': 'б', '5': 'Ƽ', '4': 'Ꮞ', '3': 'Ʒ', '2': 'ᒿ', '1': 'l',
             '0': 'O', "'": '`', 'a': 'ɑ', 'b': 'Ь', 'c': 'ϲ', 'd': 'ԁ', 'e': 'е', 'f': '𝚏', 'g': 'ɡ', 'h': 'հ',
             'i': 'і', 'j': 'ϳ', 'k': '𝒌', 'l': 'ⅼ', 'm': 'ｍ', 'n': 'ո', 'o': 'о', 'p': 'р', 'q': 'ԛ', 'r': 'ⲅ',
             's': 'ѕ', 't': '𝚝', 'u': 'ս', 'v': 'ѵ', 'w': 'ԝ', 'x': '×', 'y': 'у', 'z': 'ᴢ',
             'A': 'А', 'B': 'В', 'C': 'С', 'D': 'Đ', 'E': 'Е', 'F': 'ፑ','G': 'Б','H': 'Н', 'I': '丨', 'J': 'フ',
             'K': 'К',
             'L': '∟',
             'M': 'М', 'N': 'Ν', 'O': 'О', 'P': 'Р','Q': 'Ọ',
             'R': 'Я','S':'$', 'T': 'Т', 'U': 'ሀ', 'V': 'Ѵ', 'W': 'ሠ',
             'X': 'Х','Y': 'У','Z': 'Ζ'
              }


TokenList = List[Token]

DEBUG = False

def set_seed(seed):
  print('Set seed: ', seed)
  random.seed(seed)
  np.random.seed(seed)
  return seed


class MemoryOptimizedTextField(TextField):

  @overrides
  def __init__(self, tokens: List[str], token_indexers: Dict[str, TokenIndexer]) -> None:
    self.tokens = tokens
    self._token_indexers = token_indexers
    self._indexed_tokens: Optional[Dict[str, TokenList]] = None
    self._indexer_name_to_indexed_token: Optional[Dict[str, List[str]]] = None
    # skip checks for tokens

  @overrides
  def index(self, vocab):
    super().index(vocab)
    self.tokens = None  # empty tokens


def has_newline(s):
  return s.find('\n') >=0

def count_lowercase(s):
  return sum([int(c == c.lower()) for c in s])


def get_canon_case_map(nlp):
  dict_tmp = dict()

  for t in nlp.vocab:
    dict_tmp[t.text.lower()] = set()

  for t in nlp.vocab:
    dict_tmp[t.text.lower()].add(t.text)

  dict_map = dict()

  for key, tset in dict_tmp.items():
    lst = [(count_lowercase(s), s) for s in tset]
    lst.sort(reverse=True)
    choice_str = lst[0][1]
    dict_map[key] = choice_str

  return dict_map


def get_spacy_vocab_instances(nlp) -> Iterator[Instance]:
  words = set([t.text.lower().strip() for t in nlp.vocab])

  fields = {}

  for w in words:
    w = w.strip()
    if w and w.find('\n') < 0:
      fields["tokens"] = MemoryOptimizedTextField([w], {"tokens": SingleIdTokenIndexer()})
      yield Instance(fields)


def read_embeds_and_words(file_name):
  word_list, embed_list = [], []
  word_to_embed_map = dict()

  with open(file_name, encoding="utf8") as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      fld = line.split()
      w = fld[0]
      embed = np.array([float(s) for s in fld[1:]])
      embed_list.append(embed)
      word_list.append(w)
      word_to_embed_map[w] = embed

  return word_list, np.vstack(embed_list), word_to_embed_map


def create_embed_index(embeds, num_threads = 0, M = 30, efC = 200):

  index_time_params = {'M': M, 'indexThreadQty': num_threads, 'efConstruction': efC, 'post': 0}
  print('Index-time parameters', index_time_params)

  # Space name should correspond to the space name
  # used for brute-force search
  space_name = 'cosinesimil'

  # Intitialize the library, specify the space, the type of the vector and add data points
  index = nmslib.init(method='hnsw', space=space_name, data_type=nmslib.DataType.DENSE_VECTOR)
  index.addDataPointBatch(embeds)

  # Create an index
  start = time.time()
  index_time_params = {'M': M, 'indexThreadQty': num_threads, 'efConstruction': efC}
  index.createIndex(index_time_params)
  end = time.time()
  if DEBUG:
    print('Index-time parameters', index_time_params)
    print('Indexing time = %f' % (end - start))

  return index


def create_word_index(words, num_threads = 0, M = 30, efC = 200):
  index_time_params = {'M': M, 'indexThreadQty': num_threads, 'efConstruction': efC, 'post': 0}
  print('Index-time parameters', index_time_params)

  # Space name should correspond to the space name
  # used for brute-force search
  space_name = 'leven'

  # Intitialize the library, specify the space, the type of the vector and add data points
  index = nmslib.init(method='hnsw', space=space_name, dtype=nmslib.DistType.INT, data_type=nmslib.DataType.OBJECT_AS_STRING)
  index.addDataPointBatch(list(words))

  # Create an index
  start = time.time()
  index_time_params = {'M': M, 'indexThreadQty': num_threads, 'efConstruction': efC}
  index.createIndex(index_time_params)
  end = time.time()
  if DEBUG:
    print('Index-time parameters', index_time_params)
    print('Indexing time = %f' % (end - start))


  return index

def query_index(index, K, query_arr, num_threads=0, efS=200):
  # Querying
  query_time_params = {'efSearch': efS}
  if DEBUG:
    print('Setting query-time parameters', query_time_params)
  index.setQueryTimeParams(query_time_params)

  query_qty = len(query_arr)
  start = time.time()
  res = index.knnQueryBatch(query_arr, k=K, num_threads=num_threads)
  end = time.time()
  if DEBUG:
    print('kNN time total=%f (sec), per query=%f (sec), per query adjusted for thread number=%f (sec)' %
          (end - start, float(end - start) / query_qty, num_threads * float(end - start) / query_qty))

  return res


def looks_like_url(w):
  w = w.lower()
  return w.startswith('http:/') or w.startswith('https:/') or w.startswith('ftp:/') or \
          w.startswith('<http:/') or w.startswith('<https:/') or w.startswith('<ftp:/') or \
          w.find('@') >= 0


def get_spacy_vocab_instances(nlp) -> Iterator[Instance]:
  words = set([t.text.lower().strip() for t in nlp.vocab])

  fields = {}

  for w in words:
    w = w.strip()
    if looks_like_url(w):
        continue
    if w and w.find('\n') < 0:
      fields["tokens"] = MemoryOptimizedTextField([w], {"tokens": SingleIdTokenIndexer()})
      yield Instance(fields)


import copy

LEVEN_SIMIL = 'leven'
FASTTEXT_SIMIL = 'fast'
BERT_SIMIL = 'bert'

EMPTY_RES = {LEVEN_SIMIL: 0,
             FASTTEXT_SIMIL: 0,
             BERT_SIMIL: 0}


def remove_nonalpha_startend(s):
  # print('!!!', s)
  while s and not s[0].isalnum():
    s = s[1:]
  while s and not s[-1].isalnum():
    s = s[0:-1]
  # print('@@@@', s)
  return s

def add_pooled_res(res, w, simil, res_key):
  if not w in res:
    res[w] = copy.copy(EMPTY_RES)
  res[w][res_key] = simil


# We assume that there are word_arr and embed_arr that
# 1) have the same size
# 2) embed_arr[i] corresponds to word_arr[i] (for all i)
# 3) they are indexed using word_index and embed_index respectively
def get_pooled_neighbors(embed_index, word_index,
                         word_arr, word_to_embed_map,
                         K,
                         merge_K,
                         query_arr, num_threads=0, efS=200):
  res = []
  query_qty = len(query_arr)

  query_arr_leven = []
  for w in query_arr:
    w1 = remove_nonalpha_startend(w)
    if w1 != w and w1 != '':
      w = w1
    query_arr_leven.append(w)

  ids_word, dists_word = zip(*query_index(word_index, merge_K, query_arr_leven, num_threads=num_threads, efS=efS))

  query_arr_embed = []
  for w in query_arr:
    w1 = remove_nonalpha_startend(w)
    if w != w1 and w1 in word_to_embed_map:
      w = w1
    query_arr_embed.append(word_to_embed_map[w])

  ids_embed, dists_embed = zip(*query_index(embed_index, merge_K,
                                            np.array(query_arr_embed),
                                            num_threads=num_threads, efS=efS))

  for qid in range(query_qty):
    tmp_query_res = dict()

    qty_word = len(ids_word[qid])
    for i in range(qty_word):
      w_res = word_arr[ids_word[qid][i]]
      query_word = query_arr[qid]
      max_len = float(max(len(w_res), len(query_word)))
      simil = 1 - float(dists_word[qid][i]) / max_len
      add_pooled_res(tmp_query_res, w_res, simil, LEVEN_SIMIL)

    qty_embed = len(ids_embed[qid])
    for i in range(qty_embed):
      w_res = word_arr[ids_embed[qid][i]]
      simil = 1 - 0.5 * dists_embed[qid][i]
      add_pooled_res(tmp_query_res, w_res, simil, FASTTEXT_SIMIL)

    final_query_res = dict()

    i_word = 0
    i_embed = 0

    while (i_word < qty_word or i_embed < qty_embed) and (len(final_query_res) < K):
      if i_word < qty_word:
        w_res = word_arr[ids_word[qid][i_word]]
        if not w_res in final_query_res:
          final_query_res[w_res] = tmp_query_res[w_res]
        else:
          final_query_res[w_res][LEVEN_SIMIL] = tmp_query_res[w_res][LEVEN_SIMIL]
        i_word += 1
      if i_embed < qty_embed:
        w_res = word_arr[ids_embed[qid][i_embed]]
        if not w_res in final_query_res:
          final_query_res[w_res] = tmp_query_res[w_res]
        else:
          final_query_res[w_res][FASTTEXT_SIMIL] = tmp_query_res[w_res][FASTTEXT_SIMIL]
        i_embed += 1

    res.append(final_query_res)

  return res



# Returns arrays of arrays if there's an OOV word for an empty array instead
# Each array element is a tuple:
# position of OOV word (with respect to the original tokenizer), sent for BERT tokenizer
def get_bert_masked_inputs(toks, vocab, bert_tokenizer):
  res = []

  oov_pos = []
  bert_vocab = bert_tokenizer.vocab

  for i in range(len(toks)):
    if toks[i] not in vocab:
      oov_pos.append(i)

  for pos in oov_pos:
    res.append((pos, '[CLS] %s [MASK] %s [SEP]' %
                (' '.join(toks[0:pos]), ' '.join(toks[pos + 1:]))))

  return res


from collections import namedtuple

# pos_oov is OOV index with respect to the original (not BERT) tokenizer!!!
UtterData = namedtuple('SentData', ['batch_sent_id', 'sent_pos_oov', 'bert_pos_oov', 'tok_ids', 'oov_token'])


# sent_list contains list of token lists
def get_batch_data(torch_device, vocab, bert_tokenizer, sent_list, max_len=MAX_BERT_LEN):
  batch_data_raw = []
  batch_max_seq_qty = 0
  batch_sent_id = -1
  for sent_toks in sent_list:
    batch_sent_id += 1
    for sent_pos_oov, text in get_bert_masked_inputs(sent_toks, vocab, bert_tokenizer):
      # To accurately get what is the position of [MASK] according
      # to BERT tokenizer, we need to re-tokenize the sentence using
      # the BERT tokenizer
      all_bert_toks = bert_tokenizer.tokenize(text)
      bert_toks = all_bert_toks[0:max_len]  # 512 is the max. Bert seq. length

      tok_ids = bert_tokenizer.convert_tokens_to_ids(bert_toks)
      bert_pos_oov = None
      for i in range(len(bert_toks)):
        if bert_toks[i] == '[MASK]':
          bert_pos_oov = i
          break
      assert (bert_pos_oov is not None or len(all_bert_toks) > max_len)
      if bert_pos_oov is not None:
        tok_qty = len(tok_ids)
        batch_max_seq_qty = max(batch_max_seq_qty, tok_qty)
        batch_data_raw.append(
          UtterData(batch_sent_id=batch_sent_id,
                    sent_pos_oov=sent_pos_oov,
                    bert_pos_oov=bert_pos_oov,
                    tok_ids=tok_ids,
                    oov_token=sent_toks[sent_pos_oov]))

  batch_qty = len(batch_data_raw)
  tok_ids_batch = np.zeros((batch_qty, batch_max_seq_qty), dtype=np.int64)  # zero is a padding symbol
  tok_mask_batch = np.zeros((batch_qty, batch_max_seq_qty), dtype=np.int64)
  for k in range(batch_qty):
    tok_ids = batch_data_raw[k].tok_ids
    tok_qty = len(tok_ids)
    tok_ids_batch[k, 0:tok_qty] = tok_ids
    tok_mask_batch[k, 0:tok_qty] = np.ones(tok_qty)

  tok_ids_batch = torch.from_numpy(tok_ids_batch).to(device=torch_device)

  return batch_data_raw, tok_ids_batch, tok_mask_batch


BertPredProbs = namedtuple('BertPred', ['batch_sent_id', 'sent_pos_oov', 'bert_pos_oov', 'logits'])


def get_bert_preds_for_words_batch(torch_device, bert_model_mlm,
                                   batch_data_raw, tok_ids_batch, tok_mask_batch,  # comes from get_batch_data
                                   word_ids  # a list of IDS for which we generate logits
                                   ):
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
  tok_mask_batch = torch.from_numpy(np.array(tok_mask_batch, dtype=np.int64)).to(device=torch_device)

  weight = predictions.decoder.weight[word_ids, :]
  bias = predictions.bias[word_ids]

  # Transformations from the main BERT model
  sequence_output, _ = bert(tok_ids_batch, seg_ids, attention_mask=tok_mask_batch, output_all_encoded_layers=False)
  # Transformations from the BertLMPredictionHead model with the restricted last layer
  hidden_states = transform(sequence_output)
  logits = torch.nn.functional.linear(hidden_states, weight) + bias

  logits = logits.detach().cpu().numpy()

  res = []

  for k in range(batch_qty):
    e = batch_data_raw[k]
    bert_pos_oov = e.bert_pos_oov
    res.append(BertPredProbs(batch_sent_id=batch_data_raw[k].batch_sent_id,
                             bert_pos_oov=bert_pos_oov,
                             sent_pos_oov=e.sent_pos_oov,
                             logits=logits[k, bert_pos_oov]
                             )
               )

  return res


from scipy.special import softmax


# This functions add BERT scores to entries in the oov_scores
def add_bert_scores(bert_model_mlm, mini_batch_qty,
                    torch_device, bert_tokenizer,
                    oov_scores,
                    batch_data_raw, tok_ids_batch, tok_mask_batch):
  qty = len(batch_data_raw)
  assert (len(oov_scores) == qty)

  bert_ids_flat = []  # All BERT token IDs
  bert_ids_tok_start = []  # start offsets of bert BERT token IDs for each OOV token
  bert_ids_tok_qty = []  # number of BERT token IDs for each OOV token

  for query_res in oov_scores:
    for w in query_res:
      btok_ids = bert_tokenizer.convert_tokens_to_ids(bert_tokenizer.tokenize(w))
      bert_ids_tok_start.append(len(bert_ids_flat))
      bert_ids_tok_qty.append(len(btok_ids))
      bert_ids_flat.extend(btok_ids)

  # Predictions are named tuples like this one:
  # namedtuple('BertPred', ['batch_sent_id', 'sent_pos_oov', 'bert_pos_oov', 'logits'])
  pres = []
  bqty = int((qty + mini_batch_qty - 1) / mini_batch_qty)
  # ps = ''
  for bid in range(bqty):
    bs = bid * mini_batch_qty
    be = min(bs + mini_batch_qty, qty)
    # ps += f'{bid} {bqty} Mini-batch {bs} {be} {qty}\n'
    pres.extend(get_bert_preds_for_words_batch(torch_device,
                                               bert_model_mlm,
                                               batch_data_raw[bs:be], tok_ids_batch[bs:be], tok_mask_batch[bs:be],
                                               bert_ids_flat))
  # print(ps)

  query_start = 0
  wid = 0
  assert (len(oov_scores) == len(pres))
  for qid in range(len(oov_scores)):
    query_res = oov_scores[qid]
    all_logits = pres[qid].logits

    word_start = 0
    query_logits = []
    wid = 0
    for w in query_res:
      start = bert_ids_tok_start[wid]
      # There are often multiple BERT wid per single word
      lqty = bert_ids_tok_qty[wid]

      query_logits.append(np.mean(all_logits[start:start + lqty]))

      wid += 1

    softmaxes = softmax(query_logits)
    k = 0

    for w in query_res:
      query_res[w][BERT_SIMIL] = softmaxes[k]
      k += 1

    query_start += word_start


def toksToLower(inp):
  return [s.lower() for s in inp]

