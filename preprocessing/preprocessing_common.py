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

JIGSAW_LABEL_NAMES = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

MAX_SEQ_LEN = 512

NON_WORDS = ['@@UNKNOWN@@', '@@PADDING@@']

SPACY_MODEL_TYPE = "en_core_web_sm"

TokenList = List[Token]


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


def read_embeds_and_words_subset(file_name, word_map):
  word_list, embed_list = [], []

  with open(file_name, encoding="utf8") as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      fld = line.split()
      w = fld[0]
      if w in word_map:
        word_list.append(word_map[w])
        embed_list.append(np.array([float(s) for s in fld[1:]]))

  return word_list, np.vstack(embed_list)


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
  print('Index-time parameters', index_time_params)
  print('Indexing time = %f' % (end - start))


  return index

def query_index(index, K, query_arr, num_threads=0, efS=200):
  # Querying
  query_time_params = {'efSearch': efS}
  print('Setting query-time parameters', query_time_params)
  index.setQueryTimeParams(query_time_params)

  query_qty = len(query_arr)
  start = time.time()
  res = index.knnQueryBatch(query_arr, k=K, num_threads=num_threads)
  end = time.time()
  print('kNN time total=%f (sec), per query=%f (sec), per query adjusted for thread number=%f (sec)' %
        (end - start, float(end - start) / query_qty, num_threads * float(end - start) / query_qty))

  return res


