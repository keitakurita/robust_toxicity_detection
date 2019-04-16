#NOTE spaCy version 2.0.18
import nmslib
import time
import numpy as np
from allennlp.data.fields import TextField
from allennlp.data.token_indexers import TokenIndexer
from typing import *
from overrides import overrides
from allennlp.data import  Token

JIGSAW_LABEL_NAMES = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

MAX_SEQ_LEN = 512

NON_WORDS = ['@@UNKNOWN@@', '@@PADDING@@']

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


def read_embeds_and_words_subset(file_name, word_map):
  word_list, embed_list = [], []

  with open(file_name, encoding="utf8") as f:
    for line in f:
      line = line.strip()
      fld = line.split()
      w = fld[0]
      if w in word_map:
        word_list.append(word_map[w])
        embed_list.append(np.array([float(s) for s in fld[1:]]))

  return word_list, np.vstack(embed_list)


def create_embeds_index(embeds, M = 30, efC = 200, efS = 200):
  num_threads = 0
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

  # Setting query-time parameters
  query_time_params = {'efSearch': efS}
  print('Setting query-time parameters', query_time_params)
  index.setQueryTimeParams(query_time_params)

  return index


def create_word_index(words, M = 30, efC = 200, efS = 200):
  num_threads = 0
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

  # Setting query-time parameters
  query_time_params = {'efSearch': efS}
  print('Setting query-time parameters', query_time_params)
  index.setQueryTimeParams(query_time_params)

  return index
