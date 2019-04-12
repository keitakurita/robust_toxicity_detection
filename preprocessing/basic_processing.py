"""
Perform basic text transformation

Replace real URL by “url” token (because some texts have nothing but urls)
Tokenize using the SpaCy model
Lower casing
Remove repeated symbols
Obtain FastText embeddings for tokens

NOTE: The process assumes that id is col0, comment text is col1

"""

import os, sys
import string
import argparse
import numpy as np
import fastText
import re
from itertools import groupby
import json
from typing import *
from overrides import overrides
from allennlp.data import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, ArrayField
import csv
from allennlp.data.vocabulary import Vocabulary

MAX_SEQ_LEN = 512

import spacy
nlp = spacy.load("en_core_web_sm",disable=['parser', 'ner'])


def replace_url(s):
  return re.sub(r"http\S+", "url", s)


def remove_extra_chars(s, max_qty=2):
  res = [c * min(max_qty, len(list(group_iter))) for c, group_iter in groupby(s)]
  return ''.join(res)

#TODO: clean up problematic substitutions by spaCy (e.g. "n't" that confuse BERT)
def tokenizer(x: str):
  doc = nlp(replace_url(x))
  toks = [token.text for token in doc]
  return toks

alphabet = set(string.ascii_lowercase)

sentence_level_features: List[Callable[[List[str]], float]] = [
#     lambda x: (np.log1p(len(x)) - 3.628) / 1.065, # stat computed on train set
]

word_level_features: List[Callable[[str], float]] = [
    lambda x: 1 if (x.lower() == x) else 0,
    lambda x: len([c for c in x.lower() if c not in alphabet]) / len(x),
    lambda x: 1 if (remove_extra_chars(x.lower()) == x.lower()) else 0
]

def proc(x: str) -> str:
    return remove_extra_chars(x.lower())

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
        self.tokens = None # empty tokens

class JigsawDatasetTransformer(DatasetReader):

  def __init__(self, tokenizer: Callable[[str], List[str]] = lambda x: x.split(),
               token_indexers: Dict[str, TokenIndexer] = None,  # TODO: Handle mapping from BERT
               max_seq_len: Optional[int] = MAX_SEQ_LEN) -> None:
    super().__init__(lazy=False)
    self.tokenizer = tokenizer
    self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
    self.max_seq_len = max_seq_len

  @overrides
  def text_to_instance(self, tokens: List[str], id: str,
                       labels: np.ndarray) -> Instance:

    fields = {"id" : id}

    sentence_field = MemoryOptimizedTextField([proc(x) for x in tokens],
                                              self.token_indexers)
    fields["tokens"] = sentence_field

    wl_feats = np.array([[func(w) for func in word_level_features] for w in tokens])
    fields["word_level_features"] = ArrayField(array=wl_feats)

    sl_feats = np.array([func(tokens) for func in sentence_level_features])
    fields["sentence_level_features"] = ArrayField(array=sl_feats)

    label_field = ArrayField(array=labels)
    fields["label"] = label_field

    return Instance(fields)

  @overrides
  def _read(self, file_path: str) -> Iterator[Instance]:
    with open(file_path) as f:
      reader = csv.reader(f)
      next(reader)
      for i, line in enumerate(reader):
        if len(line) == 9:
          _, id_, text, *labels = line
        elif len(line) == 8:
          id_, text, *labels = line
        else:
          raise ValueError(f"line has {len(line)} values")
        yield self.text_to_instance(
          self.tokenizer(text),
          id_, np.array([int(x) for x in labels]),
        )

  def save_to_file(self, file_path: str) -> None:

    for i, instance in enumerate(self):

      with open(file_path, 'w') as tf:

        obj = {"id": instance.get("id"),
               "labels" : instance.get("labels").array,
               "tokens": instance.get("tokens").tokens,
               "word_level_features" : instance.get("word_level_features").array,
               "sentence_level_features": instance.get("sentence_level_features").array
              }

        objStr = json.dumps(obj)
        tf.write(objStr + '\n')

def main(argv):

  parser = argparse.ArgumentParser(description='Basic Processing')
  parser.add_argument('--datapath', type=str,
                      required=False,
                      default = '',
                      help = '../data/jigsaw/')
  parser.add_argument('--rawtrain', type=str,
                      required = True,
                      default = 'train.csv',
                      help = 'Raw train file')
  parser.add_argument('--rawtest', type=str,
                      required = True,
                      default = 'test_proced.csv',
                      help = 'Raw test file')
  parser.add_argument('--ftmatname', type=str,
                      required = False,
                      default = 'ft_basic_toks',
                      help = 'Output train file')
  parser.add_argument('--vocabname', type=str,
                      required = False,
                      default = 'voc_basic_toks',
                      help = 'Output vocabulary file')
  parser.add_argument('--proctrain', type=str,
                      required = False,
                      default = 'train_basic.jsonl',
                      help = 'Output train file')
  parser.add_argument('--proctest', type=str,
                      required = False,
                      default = 'test_basic.jsonl',
                      help = 'Summary test file')

  args = parser.parse_args(argv)
  print(args)

  token_indexer = SingleIdTokenIndexer(
    lowercase_tokens=True,
  )

  transformer = JigsawDatasetTransformer(
    tokenizer=tokenizer,
    token_indexers={"tokens": token_indexer}
  )

  train_ds = transformer.read(os.path.join(args.datapath,args.rawtrain))
  test_ds = transformer.read(os.path.join(args.datapath,args.rawtest))
  full_ds = train_ds + test_ds

  train_ds.save_to_file(os.path.join(args.datapath, args.proctrain))
  test_ds.save_to_file(os.path.join(args.datapath, args.proctest))

  vocab = Vocabulary.from_instances(full_ds)
  vocab.save_to_files(os.path.join(args.datapath, args.vocabname))

  ft_model = fastText.load_model(os.path.join(args.datapath, "wiki.en.bin"))
  ft_emb = []
  with (os.path.join(args.datapath, args.ftmatname+".txt")).open("wt") as f:
    for idx, token in vocab.get_index_to_token_vocabulary().items():
      emb = ft_model.get_word_vector(token)
      emb_as_str = " ".join(["%.4f" % x for x in emb])
      ft_emb.append(np.array(emb))
      f.write(f"{token} {emb_as_str}\n")

  ft_emb = np.vstack(ft_emb)
  np.save(os.path.join(args.datapath,  args.ftmatname+".npy"), ft_emb)

if __name__ == '__main__':
  main(sys.argv[1:])