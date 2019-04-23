#!/usr/bin/env python3
"""
Perform basic text transformation

Replace real URL by 'url' token (because some texts have nothing but urls)
Tokenize using the SpaCy model
Preserve casing
Remove repeated symbols
Obtain FastText embeddings for tokens

NOTE: The process assumes that id is col0, comment text is col1

"""

import os, sys
import string
import argparse
import fastText
import re
from itertools import groupby
import json
from allennlp.data import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import  ArrayField, MetadataField
import csv
from allennlp.data.vocabulary import Vocabulary
from preprocessing_common import *
import spacy

from typing import *
from overrides import overrides
from itertools import chain
import numpy as np


class SpacyTokenizer:

  def __init__(self, spacy_nlp):
    self.pat_part = re.compile(r'^[a-z]{0,2}\'[a-z]{1,2}$', flags=re.IGNORECASE)
    self.spacy_nlp = spacy_nlp

  def replace_url(s):
    return re.sub(r"http\S+", "url", s)

  def __call__(self, text):

    toks0 = [token.text for token in self.spacy_nlp(SpacyTokenizer.replace_url(text))]

    toks1 = [token for token in toks0 if len(token.strip())>0]

    toks2 = []

    for i in range(len(toks1)):
      s = toks1[i]
      if i > 0 and self.pat_part.match(s) is not None:
        arr = s.split("'")
        toks2[-1] += arr[0]
        toks2.append("'")
        toks2.append(arr[1])
      else:
        toks2.append(s)

    return toks2

spacy_nlp = spacy.load(SPACY_MODEL_TYPE, disable=['parser', 'ner', 'pos'])
tok_obj = SpacyTokenizer(spacy_nlp)

def tokenizer(x: str):
  return tok_obj(x)

def remove_extra_chars(s, max_qty=2):
  res = [c * min(max_qty, len(list(group_iter))) for c, group_iter in groupby(s)]
  return ''.join(res)

alphabet = set(string.ascii_lowercase)

sentence_level_features: List[Callable[[List[str]], float]] = [
#     lambda x: (np.log1p(len(x)) - 3.628) / 1.065, # stat computed on train set
]

word_level_features: List[Callable[[str], float]] = [
    lambda x: len([c for c in x if c.lower() != c]) / len(x),
    lambda x: len([c for c in x.lower() if c not in alphabet]) / len(x),
    lambda x: 1 if (remove_extra_chars(x.lower()) == x.lower()) else 0
]


class TokenTransfomer:

  def __init__(self, nlp):

    self.case_map_dict = get_canon_case_map(nlp)

  def proc(self, s: str) -> str:
    x = remove_extra_chars(s.lower())
    if x in self.case_map_dict:
      # Get a canonical case version
      x = self.case_map_dict[x]
      # If the word is all capital, we want to preserve all upper case
      if s == s.upper():
        x = x.upper()
      # If the first letter is upper-cased, we want to preserve this too
      elif s[0] != s[0].lower():
        x = x[0].upper() + x[1:]

      return x

    return remove_extra_chars(s)




class JigsawDatasetTransformer(DatasetReader):

  def __init__(self,
               tok_transf: TokenTransfomer,
               tokenizer: Callable[[str], List[str]] = lambda x: x.split(),
               token_indexers: Dict[str, TokenIndexer] = None, 
               max_seq_len: Optional[int] = MAX_SEQ_LEN) -> None:
    super().__init__(lazy=False)
    self.tokenizer = tokenizer
    self.tok_transf = tok_transf
    self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
    self.max_seq_len = max_seq_len
    self.headers = None

  @overrides
  def text_to_instance(self, tokens: List[str], id: str,
                       labels: np.ndarray, text: str) -> Instance:

    fields = {}

    sentence_field = MemoryOptimizedTextField([self.tok_transf.proc(x) for x in tokens],
                                               self.token_indexers)
    fields["tokens"] = sentence_field

    wl_feats = np.array([[func(w) for func in word_level_features] for w in tokens])
    fields["word_level_features"] = ArrayField(array=wl_feats)

    sl_feats = np.array([func(tokens) for func in sentence_level_features])
    fields["sentence_level_features"] = ArrayField(array=sl_feats)

    label_field = ArrayField(array=labels)
    fields["label"] = label_field

    fields["text"] =  MetadataField(text)

    fields["id"] = MetadataField(id)


    return Instance(fields)

  @overrides
  def _read(self, file_path: str) -> Iterator[Instance]:
    with open(file_path) as f:

      reader = csv.reader(f)
      self.headers = next(reader)

      for i, line in enumerate(reader):
        if len(line) == 9:
          _, id_, text, *labels = line
        elif len(line) == 8:
          id_, text, *labels = line
        else:
          raise ValueError(f"line has {len(line)} values")


        text = text.replace('\n', ' ').replace('\r', ' ')

        yield self.text_to_instance(
          self.tokenizer(text),
          id_, np.array([int(x) for x in labels]), text 
        )

  @staticmethod
  def to_int_list(arr):
    if len(arr.shape) == 1:
      return [int(x) for x in arr]
    elif len(arr.shape) == 2:
      return [ [int(x) for x in e] for e in arr]

  @staticmethod
  def to_float_list(arr):
    if len(arr.shape) == 1:
      return [float(x) for x in arr]
    elif len(arr.shape) == 2:
      return [ [float(x) for x in e] for e in arr]

  @staticmethod
  def save_to_file(data, file_path: str) -> None:

    with open(file_path, 'w') as tf:
      for _, instance in enumerate(data):

        obj = {"id": instance.get("id").metadata,
               "label" : JigsawDatasetTransformer.to_int_list(instance.get("label").array),
               "tokens": list(instance.get("tokens").tokens),
               "word_level_features" : JigsawDatasetTransformer.to_float_list(instance.get("word_level_features").array),
               "sentence_level_features": JigsawDatasetTransformer.to_float_list(instance.get("sentence_level_features").array),
               "text" : instance.get("text").metadata                  }


        objStr = json.dumps(obj)
        tf.write(objStr + '\n')

  @staticmethod
  def save_to_csv(data, file_path: str, headers: List[str]) -> None:

    with open(file_path, 'w',newline='', encoding='utf-8') as csv_file:
      writer = csv.writer(csv_file, delimiter=',')
      writer.writerow(headers)

      for _, instance in enumerate(data):

        row = [instance.get("id").metadata,
               ' '.join(list(instance.get("tokens").tokens)),
               JigsawDatasetTransformer.to_int_list(instance.get("label").array)]

        writer.writerow(row)


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
  parser.add_argument('--ftmodelpath', type=str,
                      required = False,
                      default = 'wiki.en.bin',
                      help = 'Summary test file')

  args = parser.parse_args(argv)
  print(args)

  spacy_ds = get_spacy_vocab_instances(spacy_nlp)
  vocab = Vocabulary.from_instances(spacy_ds)
  print('Spacy vocabulary:', vocab)

  token_indexer = SingleIdTokenIndexer(
    lowercase_tokens=True,
  )

  transformer = JigsawDatasetTransformer(
    TokenTransfomer(tok_obj.spacy_nlp),
    tokenizer=tokenizer,
    token_indexers={"tokens": token_indexer}
  )

  train_ds = transformer.read(os.path.join(args.datapath,args.rawtrain))
  test_ds = transformer.read(os.path.join(args.datapath,args.rawtest))
  full_ds = list(train_ds) + list(test_ds)

  transformer.save_to_file(train_ds, os.path.join(args.datapath, args.proctrain))
  transformer.save_to_file(test_ds, os.path.join(args.datapath, args.proctest))

  # Need to re-init spacy from scratch
  spacy_ds = get_spacy_vocab_instances(spacy_nlp)
  vocab = Vocabulary.from_instances(chain(spacy_ds, spacy_ds, full_ds))
  vocab.save_to_files(os.path.join(args.datapath, args.vocabname))

  ft_model = fastText.load_model(args.ftmodelpath)

  with open(os.path.join(args.datapath, args.ftmatname+".txt"),"wt") as f:
    for idx, token in vocab.get_index_to_token_vocabulary().items():
      token = token.strip()
      if token:
        emb = ft_model.get_word_vector(token)
        emb_as_str = " ".join(["%.4f" % x for x in emb])
        f.write(f"{token} {emb_as_str}\n")

if __name__ == '__main__':
  main(sys.argv[1:])
