#!/usr/bin/env python3
"""
Creates instances from processed JSONL that includes

tokens
word level features
sentence level features
word substitutions if as available
metadata (id, raw text)

optionally serializes the data

"""

import os, sys
import string
import numpy as np
import json
from typing import *
from overrides import overrides
from allennlp.data import Instance, Token
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, ArrayField, MetadataField

from basic_processing import MAX_SEQ_LEN, MemoryOptimizedTextField

from allennlp.training import TrainerWithCallbacks

class JigsawDatasetJSONReader(DatasetReader):

  def __init__(self, tokenizer: Callable[[str], List[str]] = lambda x: x.split(),
               token_indexers: Dict[str, TokenIndexer] = None,
               max_seq_len: Optional[int] = MAX_SEQ_LEN) -> None:
    super().__init__(lazy=False)
    self.tokenizer = tokenizer
    self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
    self.max_seq_len = max_seq_len

  @overrides
  def text_to_instance(self, tokens: List[str], id: str,
                       labels: np.ndarray, text: str) -> Instance:

    fields = {}

    sentence_field = MemoryOptimizedTextField([proc(x) for x in tokens],
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
               "label" : JigsawDatasetJSONReader.to_int_list(instance.get("label").array),
               "tokens": list(instance.get("tokens").tokens),
               "word_level_features" : JigsawDatasetJSONReader.to_float_list(instance.get("word_level_features").array),
               "sentence_level_features": JigsawDatasetJSONReader.to_float_list(instance.get("sentence_level_features").array),
               "text" : instance.get("text").metadata                  }

        objStr = json.dumps(obj)
        tf.write(objStr + '\n')