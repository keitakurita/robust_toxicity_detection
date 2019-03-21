#!/usr/bin/env python
# coding: utf-8

save_template = False
load_template = True
TEMPLATE_NAME = 'template.mdl'

# In[1]:


depends_on = [
    "preproc_jigsaw",
    "jigsaw_create_augmented_data",
    "create_fasttext_matrix",
]


# In[2]:


#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')


# In[3]:


# settings for seamlessly running on colab
import os
try:
    from google.colab import drive
    drive.mount('/content/gdrive')
    os.environ["IS_COLAB"] = "True"
except ImportError:
    os.environ["IS_COLAB"] = "False"    


# In[4]:


if "SLACK_TOKEN" not in os.environ:
    os.environ["SLACK_TOKEN"] = "" # TODO: insert here for slack notifications
if "SLACK_ID" not in os.environ:
    os.environ["SLACK_ID"] = "" # TODO: insert here for slack notifications


# In[5]:


#get_ipython().run_cell_magic('bash', '', 'if [ "$IS_COLAB" = "True" ]; then\n    pip install git+https://github.com/facebookresearch/fastText.git\n    pip install torch\n    pip install torchvision\n    pip install allennlp\n    pip install dnspython\n    pip install jupyter_slack\n    pip install git+https://github.com/keitakurita/Better_LSTM_PyTorch.git\nfi')


# In[6]:


from pathlib import Path
from typing import *
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from functools import partial
from overrides import overrides
import warnings

from allennlp.data import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.nn import util as nn_util

import logging
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


# In[7]:


import time
from contextlib import contextmanager

class Config(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    def set(self, key, val):
        self[key] = val
        setattr(self, key, val)

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')
    
import functools
import traceback
import sys

def get_ref_free_exc_info():
    "Free traceback from references to locals/globals to avoid circular reference leading to gc.collect() unable to reclaim memory"
    type, val, tb = sys.exc_info()
    traceback.clear_frames(tb)
    return (type, val, tb)

def gpu_mem_restore(func):
    "Reclaim GPU RAM if CUDA out of memory happened, or execution was interrupted"
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except:
            type, val, tb = get_ref_free_exc_info() # must!
            raise type(val).with_traceback(tb) from None
    return wrapper

def ifnone(a: Any, alt: Any): return alt if a is None else a


# Custom Types

# In[8]:


T = TypeVar("T")
TensorDict = Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]  # pylint: disable=invalid-name


# In[9]:


# for papermill
testing = True # set to False when running experiments
debugging = False
seed = 1
use_bt = False
computational_batch_size = 128
batch_size = 128
lr = 2e-3
lr_schedule = "slanted_triangular"
epochs = 6 if not testing else 1
hidden_sz = 128
dataset = "jigsaw"
n_classes = 6
max_seq_len = 512
download_data = False
ft_model_path = "../data/jigsaw/ft_model.txt"
max_vocab_size = 400000
dropouti = 0.2
dropoutw = 0.0
dropoute = 0.1
dropoute_max = None
dropoutr = 0.3 # TODO: Implement
val_ratio = 0.0
use_focal_loss = False
focal_loss_alpha = 1.
focal_loss_gamma = 2.
use_augmented = False
freeze_embeddings = True
mixup_ratio = 0.0
discrete_mixup_ratio = 0.0
attention_bias = True
use_attention_aux = False
weight_decay = 0.
bias_init = True
neg_splits = 1
num_layers = 2
rnn_type = "lstm"
rnn_residual = False
pooling_type = "augmented_multipool" # attention or multipool or augmented_multipool
model_type = "elmo"
cache_elmo_embeddings = True
use_word_level_features = False
use_sentence_level_features = False
bucket = True
compute_thres_on_test = True
find_lr = False
permute_sentences = False
run_id = "ELMO_tuning_0"


# In[10]:


# TODO: Can we make this play better with papermill?
config = Config(
    testing=testing,
    debugging=debugging,
    seed=seed,
    use_bt=use_bt,
    computational_batch_size=computational_batch_size,
    batch_size=batch_size,
    lr=lr,
    lr_schedule=lr_schedule,
    epochs=epochs,
    hidden_sz=hidden_sz,
    dataset=dataset,
    n_classes=n_classes,
    max_seq_len=max_seq_len, # necessary to limit memory usage
    ft_model_path=ft_model_path,
    max_vocab_size=max_vocab_size,
    dropouti=dropouti,
    dropoutw=dropoutw,
    dropoute=dropoute,
    dropoute_max=dropoute_max,
    dropoutr=dropoutr,
    val_ratio=val_ratio,
    use_focal_loss=use_focal_loss,
    focal_loss_alpha=focal_loss_alpha,
    focal_loss_gamma=focal_loss_gamma,
    use_augmented=use_augmented,
    freeze_embeddings=freeze_embeddings,
    attention_bias=attention_bias,
    use_attention_aux=use_attention_aux,
    weight_decay=weight_decay,
    bias_init=bias_init,
    neg_splits=neg_splits,
    num_layers=num_layers,
    rnn_type=rnn_type,
    rnn_residual=rnn_residual,
    pooling_type=pooling_type,
    model_type=model_type,
    cache_elmo_embeddings=cache_elmo_embeddings,
    use_word_level_features=use_word_level_features,
    use_sentence_level_features=use_sentence_level_features,
    mixup_ratio=mixup_ratio,
    discrete_mixup_ratio=discrete_mixup_ratio,
    bucket=bucket,
    compute_thres_on_test=compute_thres_on_test,
    permute_sentences=permute_sentences,
    find_lr=find_lr,
    run_id=run_id,
)


# In[11]:


from allennlp.common.checks import ConfigurationError


# In[12]:


if config.model_type != "standard" and "bert" not in config.model_type and "elmo" not in config.model_type:
    raise ConfigurationError(f"Invalid model type {config.model_type} given")


# In[13]:


if config.mixup_ratio > 0. and config.bucket:
    raise ConfigurationError(f"Mixup should be combined with complete random shuffling of the input")


# In[14]:


if "bert" in config.model_type and config.computational_batch_size > 16:
    raise ConfigurationError("Batch size too large for BERT")


# In[15]:


import datetime
now = datetime.datetime.now()
RUN_ID = config.run_id if config.run_id is not None else now.strftime("%m_%d_%H:%M:%S")


# In[16]:


USE_GPU = torch.cuda.is_available()


# In[17]:


DATA_ROOT = Path("../data") / config.dataset
SER_DIR = DATA_ROOT / "ckpts" / RUN_ID

print('Serialization dir:' + str(SER_DIR))


# In[18]:


#get_ipython().system('mkdir -p {DATA_ROOT}')


# In[19]:


import subprocess
if download_data:
    if config.val_ratio > 0.0:
        fnames = ["train_wo_val.csv", "test_proced.csv", "val.csv", "ft_model.txt"]
    else:
        fnames = ["train.csv", "test_proced.csv", "ft_model.txt"]
    if config.use_augmented or config.discrete_mixup_ratio > 0.0: fnames.append("train_extra.csv")
    for fname in fnames:
        if not (DATA_ROOT / fname).exists():
            print(subprocess.Popen([f"aws s3 cp s3://nnfornlp/raw_data/jigsaw/{fname} {str(DATA_ROOT)}"],
                                   shell=True, stdout=subprocess.PIPE).stdout.read())


# In[20]:


#get_ipython().system('ls {DATA_ROOT}')


# In[ ]:





# Set random seed manually to replicate results

# In[21]:


torch.manual_seed(config.seed)


# In[ ]:





# # Load Data

# In[22]:


from allennlp.data.dataset_readers import DatasetReader, StanfordSentimentTreeBankDatasetReader


# In[ ]:





# ### Prepare dataset

# In[23]:


reader_registry = {}


# In[24]:


def register(name: str):
    def dec(x: Callable):
        reader_registry[name] = x
        return x
    return dec


# In[25]:


label_cols = ["toxic", "severe_toxic", "obscene",
              "threat", "insult", "identity_hate"]

from enum import IntEnum
ColIdx = IntEnum('ColIdx', [(x.upper(), i) for i, x in enumerate(label_cols)])


# In[26]:


import csv
import gc

from allennlp.data.fields import TextField, SequenceLabelField, LabelField, MetadataField, ArrayField
import string
alphabet = set(string.ascii_lowercase)

sentence_level_features: List[Callable[[List[str]], float]] = [
#     lambda x: (np.log1p(len(x)) - 3.628) / 1.065, # stat computed on train set
]

word_level_features: List[Callable[[str], float]] = [
    lambda x: 1 if (x.lower() == x) else 0,
    lambda x: len([c for c in x.lower() if c not in alphabet]) / len(x),
]

def proc(x: str) -> str:
    if config.model_type == "standard" or "uncased" in config.model_type:
        return x.lower()
    else:
        return x

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


# In[27]:


@register("jigsaw")
class JigsawDatasetReader(DatasetReader):
    def __init__(self, tokenizer: Callable[[str], List[str]]=lambda x: x.split(),
                 token_indexers: Dict[str, TokenIndexer] = None, # TODO: Handle mapping from BERT
                 max_seq_len: Optional[int]=config.max_seq_len) -> None:
        super().__init__(lazy=False)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.max_seq_len = max_seq_len

    @overrides
    def text_to_instance(self, tokens: List[str], id: str,
                         labels: np.ndarray) -> Instance:
        sentence_field = MemoryOptimizedTextField([proc(x) for x in tokens],
                                   self.token_indexers)
        fields = {"tokens": sentence_field}
        
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
                else: raise ValueError(f"line has {len(line)} values")
                yield self.text_to_instance(
                    self.tokenizer(text),
                    id_, np.array([int(x) for x in labels]),
                )
                if config.testing and i == 1000: break


# In[ ]:





# In[ ]:





# ### Prepare token handlers

# In[28]:


import random
from functools import wraps

def maybeshuffle(_tokenize):
    def func(*args, **kwargs):
        arr = _tokenize(*args, **kwargs)
        if config.permute_sentences:
            random.shuffle(arr)
        return arr
    return func


# In[29]:


from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.data.token_indexers import WordpieceIndexer, SingleIdTokenIndexer

_spacy_tok = SpacyWordSplitter(language='en_core_web_sm', pos_tags=False).split_words

if config.model_type == "standard" or ("elmo" in config.model_type and config.cache_elmo_embeddings):
    from allennlp.data.token_indexers import SingleIdTokenIndexer
    token_indexer = SingleIdTokenIndexer(
        lowercase_tokens="elmo" not in config.model_type,
    )
    @maybeshuffle
    def tokenizer(x: str):
        return [w.text for w in
                _spacy_tok(x)[:config.max_seq_len]]
elif "elmo" in config.model_type:
    from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer
    token_indexer = ELMoTokenCharactersIndexer()
    @maybeshuffle
    def tokenizer(x: str):
        # add start and end of sentence tokens
        return ["<S>"] + [w.text for w in
                _spacy_tok(x)[:config.max_seq_len - 2]] + ["</S>"]
elif "bert" in config.model_type:
    def flatten(x: List[List[T]]) -> List[T]:
        return [item for sublist in x for item in sublist]

    from allennlp.data.token_indexers import PretrainedBertIndexer
    token_indexer = PretrainedBertIndexer(
        pretrained_model=config.model_type,
        max_pieces=config.max_seq_len,
        do_lowercase=True,
     )
    # apparently we need to truncate the sequence here, which is a stupid design decision
    @maybeshuffle
    def tokenizer(s: str):
        return [w.text for w in _spacy_tok(s)]


# In[30]:


reader = JigsawDatasetReader(
    tokenizer=tokenizer,
    token_indexers={"tokens": token_indexer}
)


# In[31]:


if config.val_ratio > 0.0:
    train_ds, val_ds, test_ds = (reader.read(DATA_ROOT / fname) for fname in ["train_wo_val.csv",
                                                                              "val.csv",
                                                                              "test_proced.csv"])
else:
    train_ds, test_ds = (reader.read(DATA_ROOT / fname) for fname in [
        "train_with_bt.csv" if config.use_bt else "train.csv",
      "test_proced.csv"])


# In[32]:


if config.use_augmented or config.discrete_mixup_ratio > 0.0:
    # TODO: Handle data leak for validation!
    train_aug_ds = reader.read(DATA_ROOT / "train_extra.csv")


# In[33]:


len(train_ds)


# In[34]:


vars(train_ds[0].fields["tokens"])


# In[ ]:





# ### Prepare labels

# In[35]:


if config.val_ratio > 0.0:
    train_labels = pd.read_csv(DATA_ROOT / "train_wo_val.csv")[label_cols].values
else:
    train_labels = pd.read_csv(DATA_ROOT / "train.csv")[label_cols].values
if config.testing: train_labels = train_labels[:len(train_ds), :]
if config.use_augmented:
    train_aux_labels = pd.read_csv(DATA_ROOT / "train_extra.csv")[label_cols].values
    if config.testing: train_aux_labels = train_aux_labels[:len(train_ds), :]


# ### Prepare vocabulary

# In[36]:


from allennlp.data.vocabulary import Vocabulary
if "bert" in config.model_type:
    vocab = Vocabulary()
elif config.model_type == "standard" or config.cache_elmo_embeddings:
    full_ds = train_ds + test_ds
    if config.val_ratio > 0.0: full_ds = full_ds + val_ds
    vocab = Vocabulary.from_instances(full_ds, max_vocab_size=config.max_vocab_size)
else:
    vocab = Vocabulary()


# In[ ]:





# ### Prepare iterator

# In[37]:


from allennlp.data.iterators import BucketIterator, DataIterator


# In[38]:


from sklearn.model_selection import KFold

class Sampler:
    def sample(self, ds: List[Instance]) -> List[Instance]:
        return ds
    def sample_size(self, ds: List[Instance]) -> int:
        return len(ds)

class BiasedSampler(Sampler):
    def __init__(self, mask: np.ndarray, n_splits: int):
        self.mask = mask
        self.n_splits = n_splits
        self.pos = np.where(self.mask)[0]
        self.neg = np.where(~self.mask)[0]
        self._n_splits_iterated = 0
        
    def sample(self, ds: List[Instance]) -> List[Instance]:
        if self._n_splits_iterated % self.n_splits == 0:
            self.folds = KFold(n_splits=self.n_splits).split(self.neg)
        _, neg_idxs = next(self.folds)
        
        p = np.random.permutation(len(self.pos) + len(neg_idxs))
        smpl = np.r_[self.pos, self.neg[neg_idxs]][p]
        
        self._n_splits_iterated += 1
        return [ds[i] for i in smpl]
    
    def sample_size(self, ds: List[Instance]) -> int:
        """Returns number of samples that would be returned upon a call to sample"""
        # there might be a slight difference depending on the epoch, but it's okay
        return len(self.pos) + len(self.neg) // self.n_splits 


# In[39]:


class ScoredSampler:
    def __init__(self, mask: np.ndarray, ratio: float):
        self.mask = mask
        self.ratio = ratio
        self.n_samples = int(len(self.tgt) * self.ratio)
        self.score = mask.astype("int")
    
    def set_score(self, score: np.ndarray):
        assert len(score) == len(self.tgt)
        self.score = score
    
    def sample(self, ds: List[Instance]):
        """Sample top n targets sorted by score descending"""
        smpl = np.arange(len(self.mask))[np.argsort(-self.score)][:self.n_samples]
        return [ds[i] for i in smpl]
    
    def sample_size(self, ds: List[Instance]) -> int:
        """Returns number of samples that would be returned upon a call to sample"""
        return self.n_samples


# In[40]:


import random
from collections import deque
from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.common.util import lazy_groups_of, add_noise_to_dict_values
from allennlp.data.dataset import Batch
from allennlp.data.instance import Instance
from allennlp.data.iterators import DataIterator, BucketIterator, BasicIterator
from allennlp.data.vocabulary import Vocabulary

class SamplingIteratorMixin:
    """Uses Python's MRO to add sampling.
    DANGER: This is pushing the limits of OOP and might lead to bugs
    """
    def __init__(self, *args, sampler: Sampler=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.sampler = ifnone(sampler, Sampler())
        
    def get_num_batches(self, instances: List[Instance]):
        return math.ceil(self.sampler.sample_size(instances) / self._batch_size)

    def _create_batches(self, instances: Iterable[Instance], shuffle: bool) -> Iterable[Batch]:
        yield from super()._create_batches(self.sampler.sample(instances), shuffle)

# Caution: Inheritance must be in order: SamplingIteratorMixin, BucketIterator
class CustomBucketIterator(SamplingIteratorMixin, BucketIterator): pass
class CustomBasicIterator(SamplingIteratorMixin, BasicIterator): pass


# In[41]:


# TODO: Allow for customization
if config.neg_splits > 1:
    if config.use_augmented:
        full_trn_labels = np.concatenate([train_labels, train_aux_labels], axis=0)
    else:
        full_trn_labels = train_labels
    sampler = BiasedSampler(full_trn_labels.sum(1) >= 1,
                            config.neg_splits)
else:
    sampler = Sampler()
if config.bucket:
    iterator = CustomBucketIterator(
        batch_size=config.computational_batch_size, 
        biggest_batch_first=config.testing,
        sorting_keys=[("tokens", "num_tokens")],
        max_instances_in_memory=config.batch_size * 2,
        sampler=sampler,
    )
else:
    # CAUTION: BasicIterator shuffles the dataset internally
    # TODO: Either fix this bug or ensure evalutation can handle shuffle
    # in the dataset order
    iterator = CustomBasicIterator(
        batch_size=config.batch_size, 
        max_instances_in_memory=config.batch_size * 2,
        sampler=sampler,
    )
iterator.index_with(vocab)


# In[ ]:





# ### Read sample

# In[42]:


batch = next(iter(iterator(train_ds)))


# In[ ]:


batch


# In[ ]:


batch["tokens"]["tokens"]


# In[ ]:


batch["tokens"]["tokens"].shape


# In[ ]:





# # Prepare Model

# In[ ]:


import torch
import torch.nn as nn
import torch.optim as optim


# In[ ]:


from allennlp.modules.token_embedders import Embedding
from allennlp.modules.token_embedders.bert_token_embedder import BertEmbedder, PretrainedBertEmbedder
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, PytorchSeq2VecWrapper
from allennlp.modules.stacked_bidirectional_lstm import StackedBidirectionalLstm
from allennlp.nn.util import get_text_field_mask


# In[ ]:


class Attention(Seq2VecEncoder):
    def __init__(self, inp_sz, aug_sz=None,
                 hidden_sz=None, out_sz=None, dim=1, eps=1e-9,
                 return_attention=False, use_bias=True):
        super().__init__()
        self.inp_sz, self.dim, self.eps = inp_sz, dim, eps
        self.out_sz = ifnone(out_sz, self.inp_sz)
        self.return_attention = return_attention
        self.l1 = nn.Linear(inp_sz, ifnone(inp_sz * 2, hidden_sz))
        nn.init.xavier_uniform_(self.l1.weight.data)
        nn.init.zeros_(self.l1.bias.data)
        
        vw = torch.zeros(ifnone(inp_sz * 2, hidden_sz), 1)
        nn.init.xavier_uniform_(vw)        
        self.vw = nn.Parameter(vw)
        self.use_bias = use_bias
        if self.use_bias: self.b = nn.Parameter(torch.zeros(1))
    
    @overrides
    def get_input_dim(self) -> int:
        return self.inp_sz
    
    @overrides
    def get_output_dim(self) -> int:
        return self.out_sz
        
    def forward(self, x, aug=None, mask=None):
        e = torch.tanh(self.l1(x))
        e = torch.einsum("bij,jk->bi", [e, self.vw]) 
        if self.use_bias: e = e + self.b
        a = torch.exp(e)
        
        if mask is not None: a = a.masked_fill(mask == 0, 0)

        a = a / (torch.sum(a, dim=self.dim, keepdim=True) + self.eps)

        weighted_input = x * a.unsqueeze(-1)
        if self.return_attention:
            return torch.sum(weighted_input, dim=1), a
        else:
            return torch.sum(weighted_input, dim=1)


# In[ ]:


from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper, Seq2SeqEncoder
from better_lstm import LSTM, VariationalDropout

class MultiPooling(Seq2VecEncoder):
    """Does max and mean pooling over the temporal dimension"""
    def __init__(self, input_sz: int):
        super().__init__()
        self.input_sz = input_sz
        
    @overrides
    def get_input_dim(self) -> int:
        return self.input_sz
    
    @overrides
    def get_output_dim(self) -> int:
        return self.input_sz * 2
        
    def forward(self, x, mask=None, aug=None):
        max_, _ = torch.max(x, dim=1)
        mean_ = torch.mean(x, dim=1)
        return torch.cat([max_, mean_], dim=-1)

class AugmentedMultiPool(MultiPooling):
    def __init__(self, input_sz, aug_sz):
        super().__init__(input_sz)
        self.attn = Attention(input_sz, hidden_sz=input_sz, 
                              out_sz=input_sz)
    @overrides
    def get_output_dim(self) -> int:
        return self.input_sz * 3
    
    def forward(self, x, mask=None, aug=None):
        pooled = super().forward(x, mask=mask, aug=aug)
        attn = self.attn(x, mask=mask, aug=None)
        return torch.cat([pooled, attn], dim=-1)
    
class BiRNN(Seq2SeqEncoder):
    def __init__(self, rnn_type, n_layers, embed_sz, hidden_sz, dropoutw=0.):
        super().__init__()
        self.rnn_type = rnn_type
        self.n_layers = n_layers
        self.embed_sz = embed_sz
        self.hidden_sz = hidden_sz
        in_szs = [embed_sz] + [hidden_sz * 2] * (n_layers - 1)
        if rnn_type == "lstm":
            rnns = [LSTM(in_sz, hidden_sz, batch_first=True, num_layers=1,
                         bidirectional=True, dropoutw=dropoutw)
                    for in_sz in in_szs]
        else:
            if dropoutw > 0.0:
                warnings.warn("Weight dropout not currently supported with GRUs")
            rnns = [nn.GRU(in_sz, hidden_sz, batch_first=True, num_layers=1, 
                           bidirectional=True)
                    for in_sz in in_szs]
            for gru in rnns:
                for name, param in gru.named_parameters():
                    if "weight_hh" in name:
                        nn.init.orthogonal_(param.data)
                    elif "weight_ih" in name:
                        nn.init.xavier_uniform_(param.data)
                    elif "bias" in name:
                        nn.init.zeros_(param.data)
        self.rnns = nn.ModuleList([PytorchSeq2SeqWrapper(rnn) for rnn in rnns])
        if config.use_attention_aux:
            self.ln = nn.Linear(embed_sz, 64) # handle attention auxillary input here

    @overrides
    def get_input_dim(self) -> int:
        return self.embed_sz
    
    @overrides
    def get_output_dim(self) -> int:
        if config.rnn_residual:
            out_sz = self.hidden_sz * 2 * self.n_layers
        else:
            out_sz = self.hidden_sz * 2
        if config.use_attention_aux: out_sz += 64
        return out_sz
    
    def forward(self, embeds, mask=None):
        x = embeds
        outputs = []
        for rnn in self.rnns:
            x = rnn(x, mask=mask)
            if config.rnn_residual:
                outputs.append(x)
        if config.rnn_residual:
            x = torch.cat(outputs, dim=-1)
        #else:
        #    x = outputs[-1]
        if config.use_attention_aux:
            x = torch.cat([torch.tanh(self.ln(embeds)), x], dim=-1)
        return x
    
class BiRNNEncoder(Seq2VecEncoder):
    def __init__(self, rnn: Seq2SeqEncoder,
                 pooler: Seq2VecEncoder,
                 dropouti=0.0, dropoutr=0.0):
        super().__init__()
        self.dropouti = VariationalDropout(dropouti, batch_first=True)
        self.rnn = rnn
        self.dropouto = VariationalDropout(dropoutr, batch_first=True)
        self.pool = pooler
        
    @overrides
    def get_input_dim(self) -> int:
        return self.rnn.get_input_dim()
    
    @overrides
    def get_output_dim(self) -> int:
        out_dim = self.pool.get_output_dim()
        if config.use_sentence_level_features:
            out_dim += len(sentence_level_features)
        return out_dim
    
    def _init_hidden_state(self, bs:int):
        if self.rnn.rnn_type == "lstm":
            return torch.zeros(bs, self.hidden_sz), torch.zeros(bs, self.hidden_sz)
        else:
            return torch.zeros(bs, self.hidden_sz)
    
    @overrides
    def forward(self, x: torch.Tensor, sentence_feats: torch.Tensor,
                mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        seq = self.rnn(x, mask)
        seq = self.dropouto(seq)
        vec = self.pool(seq, aug=x, mask=mask)
        if config.use_sentence_level_features:
            return torch.cat([sentence_feats, vec], dim=-1)
        else:
            return vec


# In[ ]:


from allennlp.training.metrics import CategoricalAccuracy, BooleanAccuracy, Metric

def prod(x: Iterable):
    acc = 1
    for v in x: acc *= v
    return acc

class MultilabelAccuracy(Metric):
    def __init__(self, thres=0.5):
        self.thres = 0.5
        self.correct_count = 0
        self.total_count = 0
    
    def __call__(self, logits: torch.FloatTensor, 
                 t: torch.LongTensor) -> float:
        logits = logits.detach().cpu().numpy()
        t = t.detach().cpu().numpy()
        cc = ((logits >= self.thres) == t).sum()
        tc = prod(logits.shape)
        self.correct_count += cc
        self.total_count += tc
        return cc / tc
    
    def get_metric(self, reset: bool=False):
        acc = self.correct_count / self.total_count
        if reset:
            self.reset()
        return acc
    
    @overrides
    def reset(self):
        self.correct_count = 0
        self.total_count = 0


# In[ ]:


from allennlp.nn.util import move_to_device, has_tensor

def permute(obj, p: torch.Tensor):
    """
    Given a structure (possibly) containing Tensors on the CPU,
    permute all the Tensors
    """
    if not has_tensor(obj):
        return obj
    elif isinstance(obj, torch.Tensor):
        return obj[p]
    elif isinstance(obj, dict):
        return {key: permute(value, p) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [permute(item, p) for item in obj]
    elif isinstance(obj, tuple):
        return tuple([permute(item, p) for item in obj])
    else:
        return obj


# In[ ]:


import torch.functional as F

class FocalLossWithLogits(nn.Module):
    """Borrowed from https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/65938"""
    def __init__(self, alpha=1., gamma=2.):
        super().__init__()
        self.alpha,self.gamma = alpha,gamma
        self._loss = nn.BCEWithLogitsLoss(reduction="none")
        
    def forward(self, y, t):
        bce_loss = self._loss(y, t)
        pt = torch.exp(-bce_loss)
        return (self.alpha * (1-pt) ** self.gamma * bce_loss).mean()


# In[ ]:


from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from torch.distributions.beta import Beta

class BaselineModel(Model):
    def __init__(self, word_embeddings: TextFieldEmbedder,
                 encoder: Seq2VecEncoder,
                 loss: nn.Module,
                 out_sz: int=config.n_classes,
                 multilabel: bool=True, 
                 dropouto=0.1,
                 mixup_alpha: int=0.2):
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.encoder = encoder
        feature_sz = self.encoder.get_output_dim()
        self.projection = nn.Sequential(
            nn.Linear(feature_sz, 50),
            nn.ELU(),
            nn.Dropout(dropouto),
            nn.Linear(50, out_sz),
        )        
        self.multilabel = multilabel
        self.lambda_sampler = Beta(torch.tensor([mixup_alpha]), torch.tensor([mixup_alpha]))
        if self.multilabel:
            self.accuracy = MultilabelAccuracy()
            self.per_label_acc = {c: MultilabelAccuracy() for c in label_cols}
            self.loss = loss
        else:
            self.accuracy = CategoricalAccuracy()
        self.is_test_mode = False
        self.loss = loss
            
    def test_mode(self, val=True):
        self.is_test_mode = val
        
    def get_embeddings(self, toks: Dict[str, torch.Tensor],
                       word_feats: torch.Tensor) -> torch.Tensor:
        """Encapsulates addition of word level features"""
        embeddings = self.word_embeddings(toks)
        if config.use_word_level_features:
            embeddings = torch.cat([word_feats, embeddings], dim=-1)
        return embeddings

    def forward(self, tokens: Dict[str, torch.Tensor],
                label: torch.Tensor,
                word_level_features: torch.Tensor,
                sentence_level_features: torch.Tensor,
                **meta) -> torch.Tensor:
        if self.is_test_mode: tokens["tokens"] *= 0
        
        mask = get_text_field_mask(tokens)
        embeddings = self.get_embeddings(tokens, word_level_features)
        state = self.encoder(embeddings, 
                             sentence_feats=sentence_level_features, 
                             mask=mask)
        class_logits = self.projection(state)
        
        output = {"class_logits": class_logits}

        output["accuracy"] = self.accuracy(class_logits, label)
        output["loss"] = self.loss(class_logits, label)

        return output

    def mixup(self, tokens: Dict[str, torch.Tensor],
              label: torch.Tensor,
              word_level_features: torch.Tensor,
              sentence_level_features: torch.Tensor,
              **meta) -> TensorDict:
        # generate new tokens and labels
        bs = label.size(0)
        shuf = torch.randperm(bs).to(label.device)
        tokens2 = permute(tokens, shuf)
        labels1, labels2 = label, permute(label, shuf)
        # TODO: Think of how to handle this masking intelligently
        mask1, mask2 = (get_text_field_mask(t) for t in (tokens, tokens2))
        embs1, embs2 = (self.get_embeddings(t, word_level_features) for t in (tokens, tokens2))
        # interpolate
        ratios = self.lambda_sampler.sample((bs, 1)).to(label.device)
        embs = ratios * embs1 + (1-ratios) * embs2
        label = ratios.squeeze(2) * labels1 + (1-ratios.squeeze(2)) * labels2
        
        # remaining process is the same
        # TODO: Handle stat feats
        state = self.encoder(embs, sentence_level_features, mask1 * mask2) # TODO: Handle masking
        class_logits = self.projection(state)
        
        output = {"loss": self.loss(class_logits, label)}
        return output
    
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}


# ### Prepare embeddings

# In[ ]:


config.set("vocab_size", min(vocab.get_vocab_size(), config.max_vocab_size))
if config.model_type == "standard":
    config.set("embedding_dim", 300)


# In[ ]:


from tqdm import tqdm
import warnings

def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')

def get_fasttext_embeddings(model_path: str, vocab: Vocabulary):
    prog_bar = tqdm(open(model_path, encoding="utf8", errors='ignore'))
    prog_bar.set_description("Loading embeddings")
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in prog_bar
                             if len(o)>100)
    all_embs = np.stack(embeddings_index.values())

    embeddings = np.zeros((config.vocab_size + 5, 300))
    n_missing_tokens = 0
    prog_bar = tqdm(vocab.get_index_to_token_vocabulary().items())
    prog_bar.set_description("Creating matrix")
    for idx, token in prog_bar:
        if idx == 0: continue # keep padding as all zeros
        if idx == 1: continue # Treat unknown words as dropped words
        if token == "[MASK]":
            embeddings[idx, :] = np.random.randn(300) * 0.5
        if token not in embeddings_index:
            n_missing_tokens += 1
            if n_missing_tokens < 10:
                warnings.warn(f"Token {token} not in embeddings: did you change preprocessing?")
            if n_missing_tokens == 10:
                warnings.warn(f"More than {n_missing_tokens} missing, supressing warnings")
        else:
            embeddings[idx, :] = embeddings_index[token]
    
    if n_missing_tokens > 0:
        warnings.warn(f"{n_missing_tokens} in total are missing from embedding text file")
    return embeddings


# In[ ]:


with timer("Loading embeddings"):
    if config.model_type == "standard":
        embedding_weights = get_fasttext_embeddings(config.ft_model_path, vocab)


# In[ ]:


class CustomEmbedding(Embedding):
    # TODO: Fix (make this decently efficient: currently allocating two embeddings)
    def __init__(self, num_embeddings, embedding_dim,
                 padding_index=None, max_norm=None, trainable=True,
                 weight=None, dropout=0., scale=None):
        super().__init__(num_embeddings, embedding_dim, weight=weight,
                         padding_index=padding_index, max_norm=max_norm,
                         trainable=trainable)
        self.dropout = dropout
        self.scale = scale
        self.padding_idx = padding_index

    def forward(self, words):
        weight = self.weight
        if self.dropout > 0.0 and self.training:
            mask = weight.data.new().resize_((weight.size(0), 1)).bernoulli_(1 - self.dropout).expand_as(weight) / (1 - self.dropout)
            masked_embed_weight = mask * weight
        else:
            masked_embed_weight = weight
        if self.scale:
            masked_embed_weight = scale.expand_as(masked_embed_weight) * masked_embed_weight

        padding_idx = self.padding_idx
        if padding_idx is None:
            padding_idx = -1

        X = torch.nn.functional.embedding(words, masked_embed_weight,
            padding_idx, self.max_norm, self.norm_type,
            self.scale_grad_by_freq, self.sparse
          )
        return X


# In[ ]:


from allennlp.modules.text_field_embedders.text_field_embedder import TextFieldEmbedder
from allennlp.modules.time_distributed import TimeDistributed

# TODO: Implement
class ElmoTextFieldEmbedder(TextFieldEmbedder):
    # AllenNLP support for caching sucks by default
    # so we have to write our own embedder to bypass this problem
    def __init__(self,
                 token_embedders: Dict[str, Any],
                 embedder_to_indexer_map: Dict[str, List[str]] = None,
                 allow_unmatched_keys: bool = False) -> None:
        super().__init__()
        self._token_embedders = token_embedders
        self._embedder_to_indexer_map = embedder_to_indexer_map
        for key, embedder in token_embedders.items():
            name = 'token_embedder_%s' % key
            self.add_module(name, embedder)
        self._allow_unmatched_keys = allow_unmatched_keys
    
    @overrides
    def get_output_dim(self) -> int:
        output_dim = 0
        for embedder in self._token_embedders.values():
            output_dim += embedder.get_output_dim()
        return output_dim
    
    def forward(self, text_field_input: Dict[str, torch.Tensor],
                num_wrapping_dims: int = 0) -> torch.Tensor:
        if self._token_embedders.keys() != text_field_input.keys():
            if not self._allow_unmatched_keys:
                message = "Mismatched token keys: %s and %s" % (str(self._token_embedders.keys()),
                                                                str(text_field_input.keys()))
                raise ConfigurationError(message)
        embedded_representations = []
        keys = sorted(self._token_embedders.keys())
        for key in keys:
            # If we pre-specified a mapping explictly, use that.
            if self._embedder_to_indexer_map is not None:
                tensors = [text_field_input[indexer_key] for
                           indexer_key in self._embedder_to_indexer_map[key]]
            else:
                # otherwise, we assume the mapping between indexers and embedders
                # is bijective and just use the key directly.
                tensors = [text_field_input[key]]
            # Note: need to use getattr here so that the pytorch voodoo
            # with submodules works with multiple GPUs.
            embedder = getattr(self, 'token_embedder_{}'.format(key))
            for _ in range(num_wrapping_dims):
                embedder = TimeDistributed(embedder)
            # Force embedder to use word inputs
            if key == "tokens":
                token_vectors = embedder(tensors[0], 
                                         word_inputs=tensors[0] if config.cache_elmo_embeddings else None)
            else:
                token_vectors = embedder(*tensors)
            embedded_representations.append(token_vectors)
        return torch.cat(embedded_representations, dim=-1)


# In[ ]:


from pytorch_pretrained_bert.modeling import BertModel
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder

class CustomBertEmbedder(TokenEmbedder):
    """
    A ``TokenEmbedder`` that produces BERT embeddings for your tokens.
    Should be paired with a ``BertIndexer``, which produces wordpiece ids.
    Sums last 4 hidden layers for now (might use scalar mix in the future)
    """
    def __init__(self, pretrained_model: str,
                 use_scalar_mix: bool = False,
                 fine_tune: bool = False,
                 n_hidden_layers: int = 4) -> None:
        super().__init__()
        if use_scalar_mix and fine_tune:
            raise ConfigurationError("Choose mix or fine tuning")
        
        self.bert_model = BertModel.from_pretrained(pretrained_model)
        for param in self.bert_model.parameters():
            param.requires_grad = fine_tune
        self.output_dim = self.bert_model.config.hidden_size
        self.n_hidden_layers = n_hidden_layers
        if use_scalar_mix:
            self._scalar_mix = ScalarMix(n_hidden_layers,
                                         do_layer_norm=False)
        else:
            self._scalar_mix = None

    def get_output_dim(self) -> int:
        return self.output_dim

    def forward(self,
                input_ids: torch.LongTensor,
                offsets: torch.LongTensor = None,
                token_type_ids: torch.LongTensor = None) -> torch.Tensor:
        """
        Parameters
        ----------
        input_ids : ``torch.LongTensor``
            The (batch_size, ..., max_sequence_length) tensor of wordpiece ids.
        offsets : ``torch.LongTensor``, optional
            The BERT embeddings are one per wordpiece. However it's possible/likely
            you might want one per original token. In that case, ``offsets``
            represents the indices of the desired wordpiece for each original token.
            Depending on how your token indexer is configured, this could be the
            position of the last wordpiece for each token, or it could be the position
            of the first wordpiece for each token.

            For example, if you had the sentence "Definitely not", and if the corresponding
            wordpieces were ["Def", "##in", "##ite", "##ly", "not"], then the input_ids
            would be 5 wordpiece ids, and the "last wordpiece" offsets would be [3, 4].
            If offsets are provided, the returned tensor will contain only the wordpiece
            embeddings at those positions, and (in particular) will contain one embedding
            per token. If offsets are not provided, the entire tensor of wordpiece embeddings
            will be returned.
        token_type_ids : ``torch.LongTensor``, optional
            If an input consists of two sentences (as in the BERT paper),
            tokens from the first sentence should have type 0 and tokens from
            the second sentence should have type 1.  If you don't provide this
            (the default BertIndexer doesn't) then it's assumed to be all 0s.
        """
        # pylint: disable=arguments-differ
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        input_mask = (input_ids != 0).long()

        # input_ids may have extra dimensions, so we reshape down to 2-d
        # before calling the BERT model and then reshape back at the end.
        all_encoder_layers, _ = self.bert_model(input_ids=nn_util.combine_initial_dims(input_ids),
                                                token_type_ids=nn_util.combine_initial_dims(token_type_ids),
                                                attention_mask=nn_util.combine_initial_dims(input_mask))
        if self._scalar_mix is not None:
            mix = self._scalar_mix(all_encoder_layers[-self.n_hidden_layers:], input_mask)
        else:
            mix = torch.stack(all_encoder_layers[-self.n_hidden_layers:]).mean(dim=0)

        # At this point, mix is (batch_size * d1 * ... * dn, sequence_length, embedding_dim)

        if offsets is None:
            # Resize to (batch_size, d1, ..., dn, sequence_length, embedding_dim)
            return nn_util.uncombine_initial_dims(mix, input_ids.size())
        else:
            # offsets is (batch_size, d1, ..., dn, orig_sequence_length)
            offsets2d = nn_util.combine_initial_dims(offsets)
            # now offsets is (batch_size * d1 * ... * dn, orig_sequence_length)
            range_vector = nn_util.get_range_vector(offsets2d.size(0),
                                                 device=nn_util.get_device_of(mix)).unsqueeze(1)
            # selected embeddings is also (batch_size * d1 * ... * dn, orig_sequence_length)
            selected_embeddings = mix[range_vector, offsets2d]

            return util.uncombine_initial_dims(selected_embeddings, offsets.size())


# In[ ]:


from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder

if config.model_type == "standard":
    token_embedding = CustomEmbedding(num_embeddings=config.vocab_size + 5,
                                      embedding_dim=config.embedding_dim,
                                      trainable=not config.freeze_embeddings,
                                      weight=torch.tensor(embedding_weights, dtype=torch.float),
                                      dropout=config.dropoute, padding_index=0)
    word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})
elif "elmo" in config.model_type:
    from allennlp.modules.token_embedders import ElmoTokenEmbedder
    from allennlp.modules.elmo import Elmo

    options_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json'
    weight_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5'
    
    all_words_ordered = [w for i, w in sorted([(i, w) for w, i in vocab.get_token_to_index_vocabulary().items()])]
    elmo_embedder = ElmoTokenEmbedder(
        options_file, weight_file, dropout=config.dropoute,
        vocab_to_cache=all_words_ordered if config.cache_elmo_embeddings else None
    )
    # TODO: Find a way to skip character encodings
    word_embeddings = ElmoTextFieldEmbedder({"tokens": elmo_embedder})
elif "bert" in config.model_type:
    from allennlp.modules.token_embedders.bert_token_embedder import PretrainedBertEmbedder
    bert_embedder = CustomBertEmbedder(
            pretrained_model=config.model_type,
            fine_tune=False, use_scalar_mix=False,
    )
    word_embeddings: TextFieldEmbedder = BasicTextFieldEmbedder({"tokens": bert_embedder},
                                                                 # we'll be ignoring masks so we'll need to set this to True
                                                                allow_unmatched_keys = True)


# In[ ]:


if config.pooling_type != "bert_pool":
    embed_sz = word_embeddings.get_output_dim()
    if config.use_word_level_features: 
        embed_sz += len(word_level_features)
    rnn = BiRNN(rnn_type=rnn_type, n_layers=config.num_layers, 
                embed_sz=embed_sz, hidden_sz=config.hidden_sz, 
                dropoutw=config.dropoutw)


# In[ ]:


if config.pooling_type != "bert_pool":
    if config.pooling_type == "attention":
        pooler = Attention(rnn.get_output_dim(), hidden_sz=rnn.get_output_dim(),
                           out_sz=rnn.get_output_dim(), dim=1, 
                           use_bias=config.attention_bias)
    elif config.pooling_type == "multipool":
        pooler = MultiPooling(rnn.get_output_dim())
    elif config.pooling_type == "augmented_multipool":
        pooler = AugmentedMultiPool(rnn.get_output_dim(), 
                                    aug_sz=embed_sz)
    else:
        raise ValueError(f"Invalid pooling type {config.pooling_type}")

    encoder = BiRNNEncoder(
            rnn,
            pooler,
            dropouti=config.dropouti,
            dropoutr=config.dropoutr,
        )
else:
    BERT_DIM = word_embeddings.get_output_dim()

    class BertSentencePooler(Seq2VecEncoder):
        def forward(self, embs: torch.tensor, 
                    mask: torch.tensor=None,
                    **kwargs,
                   ) -> torch.tensor:
            # extract first token tensor
            return embs[:, 0]

        @overrides
        def get_output_dim(self) -> int:
            return BERT_DIM

    encoder = BertSentencePooler(vocab)


# In[ ]:


if config.n_classes > 2:
    if config.use_focal_loss:
        loss = FocalLossWithLogits(config.focal_loss_alpha,
                                   config.focal_loss_gamma)
    else:
        loss = nn.BCEWithLogitsLoss()
else:
    loss = nn.CrossEntropyLoss()


# In[ ]:


model = BaselineModel(
    word_embeddings, 
    encoder, 
    loss,
    out_sz=config.n_classes,
)


# In[ ]:





# Initialize bias according to prior

# In[ ]:


if config.bias_init:
    class_bias = torch.zeros(len(label_cols))
    for i, _ in enumerate(label_cols):
        p = train_labels[:, i].mean()
        class_bias[i] = np.log(p / (1-p))

    model.projection[-1].bias.data = class_bias


# In[ ]:


if USE_GPU: model.cuda()
else: model


# In[ ]:


from copy import deepcopy
init_state_dict = deepcopy(model.state_dict())


# In[ ]:





# ### Basic sanity checks

# In[ ]:


batch = nn_util.move_to_device(batch, 0 if USE_GPU else -1)


# In[ ]:


tokens = batch["tokens"]
labels = batch


# In[ ]:


tokens


# In[ ]:


mask = get_text_field_mask(tokens)
wlfs = batch["word_level_features"]


# In[ ]:


#embs = model.word_embeddings.token_embedder_tokens(tokens["tokens"])


# In[ ]:


embeddings = model.word_embeddings(tokens)
if config.use_word_level_features:
    embeddings = torch.cat([wlfs, embeddings], dim=-1)


# In[ ]:


embeddings.shape


# In[ ]:


mask.shape


# In[ ]:


#if config.pooling_type != "bert_pool":
    #encoded = model.encoder.rnn(embeddings, mask=mask)
#else:
    #encoded = model.encoder(embeddings, mask=mask)


# In[ ]:


#encoded.shape


# In[ ]:


model(**batch)


# In[ ]:


loss = model(**batch)["loss"]


# In[ ]:


loss


# In[ ]:


batch["label"].shape[0]


# In[ ]:


batch["tokens"]["tokens"].shape[0]


# In[ ]:


# model.mixup(**batch)


# In[ ]:


model.zero_grad()


# In[ ]:





# # Train

# In[ ]:


from allennlp.training import Callback


# In[ ]:


class StopTraining(Exception): pass


# In[ ]:


class NanWeightMonitor(Callback):
    def on_backward_end(self, data):
        for name, param in self.trainer.model.named_parameters():
            if torch.isnan(param.data).any() or torch.isinf(param.data).any():
                raise StopTraining(f"Nan/Inf weights in param {name}: \n {param}")
try:
    import jupyter_slack
    can_notify = True
except:
    jupyter_slack = None
    can_notify = False

class SlackNotification(Callback):
    def __init__(self, silent):
        self.silent = silent
    def on_train_end(self, data):
        if not self.silent and can_notify:
            try:
                jupyter_slack.notify_self(f"Finished training with state {self.trainer._state}")
            except: pass


# In[ ]:


class TensorboardCallback(Callback):
    """For now, delegate all processing to the trainer's own methods"""
    def on_batch_begin(self):
        self._log_histograms_this_batch =         self.trainer._histogram_interval is not None and (
            self.trainer._batch_num_total % self.trainer._histogram_interval == 0)
    
    def on_backward_end(self, loss):
        if self._log_histograms_this_batch:
            # get the magnitude of parameter updates for logging
            # We need a copy of current parameters to compute magnitude of updates,
            # and copy them to CPU so large models won't go OOM on the GPU.
            self.param_updates = {
                name: param.detach().cpu().clone()
                for name, param in self.trainer.model.named_parameters()
            }
    
    def on_step_end(self, loss):
        if self._log_histograms_this_batch:
            for name, param in self.trainer.model.named_parameters():
                self.param_updates[name].sub_(param.detach().cpu())
                update_norm = torch.norm(self.param_updates[name].view(-1, ))
                param_norm = torch.norm(param.view(-1, )).cpu()
                self.trainer._tensorboard.add_train_scalar(
                    "gradient_update/" + name,
                     update_norm / (param_norm + 1e-7),
                     batch_num_total
                )
            self.param_updates = {} # release memory


# In[ ]:


class Mixup(Callback):
    """Does mixup in embedding space
    TODO: Figure out how to best handle masking...
    """
    def __init__(self, weight: float, 
                 batch_iterator: Optional[Iterable[TensorDict]]=None):
        self.weight = weight
        self.batch_iterator = batch_iterator
        
    def on_batch_loss(self, batch: TensorDict, for_training=True):
        if for_training and self.weight > 0.:
            # use mixup iterator if exists
            if self.batch_iterator is not None: batch = next(self.batch_iterator)
            mixup_output_dict = self.model.mixup(**batch)
            return mixup_output_dict["loss"] * self.weight
        else:
            return None


# In[ ]:


class DiscreteMixup(Callback):
    """Mixes up and concatenates sentences within a batch"""
    def __init__(self, weight: float, 
                 batch_iterator: Optional[Iterable[TensorDict]]=None):
        self.weight = weight
        self.batch_iterator = batch_iterator
    
    def on_batch_loss(self, batch: TensorDict, for_training=True):
        if for_training and self.weight > 0.:
            # use mixup iterator if exists
            if self.batch_iterator is not None: 
                batch = next(self.batch_iterator)
                
            # create permutation
            tokens, label = batch["tokens"], batch["label"]
            bs = label.size(0)
            shuf = torch.randperm(bs).to(label.device)
            tokens2 = permute(tokens, shuf)
            labels2 = permute(label, shuf)
            
            # join the sentences
            n_tokens1 = get_text_field_mask(tokens).sum(1)
            n_tokens2 = get_text_field_mask(tokens2).sum(1)
            maxlen = min(config.max_seq_len, (n_tokens1 + n_tokens2).sum())
            # TODO: Is there a faster way?
            new_tokens = torch.zeros(bs, maxlen, 
                                     dtype=torch.long).to(label.device)
            for i, (t1, t2) in enumerate(zip(tokens["tokens"], tokens2["tokens"])):
                l1, l2 = n_tokens1[i].item(), n_tokens2[i].item()
                new_tokens[i, :l1] = t1 # TODO: Fairly divide the capacity
                new_tokens[i, l1:min(maxlen, l1+l2)] =                     t2[:min(maxlen-l1, l2)]
            
            # compute loss on new batch
            new_batch = {k: v for k, v in batch.items()}
            new_batch["tokens"] = {"tokens": new_tokens}
            new_batch["label"] = new_label
            return model(**new_batch)["loss"] * self.weight
        else:
            return None


# In[ ]:


class LinearEpochDropoutSchedule:
    def __init__(self, n_epochs, start_do, end_do):
        self.n_epochs = n_epochs
        self.start_do = start_do
        self.end_do = end_do
        if n_epochs > 1:
            self.delta_do = (end_do - start_do) / (n_epochs - 1)
        else:
            self.delta_do = 0 # cannot change dropout if only one epoch

    def __call__(self, epochs, batches_this_epoch, batches_total):
        return self.start_do + self.delta_do * epochs

class DropoutScheduler(Callback):
    def __init__(self, 
                 module: nn.Module,
                 schedule: Callable[[int, int, int], float]):
        self._module = module
        self._schedule = schedule
        self.epoch = 0
    
    def on_batch_end(self, data):
        self._module.dropout = self._schedule(self.epoch, 
                                              data["batches_this_epoch"],
                                              data["batch_num_total"],
                                             )
    def on_epoch_end(self, data):
        # handle per-epoch schedules
        self.epoch += 1
        self._module.dropout = self._schedule(self.epoch, -1, -1)


# Test performance when input is all 0s
# - If our initialization works decently, the loss should barely/not move and accuracy should stay constant

# In[ ]:


from allennlp.training import TrainerWithCallbacks


# In[ ]:


if config.debugging:
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    model.test_mode()
    trainer = TrainerWithCallbacks(
        model=model,
        optimizer=optimizer,
        iterator=iterator,
        train_dataset=train_ds[:256],
        cuda_device=0 if USE_GPU else -1,
        num_epochs=5,
    )
    metrics = trainer.train()
    model.load_state_dict(init_state_dict)
    model.test_mode(False)


# Test performance on a small batch

# In[ ]:


if config.debugging:
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    state_dict = deepcopy(model.state_dict())
    trainer = TrainerWithCallbacks(
        model=model,
        optimizer=optimizer,
        iterator=iterator,
        train_dataset=train_ds[:256],
        cuda_device=0 if USE_GPU else -1,
        num_epochs=50,
    )
    metrics = trainer.train()
    model.load_state_dict(init_state_dict)
    metrics


# In[ ]:


#from matplotlib import pyplot as plt
import math

class LRFinder(Callback):
    def __init__(self):
        self.losses = []
        self.lrs = []
        self.best_loss = 1e9

    def on_step_end(self, loss, **kwargs):
        # Log the learning rate
        self.losses.append(loss.item())
        self.lrs.append(self.trainer.optimizer.state_dict()['param_groups'][0]["lr"])

    def plot_loss(self, n_skip_beginning=10, n_skip_end=5):
        plt.ylabel("loss")
        plt.xlabel("learning rate (log scale)")
        plt.plot(self.lrs[n_skip_beginning:-n_skip_end], self.losses[n_skip_beginning:-n_skip_end])
        plt.xscale('log')

    def plot_loss_change(self, sma=1, n_skip_beginning=10, n_skip_end=5, y_lim=(-0.01, 0.01)):
        assert sma >= 1
        derivatives = [0] * sma
        for i in range(sma, len(self.lrs)):
            derivative = (self.losses[i] - self.losses[i - sma]) / sma
            derivatives.append(derivative)

        plt.ylabel("rate of loss change")
        plt.xlabel("learning rate (log scale)")
        plt.plot(self.lrs[n_skip_beginning:-n_skip_end], derivatives[n_skip_beginning:-n_skip_end])
        plt.xscale('log')
        plt.ylim(y_lim)


# In[ ]:


if config.find_lr:
    from copy import deepcopy
    
    class ExponentialIncrease(torch.optim.lr_scheduler._LRScheduler):
        def __init__(self, optimizer: torch.optim.Optimizer, n_iters: int,
                     lr_start=1e-6, lr_end=2.0) -> None:
            self.n_iters = n_iters
            self.steps = 0
            self.lr_start = lr_start
            self.gamma = (lr_end / lr_start) ** (1 / n_iters)
            super().__init__(optimizer)
        def step(self, epoch=None): pass
        def step_batch(self, epoch=None):
            self.steps += 1
            if epoch is None: epoch = self.last_epoch + 1
            self.last_epoch = epoch
            for param_group, learning_rate in zip(self.optimizer.param_groups, self.get_lr()):
                param_group['lr'] = learning_rate
        def get_lr(self):
            return [self.lr_start * (self.gamma ** self.steps) for _ in self.base_lrs]
    optimizer = optim.Adam(model.parameters(), lr=1e-6)
    lr_finder = LRFinder()
    trainer = TrainerWithCallbacks(
        model=model,
        optimizer=optimizer,
        iterator=iterator,
        train_dataset=train_ds,
        learning_rate_scheduler=ExponentialIncrease(optimizer, 
                                                    iterator.get_num_batches(train_ds)),
        cuda_device=0 if USE_GPU else -1,
        num_epochs=1,
        callbacks=[lr_finder],
    )
    trainer.train()
    model.load_state_dict(init_state_dict)
    del model_state


# In[ ]:


if config.find_lr:
    lr_finder.plot_loss(n_skip_beginning=0, n_skip_end=1)


# In[ ]:





# # Actual Training

# In[ ]:


model.load_state_dict(init_state_dict)


# In[ ]:


optimizer = optim.Adam(model.parameters(), 
                       lr=config.lr, weight_decay=config.weight_decay)


# In[ ]:


def _prod(args):
    acc = 1
    for a in args: acc *= a
    return acc
num_trainable_params = sum([_prod(p.shape) for p in model.parameters() if p.requires_grad])
num_trainable_params


# In[ ]:


from allennlp.training.learning_rate_schedulers import SlantedTriangular, CosineWithRestarts
if config.lr_schedule == "slanted_triangular":
    lr_sched = SlantedTriangular(optimizer, 
                                 num_epochs=config.epochs, 
                                 num_steps_per_epoch=iterator.get_num_batches(train_ds))
elif config.lr_scheduler == "cosine_annealing":
    lr_sched = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=iterator.get_num_batches(train_ds) * config.epochs,
    )
elif config.lr_scheduler is None:
    lr_sched = None
else:
    raise ConfigurationError(f"Invalid lr schedule {config.lr_scheduler} passed")


# In[ ]:


training_options = {
    # TODO: Add appropriate learning rate scheduler
    "should_log_parameter_statistics": True,
    "should_log_learning_rate": True,
    "num_epochs": config.epochs,
}


# In[ ]:


callbacks = [NanWeightMonitor(), 
             SlackNotification(silent=config.testing)]
if config.dropoute_max is not None:
    callbacks.append(DropoutScheduler(
        model.word_embeddings.token_embedder_tokens,
        LinearEpochDropoutSchedule(config.epochs, config.dropoute, config.dropoute_max)
    ))


# In[ ]:





# In[ ]:


trainer = TrainerWithCallbacks(
    model=model,
    optimizer=optimizer,
    iterator=iterator,
    train_dataset=train_ds + train_aug_ds if config.use_augmented else train_ds,
    validation_dataset=val_ds if config.val_ratio > 0.0 else None,
    serialization_dir=SER_DIR,
    cuda_device=0 if USE_GPU else -1,
    gradient_accumulation_steps=config.batch_size // config.computational_batch_size,
    callbacks=callbacks,
    learning_rate_scheduler=lr_sched,
    **training_options,
)


# In[ ]:

if save_template:
  torch.save(model, str(SER_DIR / TEMPLATE_NAME))

#sys.exit(0)

if load_template:
  model = torch.load(str(SER_DIR / TEMPLATE_NAME))


#metrics = trainer.train()
#metrics

best_model_state = trainer._checkpointer.best_model_state()
if best_model_state:
    model.load_state_dict(best_model_state)
else:
    raise Exception('Best state not found!')


# In[ ]:


tuning = False

if tuning:
    from allennlp.commands.find_learning_rate import search_learning_rate, _save_plot
    lrs_, losses_ = search_learning_rate(trainer, num_batches=300)

    plt.ylabel("loss")
    plt.xlabel('learning rate (log10 scale)')  
    plt.xscale('log')
    plt.plot(lrs_[0:150], losses_[0:150])

    from scipy.ndimage.filters import gaussian_filter1d
    from matplotlib import pyplot as plt
    import math

    #lsmoothed = gaussian_filter1d(losses_, sigma=3)

    lsmoothed = []
    w = 10

    for i in range(300):
        if i < w:
            lsmoothed.append(losses_[i])
        else:
            lsmoothed.append(sum(losses_[i-w:i])/w)

    plt.ylabel("loss")
    plt.xlabel('learning rate (log10 scale)')
    plt.xscale('log')
    plt.plot(lrs_[w:150], lsmoothed[w:150])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # Evaluate

# In[ ]:


from scipy.special import expit
from collections import defaultdict
from allennlp.common.tqdm import Tqdm

def dict_append(d: Dict[str, List], upd: Dict[str, Any]) -> Dict[str, List]:
    for k, v in upd.items(): d[k].append(v)

def tonp(tsr): return tsr.detach().cpu().numpy()
        
class Predictor:
    def __init__(self, model: Model, iterator: DataIterator,
                 cuda_device: int=-1) -> None:
        self.model = model
        self.iterator = iterator
        self.cuda_device = cuda_device
        
    def _extract_data(self, batch) -> Dict[str, np.ndarray]:
        out_dict = self.model(**batch)
        lens = tonp(get_text_field_mask(batch["tokens"]).sum(1))
        return {
                "preds": expit(tonp(out_dict["class_logits"])),
                "oov_ratio": tonp((batch["tokens"]["tokens"] == 1).sum(1)) / lens,
                "lens": lens,
               }
        
    def _postprocess(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        return {k: np.concatenate(v, axis=0) for k, v in predictions.items()}
    
    @gpu_mem_restore
    def predict(self, ds: Iterable[Instance]) -> np.ndarray:
        pred_generator = self.iterator(ds, num_epochs=1, shuffle=False)
        self.model.eval()
        pred_generator_tqdm = Tqdm.tqdm(pred_generator,
                                        total=self.iterator.get_num_batches(ds))
        preds = defaultdict(list)
        with torch.no_grad():
            for batch in pred_generator_tqdm:
                batch = nn_util.move_to_device(batch, self.cuda_device)
                dict_append(preds, self._extract_data(batch))
        return self._postprocess(preds)


# In[ ]:


from allennlp.data.iterators import BasicIterator
seq_iterator = BasicIterator(batch_size=64)
seq_iterator.index_with(vocab)


# In[ ]:


# Horrible solution to the shuffling problem with BasicIterator
# TODO: Solve more elegantly?
if not config.bucket:
    del train_ds; import gc; gc.collect()
    if config.val_ratio > 0.0:
        train_ds = reader.read(DATA_ROOT / "train_wo_val.csv")
    else:
        train_ds = reader.read(DATA_ROOT / "train.csv")


# In[ ]:


predictor = Predictor(model, seq_iterator, cuda_device=0 if USE_GPU else -1)
train_meta = predictor.predict(train_ds) 
train_preds = train_meta.pop("preds")
test_meta = predictor.predict(test_ds)
test_preds = test_meta.pop("preds")


# In[ ]:





# In[ ]:


tst_df = pd.read_csv(DATA_ROOT / "test_proced.csv")
test_labels = tst_df[label_cols].values
test_texts = tst_df["comment_text"].values
if config.testing:
    test_labels = test_labels[:len(test_ds), :]
    test_texts = test_texts[:len(test_ds)]


# In[ ]:


from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, confusion_matrix


# Per label

# In[ ]:


class Evaluator:
    def __init__(self, thres=0.5):
        if isinstance(thres, float):
            self.thres = np.ones(len(label_cols)) * thres
        else:
            self.thres = thres
    
    def _to_metric_dict(self, t: np.ndarray, y: np.ndarray, thres: float) -> Dict:
        tn, fp, fn, tp = confusion_matrix(t, y >= thres).ravel()
        return {"auc": roc_auc_score(t, y),
                "f1": f1_score(t, y >= thres),
                "acc": accuracy_score(t, y >= thres),
                "tnr": tn / len(t), "fpr": fp / len(t),
                "fnr": fn / len(t), "tpr": tp / len(t),
                "precision": tp / (tp + fp), "recall": tp / (tp + fn),
        }

    def _stats_per_quadrant(self, tgt, preds, 
                            metadata: Dict[str, np.ndarray],
                            texts: np.ndarray=None):
        out_data = {}
        for i, lbl in enumerate(label_cols):
            # get indicies of each quadrant`
            preds_bin = preds[:, i] >= self.thres[i]
            quads = {
                "tp": np.where((tgt[:, i] == 1) & preds_bin)[0],
                "fp": np.where((tgt[:, i] == 0) & preds_bin)[0],
                "tn": np.where((tgt[:, i] == 0) & ~preds_bin)[0],
                "fn": np.where((tgt[:, i] == 1) & ~preds_bin)[0],
            }
            
            # get stats for metadata
            out_data[lbl] = {}
            for quad, qidxs in quads.items():
                quad_data = {}
                for k, full_data in metadata.items():
                    data = full_data[qidxs]
                    for metric in ["mean", "std", "min", "max"]:
                        if len(data) > 0:
                            quad_data[f"{k}_{metric}"] = getattr(data, metric)()
                        else:
                            quad_data[f"{k}_{metric}"] = np.nan

                out_data[lbl][quad] = quad_data
            
            # do error analysis
            if texts is not None:
                for quad, qidxs in quads.items():
                    quad_preds = preds[qidxs, i]
                    if len(quad_preds) == 0: continue
                    if quad in ["tp", "fp"]:
                        out_data[lbl][quad]["most_confident"] = texts[quad_preds.argmax()]
                        out_data[lbl][quad]["most_confident_prob"] = quad_preds.max()
                        out_data[lbl][quad]["least_confident"] = texts[quad_preds.argmin()]
                        out_data[lbl][quad]["least_confident_prob"] = quad_preds.min()
                    else:
                        out_data[lbl][quad]["most_confident"] = texts[quad_preds.argmin()]
                        out_data[lbl][quad]["most_confident_prob"] = quad_preds.min()
                        out_data[lbl][quad]["least_confident"] = texts[quad_preds.argmax()]
                        out_data[lbl][quad]["least_confident_prob"] = quad_preds.max()
        return out_data        
    
    @gpu_mem_restore
    def evaluate(self, tgt: np.ndarray, preds: np.ndarray,
                 trn_tgt: np.ndarray, trn_preds: np.ndarray,
                 metadata: Dict[str, np.ndarray]={}, 
                 texts: np.ndarray=None) -> Dict:
        """
        Metadata: Data about the inputs (e.g. length, OOV ratio)
        """
        train_label_metrics = {}
        label_metrics = {}
                
        # get per-label stats
        for i, lbl in enumerate(label_cols):
            train_label_metrics[lbl] = self._to_metric_dict(trn_tgt[:, i],
                                                            trn_preds[:, i],
                                                            self.thres[i])
            label_metrics[lbl] = self._to_metric_dict(tgt[:, i], preds[:, i],
                                                      self.thres[i])
            print(f"========{lbl}=========")
            print(label_metrics[lbl])
        
        # get global stats
        for mtrc in label_metrics["toxic"].keys():
            label_metrics[f"global_{mtrc}"] =                 np.mean([label_metrics[col][mtrc] for col in label_cols])
            
        # get per-label-quadrant stats
        quad_stats = self._stats_per_quadrant(tgt, preds, metadata=metadata, texts=texts)
        if len(quad_stats) > 0:
            for c in label_cols:
                label_metrics[c]["quad_stats"] = quad_stats[c]

        label_metrics["train"] = train_label_metrics,
        return label_metrics


# In[ ]:


# Compute best threshold based on training data
if config.compute_thres_on_test:
    lbls, pds = test_labels, test_preds
else:
    lbls, pds = train_labels, train_preds
    
thres = np.zeros(len(label_cols))
best_scores = np.zeros(len(label_cols))
for i, col in enumerate(label_cols):
    best_score = -1
    best_thres = -1
    for x in np.linspace(0, 1.0, num=999):
        scr = f1_score(lbls[:, i], pds[:, i] >= x)
        if scr > best_score:
            best_thres = x
            best_score = scr
    thres[i] = best_thres
    best_scores[i] = best_score


# In[ ]:


thres


# In[ ]:


evaluator = Evaluator(thres=thres)
label_metrics = evaluator.evaluate(
    test_labels, test_preds,
    train_labels, train_preds,
    metadata=test_meta, texts=test_texts,
)


# In[ ]:


label_metrics


# In[ ]:


label_metrics["train"]


# In[ ]:


label_metrics


# # Record results and save weights

# In[ ]:


if os.environ["IS_COLAB"] != "True":
    import sys
    sys.path.append("../lib")
    from record_experiments import record
else:
    PASSWORD = "mongo11747" # FILL IN IF COLAB

    from typing import *
    import pymongo
    from bson.objectid import ObjectId
    import os
    import logging

    # Logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(levelname)s] %(asctime)s - %(name)s %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    conn_str = f"mongodb+srv://root:{PASSWORD}@cluster0-ptgoc.mongodb.net/test?retryWrites=true"

    client = pymongo.MongoClient(conn_str)
    db = client.experiments
    collection = db.logs

    def _cln(v: Any) -> Any:
        """Ensure variables are serializable"""
        if isinstance(v, (np.float, np.float16, np.float32, np.float64, np.float128)):
            return float(v)
        elif isinstance(v, (np.int, np.int0, np.int8, np.int16, np.int32, np.int64)):
            return int(v)
        elif isinstance(v, dict):
            return {k: _cln(v_) for k, v_ in v.items()}
        else:
            return v

    def record(log: dict):
        res = collection.insert_one({str(k): _cln(v) for k, v in log.items()})
        logger.info(f"Inserted results at id {res.inserted_id}")
        return res

    def find(id_: Optional[str]=None, query: Optional[dict]=None):
        if query is None: query = {"_id": ObjectId(id_)}
        res = collection.find_one(query)
        return res

    def delete(id_: Optional[str]=None, query: Optional[dict]=None):
        if query is None: query = {"_id": ObjectId(id_)}
        res = collection.delete_many(query)
        logger.info(f"Deleted {res.deleted_count} entries")
        return res


# In[ ]:







