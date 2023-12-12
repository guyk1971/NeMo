import torch
import nemo
from nemo.core import NeuralModule
from nemo.core import typecheck
from nemo.core.neural_types import NeuralType
from nemo.core.neural_types import *

import math
from typing import List, Set, Dict, Tuple, Optional
import torch.nn as nn
from torch.nn import functional as F

import pytorch_lightning as ptl
from nemo.core import ModelPT
from omegaconf import OmegaConf,MISSING
from nemo.core.classes.common import PretrainedModelInfo
# for datasets
import os 
from nemo.core import Dataset
from torch.utils import data
from torch.utils.data import TensorDataset, DataLoader
import wandb
import dataclasses
if torch.cuda.is_available():
  accelerator = 'gpu'
else:
  accelerator = 'cpu'
print(accelerator)
WANDB_API_KEY=os.getenv('WANDB_API_KEY')
wandb.login(key=WANDB_API_KEY)

# Creating Element Types
class AttentionType(EncodedRepresentation):
  """Basic Attention Element Type"""

class SelfAttentionType(AttentionType):
  """Self Attention Element Type"""

class CausalSelfAttentionType(SelfAttentionType):
  """Causal Self Attention Element Type"""

# Creating the modules
class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, n_embd, block_size, n_head, attn_pdrop, resid_pdrop):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size))
                                     .view(1, 1, block_size, block_size))
    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y
    

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embd, block_size, n_head, attn_pdrop, resid_pdrop):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, block_size, n_head, attn_pdrop, resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPTEmbedding(NeuralModule):
    def __init__(self, vocab_size: int, n_embd: int, block_size: int, embd_pdrop: float = 0.0):
        super().__init__()

        # input embedding stem: drop(content + position)
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, n_embd))
        self.drop = nn.Dropout(embd_pdrop)

    @typecheck()
    def forward(self, idx):
        b, t = idx.size()
        
        # forward the GPT model
        token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector
        position_embeddings = self.pos_emb[:, :t, :] # each position maps to a (learnable) vector
        x = self.drop(token_embeddings + position_embeddings)
        return x

    @property
    def input_types(self):
        return {
            'idx': NeuralType(('B', 'T'), Index())
        }

    @property
    def output_types(self):
        return {
            'embeddings': NeuralType(('B', 'T', 'C'), EmbeddedTextType())
        }
  

class GPTTransformerEncoder(NeuralModule):
    def __init__(self, n_embd: int, block_size: int, n_head: int, n_layer: int, attn_pdrop: float = 0.0, resid_pdrop: float = 0.0):
        super().__init__()

        self.blocks = nn.Sequential(*[Block(n_embd, block_size, n_head, attn_pdrop, resid_pdrop) 
                                    for _ in range(n_layer)])
        
    @typecheck()
    def forward(self, embed):
        return self.blocks(embed)

    @property
    def input_types(self):
        return {
            'embed': NeuralType(('B', 'T', 'C'), EmbeddedTextType())
        }

    @property
    def output_types(self):
        return {
            'encoding': NeuralType(('B', 'T', 'C'), CausalSelfAttentionType())
        }


class GPTDecoder(NeuralModule):
    def __init__(self, n_embd: int, vocab_size: int):
        super().__init__()
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False) # no need for extra bias due to one in ln_f

    @typecheck()
    def forward(self, encoding):
        x = self.ln_f(encoding)
        logits = self.head(x)
        return logits

    @property
    def input_types(self):
        return {
            'encoding': NeuralType(('B', 'T', 'C'), EncodedRepresentation())
        }
    
    @property
    def output_types(self):
        return {
            'logits': NeuralType(('B', 'T', 'C'), LogitsType())
        }

class AbstractNeMoGPT(ModelPT):
    def __init__(self, cfg: OmegaConf, trainer: ptl.Trainer = None):
        super().__init__(cfg=cfg, trainer=trainer)

        # input embedding stem: drop(content + position)
        self.embedding = self.from_config_dict(self.cfg.embedding)
        # deep transformer: just a sequence of transformer blocks
        self.encoder = self.from_config_dict(self.cfg.encoder)
        # decoder: at the end one more layernorm and decode the answers
        self.decoder = self.from_config_dict(self.cfg.decoder)

        self.block_size = self.cfg.embedding.block_size
        self.apply(self._init_weights)

        print("number of parameters: %e" % self.num_weights)

    @typecheck()
    def forward(self, idx):
        b, t = idx.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        # forward the GPT model
        # Remember: Only kwargs are allowed !
        e = self.embedding(idx=idx)
        x = self.encoder(embed=e)
        logits = self.decoder(encoding=x)

        return logits

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        """
        Vanilla model initialization:
        - all MatMul weights \in N(0, 0.02) and biases to zero
        - all LayerNorm post-normalization scaling set to identity, so weight=1, bias=0
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    @property
    def input_types(self):
        return {
            'idx': NeuralType(('B', 'T'), Index())
        }

    @property
    def output_types(self):
        return {
            'logits': NeuralType(('B', 'T', 'C'), LogitsType())
        }
    

def get_class_path(cls):
  return f'{cls.__module__}.{cls.__name__}'


common_config = OmegaConf.create({
    'vocab_size': MISSING,
    'block_size': MISSING,
    'n_layer': MISSING,
    'n_embd': MISSING,
    'n_head': MISSING,
})


embedding_config = OmegaConf.create({
    '_target_': get_class_path(GPTEmbedding),
    'vocab_size': '${model.vocab_size}',
    'n_embd': '${model.n_embd}',
    'block_size': '${model.block_size}',
    'embd_pdrop': 0.2
})

encoder_config = OmegaConf.create({
    '_target_': get_class_path(GPTTransformerEncoder),
    'n_embd': '${model.n_embd}',
    'block_size': '${model.block_size}',
    'n_head': '${model.n_head}',
    'n_layer': '${model.n_layer}',
    'attn_pdrop': 0.2,
    'resid_pdrop': 0.2
})

decoder_config = OmegaConf.create({
    '_target_': get_class_path(GPTDecoder),
    # n_embd: int, vocab_size: int
    'n_embd': '${model.n_embd}',
    'vocab_size': '${model.vocab_size}'
})

model_config = OmegaConf.create({
    'model': common_config
})

# Then let's attach the sub-module configs

OmegaConf.set_struct(model_config.model, False)
model_config.model.embedding = embedding_config
model_config.model.encoder = encoder_config
model_config.model.decoder = decoder_config
OmegaConf.set_struct(model_config.model, True)



class BasicNeMoGPT(AbstractNeMoGPT):
    
    @classmethod
    def list_available_models(cls) -> PretrainedModelInfo:
        return None

    def step_(self, split, batch, batch_idx=None):
        idx, targets = batch
        logits = self(idx=idx)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        key = 'loss' if split == 'train' else f"{split}_loss"
        self.log(key, loss)
        return {key: loss}

    def training_step(self, *args, **kwargs):
        return self.step_('train', *args, **kwargs)

    def validation_step(self, *args, **kwargs):
        return self.step_('val', *args, **kwargs)

    def test_step(self, *args, **kwargs):
        return self.step_('test', *args, **kwargs)
        
    # This is useful for multiple validation data loader setup
    def multi_validation_epoch_end(self, outputs, dataloader_idx: int = 0):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'val_loss': val_loss_mean}

    # This is useful for multiple test data loader setup
    def multi_test_epoch_end(self, outputs, dataloader_idx: int = 0):
        test_loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()
        return {'test_loss': test_loss_mean}

    def configure_optimizers(self):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('embedding.pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": self.cfg.optim.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=self.cfg.optim.lr, betas=self.cfg.optim.betas)
        return optimizer
    
    def setup_training_data(self, train_data_config: OmegaConf):
        self._train_dl = None
    
    def setup_validation_data(self, val_data_config: OmegaConf):
        self._validation_dl = None
    
    def setup_test_data(self, test_data_config: OmegaConf):
        self._test_dl = None    

# Datasets 
# TS related classes
class TinyShakespeareDataset(Dataset):
    def __init__(self, data_path, block_size, crop=None, override_vocab=None):

        # load the data and crop it appropriately
        with open(data_path, 'r') as f:
            if crop is None:
                data = f.read()
            else:
                f.seek(crop[0])
                data = f.read(crop[1])

        # build a vocabulary from data or inherit it
        vocab = sorted(list(set(data))) if override_vocab is None else override_vocab

        # Add UNK
        special_tokens = ['<PAD>', '<UNK>']  # We use just <UNK> and <PAD> in the call, but can add others.
        if not override_vocab:
            vocab = [*special_tokens, *vocab]  # Update train vocab with special tokens

        data_size, vocab_size = len(data), len(vocab)
        print('data of crop %s has %d characters, vocab of size %d.' % (str(crop), data_size, vocab_size))
        print('Num samples in dataset : %d' % (data_size // block_size))

        self.stoi = { ch:i for i,ch in enumerate(vocab) }
        self.itos = { i:ch for i,ch in enumerate(vocab) }
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = data
        self.vocab = vocab
        self.special_tokens = special_tokens

    def __len__(self):
        return len(self.data) // self.block_size

    def __getitem__(self, idx):
        # attempt to fetch a chunk of (block_size + 1) items, but (block_size) will work too
        chunk = self.data[idx*self.block_size : min(len(self.data), (idx+1)*self.block_size + 1)]
        # map the string into a sequence of integers
        ixes = [self.stoi[s] if s in self.stoi else self.stoi['<UNK>'] for s in chunk ]
        # if stars align (last idx and len(self.data) % self.block_size == 0), pad with <PAD>
        if len(ixes) < self.block_size + 1:
            assert len(ixes) == self.block_size # i believe this is the only way this could happen, make sure
            ixes.append(self.stoi['<PAD>'])
        dix = torch.tensor(ixes, dtype=torch.long)
        return dix[:-1], dix[1:]

    @property
    def output_types(self):
        return {
            'input': NeuralType(('B', 'T'), Index()),
            'target': NeuralType(('B', 'T'), LabelsType())
        }

class NeMoGPT_TS(BasicNeMoGPT):
    def _setup_data_loader(self, cfg):
        if self.vocab is None:
            override_vocab = None
        else:
            override_vocab = self.vocab

        dataset = TinyShakespeareDataset(
            data_path=cfg.data_path,
            block_size=cfg.block_size,
            crop=tuple(cfg.crop) if 'crop' in cfg else None,
            override_vocab=override_vocab
        )

        if self.vocab is None:
            self.vocab = dataset.vocab

        return DataLoader(
            dataset=dataset,
            batch_size=cfg.batch_size,
            shuffle=cfg.shuffle,
            collate_fn=dataset.collate_fn,  # <-- this is necessary for type checking
            pin_memory=cfg.pin_memory if 'pin_memory' in cfg else False,
            num_workers=cfg.num_workers if 'num_workers' in cfg else 0
        )
    
    def setup_training_data(self, train_data_config: OmegaConf):
        self.vocab = None
        self._train_dl = self._setup_data_loader(train_data_config)

        # Save the vocab into a text file for now
        with open('vocab.txt', 'w') as f:
            for token in self.vocab:
                f.write(f"{token}<SEP>")
        
        # This is going to register the file into .nemo!
        # When you later use .save_to(), it will copy this file into the tar file.
        self.register_artifact('vocab_file', 'vocab.txt')
    
    def setup_validation_data(self, val_data_config: OmegaConf):
        # This is going to try to find the same file, and if it fails, 
        # it will use the copy in .nemo
        vocab_file = self.register_artifact('vocab_file', 'vocab.txt')
    
        with open(vocab_file, 'r') as f:
            vocab = []
            vocab = f.read().split('<SEP>')[:-1]  # the -1 here is for the dangling <SEP> token in the file
            self.vocab = vocab

        self._validation_dl = self._setup_data_loader(val_data_config)
    
    def setup_test_data(self, test_data_config: OmegaConf):
        # This is going to try to find the same file, and if it fails, 
        # it will use the copy in .nemo
        vocab_file = self.register_artifact('vocab_file', 'vocab.txt')

        with open(vocab_file, 'r') as f:
            vocab = []
            vocab = f.read().split('<SEP>')[:-1]  # the -1 here is for the dangling <SEP> token in the file
            self.vocab = vocab

        self._test_dl = self._setup_data_loader(test_data_config)

# SAR related classes
class SyntheticAssociativeRecall(Dataset):
    def __init__(self, data_path, split, vocab_size, block_size, num_examples=None,**kwargs):

        self.num_examples = num_examples or 4000
        self.block_size = block_size
        self.vocab_size = vocab_size
        # the data is already tokenized so no vocab available:
        self.vocab=None
        data_tensor = torch.load(os.path.join(data_path,f"{split}_assoc_recall_{self.num_examples}_{self.vocab_size}_{self.block_size}.pt"))
        self.data = TensorDataset(data_tensor[:, 0, :], data_tensor[:, 1, :])

    def __len__(self):
        return len(self.data)

    @property
    def output_types(self):
        return {
            'input': NeuralType(('B', 'T'), Index()),
            'target': NeuralType(('B', 'T'), LabelsType())
        }

class NeMoGPT_SAR(BasicNeMoGPT):
    def _setup_data_loader(self, cfg):

        dataset = SyntheticAssociativeRecall(
            data_path=cfg.data_path,
            split=cfg.split,    # train or test
            vocab_size=cfg.vocab_size,
            num_examples=cfg.num_examples,
            block_size=cfg.block_size
        )

        self.vocab = dataset.vocab

        return DataLoader(
            dataset=dataset.data,
            batch_size=cfg.batch_size,
            shuffle=cfg.shuffle,
            collate_fn=dataset.collate_fn,  # <-- this is necessary for type checking
            pin_memory=cfg.pin_memory if 'pin_memory' in cfg else True,
            num_workers=cfg.num_workers if 'num_workers' in cfg else 4
        )
    
    def setup_training_data(self, train_data_config: OmegaConf):
        self.vocab = None
        self._train_dl = self._setup_data_loader(train_data_config)
    
    def setup_validation_data(self, val_data_config: OmegaConf):
        self._validation_dl = self._setup_data_loader(val_data_config)
    
    def setup_test_data(self, test_data_config: OmegaConf):
        self._test_dl = self._setup_data_loader(test_data_config)



DATASET="TS"       # can be either SAR or TS

# TS - TinyShakespear
if DATASET=='TS':  # TS - TinyShakespear
    model_cls=NeMoGPT_TS
    block_size = 128
    data_path='./sandbox/tiny-shakespeare.txt'
    train_dataset = TinyShakespeareDataset(data_path, block_size, crop=(0,         int(1e6)))

    train_ds = OmegaConf.create({
        'data_path': '${model.data_path}',
        'block_size': '${model.block_size}',
        'crop': [0, int(6e5)],
        'batch_size': 64,
        'shuffle': True,
    })

    validation_ds = OmegaConf.create({
        'data_path': '${model.data_path}',
        'block_size': '${model.block_size}',
        'crop': [int(6e5), int(2e5)],
        'batch_size': 64,
        'shuffle': False,
    })

    test_ds = OmegaConf.create({
        'data_path': '${model.data_path}',
        'block_size': '${model.block_size}',
        'crop': [int(8e5), int(1.15e6)],
        'batch_size': 64,
        'shuffle': False,
    })

    optim_config = OmegaConf.create({
        'lr': 3e-4,
        'weight_decay': 0.1,
        'betas': [0.9, 0.98]
    })



    OmegaConf.set_struct(model_config.model, False)
    # Set the data path and update vocabular size
    model_config.model.data_path = data_path
    model_config.model.vocab_size = train_dataset.vocab_size
    model_config.model.block_size = block_size

    model_config.model.n_layer = 6
    model_config.model.n_embd = 384
    model_config.model.n_head = 6

    model_config.model.train_ds = train_ds
    model_config.model.validation_ds = validation_ds
    model_config.model.test_ds = test_ds

    model_config.model.optim = optim_config


    OmegaConf.set_struct(model_config.model, True)

    model = NeMoGPT_TS(cfg=model_config.model)

    # configure the trainer
    trainer_config = OmegaConf.create({
        'devices': 1,
        'num_nodes': 1,
        'accelerator': 'gpu',
        'precision': 16,
        'max_epochs': 100,
        'enable_checkpointing': False,
        'use_distributed_sampler': False,
        # 'max_epochs': -1, # PTL default. In practice, max_steps will be reached first. 
        # 'max_steps': 100000, # consumed_samples = global_step * micro_batch_size * data_parallel_size * accumulate_grad_batches
        'log_every_n_steps': 10,
        # 'val_check_interval': 100,
        # 'limit_val_batches': 50,
        # 'limit_test_batches': 500,
        # 'accumulate_grad_batches': 1, # do not modify, grad acc is automatic for training megatron models
        'gradient_clip_val': 0.2,
        'benchmark': False,
        'enable_model_summary': False, # default PTL callback for this does not support model parallelism, instead we log manually
    })



elif DATASET=='SAR': # Synthetic Associative Recall
    model_cls=NeMoGPT_SAR
    data_path=os.path.abspath('./sandbox')
    block_size=1024
    vocab_size=40
    num_examples = 4000

    train_ds = OmegaConf.create({
        'data_path': data_path,
        'split': 'train',
        'num_examples':num_examples,
        'block_size': block_size,
        'vocab_size': vocab_size,
        'batch_size': 16,
        'shuffle': True,
    })

    validation_ds = OmegaConf.create({
        'data_path': '${model.data_path}',
        'split': 'test',
        'num_examples':num_examples,
        'block_size': block_size,
        'vocab_size': vocab_size,    
        'batch_size': 16,
        'shuffle': False,
    })

    optim_config = OmegaConf.create({
        'lr': 5e-4,
        'weight_decay': 0.1,
        'betas': [0.9, 0.999]
    })


    OmegaConf.set_struct(model_config.model, False)
    # Set the data path and update vocabular size
    model_config.model.data_path = data_path
    model_config.model.vocab_size = vocab_size
    model_config.model.block_size = block_size+2

    model_config.model.n_layer = 2
    model_config.model.n_embd = 64
    model_config.model.n_head = 1

    model_config.model.train_ds = train_ds
    model_config.model.validation_ds = validation_ds

    model_config.model.optim = optim_config

    OmegaConf.set_struct(model_config.model, True)    

    model = NeMoGPT_SAR(cfg=model_config.model)

    # configure the trainer
    # trainer config for AssocRecall:
    trainer_config = OmegaConf.create({
        'devices': 1,
        'num_nodes': 1,
        'accelerator': 'gpu',
        'precision': 16,
        'max_epochs': 100,
        'enable_checkpointing': False,
        'use_distributed_sampler': False,
        # 'max_epochs': -1, # PTL default. In practice, max_steps will be reached first. 
        # 'max_steps': 100000, # consumed_samples = global_step * micro_batch_size * data_parallel_size * accumulate_grad_batches
        'log_every_n_steps': 10,
        # 'val_check_interval': 100,
        # 'limit_val_batches': 50,
        # 'limit_test_batches': 500,
        # 'accumulate_grad_batches': 1, # do not modify, grad acc is automatic for training megatron models
        'gradient_clip_val': 0.2,
        'benchmark': False,
        'enable_model_summary': False, # default PTL callback for this does not support model parallelism, instead we log manually
    })

# wandb_logger = ptl.loggers.WandbLogger(project="nemo-mingpt-shakespeare")
# wandb_logger.experiment.config.update(OmegaConf.to_container(model_config))
# wandb_logger.experiment.config.update(OmegaConf.to_container(trainer_config))

# wandb_logger.watch(model, log="all", log_freq=trainer_config.log_every_n_steps)
wandb_logger=None

trainer = ptl.Trainer(**trainer_config,logger=wandb_logger)
tester = ptl.Trainer(devices=1, accelerator=accelerator, logger=None, limit_test_batches=1.0)

# test the model before training:
tester.test(model)


# train the model:
trainer.fit(model)
# wandb.finish()

# test the model after training:
tester.test(model)

# saving and reloading the model
# model.save_to(f'gpt_model_{DATASET}.nemo')
# temp_model = model_cls.restore_from(f'gpt_model_{DATASET}.nemo')
# temp_model.setup_multiple_test_data(temp_model.cfg.validation_ds)

# tester.test(model)
