from dataclasses import dataclass 
import torch 
import torch.nn as nn 
from torch.nn import functional as F 
import math 

# -------------------------------------------------------------------

# Now we write the Attention Module 
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch 
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projeciton 
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization 
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # not really a 'bias', more of a mask, but following hte OpenAI/HF naming though
        self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config,)))
        
    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality
        # calculate query, key, values for all heads in batch and move head forward to be the batch
        # nh is "number of heads", hs is "head size" and C (number of channels) = nh * hs
        # e.g in GPT-2 (124M), n_head = 12, hs=64, nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1,2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1,2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1,2)
        # attention (materializes the large (T,T) matrix for all the queries and keys)
        att = (q @ k.transpose(-2,-1)) * (1.0/math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=1)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        # This is a concatenation expression 
        y = y.transpose(1,2).contiguous().view(B,T,C)
        # output projection 
        y = self.c_proj(y)
        return y 


# Very simple MLP that puts the input dimension into 4 times the size
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        # The dead relu neuron problem, if the relu is 0, then there is no contribution 
        # Smoothing out works better in reality 
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4*config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x 

# This is one round of attention and MLP 
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Everything is pretty standard for an LLM
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, x):
        # We normalize and add before every single attention block and MLP
        # We add the results to itself in order to make backpropogation easier 
        # This can go against the dissapearing gradient problem
        x = x + self.attn(self.ln_1(x))
        x = x + self.attn(self.ln_2(x))
        return x 

# This is the config dataclass, it defines the hyperparameters for GPT-2
@dataclass 
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 end of text token
    n_layers: int = 12 # number of layers
    n_head: int = 12 # number of heads 
    n_embd: int = 768 # embedding dimension

# This is the actual PyTorch module for GPT 
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Configure the transformer
        # wte = weights of the token embedding
        # wpe = weights of the position embedding
        # h = hidden, this is a module list, indexed, n_layer blocks
        # ln_f = a final layer normalization
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embed),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        # This is the final classifer, projects from 768 to vocab size,
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
    # Classmethod makes it such that we do not need to initialize an instance of the class to use the function
    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrianed GPT-2 model weights from huggingface"""
        assert model_type in {'gpt-2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print(f"loading weights from pretrained gpt: {model_type}")

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768), # 124m params
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024), # 350m params
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280), # 774m params
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600), # 350m params
        }[model_type]

        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # alwasy 1024 for GPT model checkpoints
        # create a from-scratch intialized minGPT model 
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer

        # init a hugginface/transformers model 
        model_hf = GPT2LMHeadModel.from_pretrained('GPT2')
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj_weight', 'mlp.c_fc.weight', 'mlp.c_proj.'] # TODO 
        # basicaly the openai checkpoints use a "Conv1D" module, but we ony want to use a 
        

