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
        self.c_proj.NANOGPT_SCALE_INT = 1
        # regularization 
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # not really a 'bias', more of a mask, but following hte OpenAI/HF naming though
        self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size)).view(1,1,config.block_size, config.block_size))
        
    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality
        # calculate query, key, values for all heads in batch and move head forward to be the batch
        # nh is "number of heads", hs is "head size" and C (number of channels) = nh * hs
        # e.g in GPT-2 (124M), n_head = 12, hs=64, nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        
        ######################## Normal but slower attention
        # attention (materializes the large (T,T) matrix for all the queries and keys)
        # att = (q @ k.transpose(-2,-1)) * (1.0/math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        ##########################
        # Flash attention 
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        
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
        self.c_proj.NANOGPT_SCALE_INT = 1

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
        x = x + self.mlp(self.ln_2(x))
        return x 
    



# This is the config dataclass, it defines the hyperparameters for GPT-2
@dataclass 
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 end of text token
    n_layer: int = 12 # number of layers
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
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        # This is the final classifer, projects from 768 to vocab size,
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # weight sharing scheme (Used for better performance for similar semantics)
        self.transformer.wte.weight = self.lm_head.weight
        # this gets rid of a lot of parameters (30%)

        # init params, this applies init weigth functions to all params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # this is the weight initialization
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                # we are multiplyign by 2 because we are passing the x 2 times in MLP and ATTn
                std *=  (2* self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                # This is not the default in torch 
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx is of shape (B,T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of lenght {T}, block size is too big"
        # forward the tokens and position embeddings 
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # this is to make sure that this list is stored on the device
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)

        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier 
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size) 
        # let's calculate and return the loss 
        loss = None 
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        
        return logits, loss 


    # Classmethod makes it such that we do not need to initialize an instance of the class to use the function
    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel # type: ignore
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
# --------------------------------------------------------------------------
# prefix tokens
import tiktoken # type: ignore

class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        with open('input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B + T)} batches")

        # state
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        # Here is the sample code for the justification 
        # buf = torch.tensor(tokens[:24+1])
        # x = bug[:-1].view(4,6)
        # y = buf[1:].view(4,6)
        # Everything but the plus 1
        x = (buf[:-1]).view(B,T)
        # Everything except the first one 
        y = (buf[1:]).view(B, T)
        self.current_position += B *T
        # if loading the next batch would be otu of bounds, reset 
        if self.current_position + (B*T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y
        


# --------------------------------------------------------------------------
import time
# attempt to autodetect the device used 
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = 'mps' # this is for MACS
print(f"using device: {device}")

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

num_return_sequences = 5 
max_length = 30

# Very important line here, this does not seem to be affecting my CPU
# This may only affect GPUs !!!!!!!!!!!
torch.set_float32_matmul_precision('high')

# model = GPT.from_pretrained('gpt2')
# Increased 
model = GPT(GPTConfig(vocab_size=50304))
# model.eval()
model.to(device)
model = torch.compile(model)

train_loader = DataLoaderLite(B=4, T=32) # new GPT numbers 16, 1024

# Learning Rate optimizations 
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10 
max_steps = 50 

def get_lr(it):
    # 1. Linear warmup for warmup_iters steps 
    if it < warmup_steps: 
        return max_lr * (it+1) / warmup_steps 
    # 2 if it > lr_decay_iterns, return min learning rate 
    if it > max_steps: 
        return min_lr 
    # else, decay the learning rate in a cosine matter 
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff 
    # starts at 1 and goes to 0 
    return min_lr + coeff *(max_lr - min_lr)

# this learnign rate is a good rate 
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
# Let's overfit to a single batch 
for step in range(max_steps):
    t0 = time.time()
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    # always remember to 0 your gradient 
    optimizer.zero_grad()

    # More time savings!!!!!!, using Bfloat16
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits, loss = model(x, y)
    loss.backward()
    # Optimization 
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
     # determine and set the learning rate for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    # step updates the parameters to decrease the loss
    optimizer.step()
    # usually, the 
    # torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0)*1000
    # Let's add tokens per second which is a very important metric 
    tokens_per_sec = (train_loader.B * train_loader.T) / (t1 -  t0)
    # when we call .item and ships it back to the cpu that is turned into a float 
    print(f"step {step:4d} | loss: {loss.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")




# Let's skip the end for now
import sys; sys.exit(0) 


tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long) # (8,)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (5, 8)
x = tokens.to(device)

# generate! right now x is (B, T) where B = 5, T = 8
# set the seed to 42
torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_length:
    # forward the model to get the logits
    with torch.no_grad():
        logits = model(x) # (B, T, vocab_size)
        # take the logits at the last position
        logits = logits[:, -1, :] # (B, vocab_size)
        # get the probabilities
        probs = F.softmax(logits, dim=-1)
        # do top-k sampling of 50 (huggingface pipeline default)
        # topk_probs here becomes (5, 50), topk_indices is (5, 50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # select a token from the top-k probabilities
        # note: multinomial does not demand the input to sum to 1
        ix = torch.multinomial(topk_probs, 1) # (B, 1)
        # gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
        # append to the sequence
        x = torch.cat((x, xcol), dim=1)

# print the generated text
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)