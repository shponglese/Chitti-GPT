from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        #key, query, value projects for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd)

        #output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        #regularization

        self.n_head = config.n_head
        self.n_embd = config.n_embd

        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                .view(1,1,config.block_size, config.block_size))
    
    def forward(self, x):
        B, T, C = x.size() #batch size, sequence length, embedding dimensionality (n_embd)

        #calculate query, key, values for all heads in batch and move head forward to be the batch
        #nh is "number of heads", hs is "head size" and C (number of channels ) = nh * hs
        #eg in GPT2 (124M), h_head=12, hs(64), so nh * hs = C = 768 channels in the Transformer

        qkv = self.c_attn(x)

        q, k, v = qkv.split(self.n_embd, dim=2)

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1,2) #(b, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1,2) #(b, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1,2) #(b, nh, T, hs) 

        #attention (materializes the large (T,T) matrix for all the queries and keys)
        att = (q @ k.transpose(-2,-1)) * (1.0/math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T]==0, float('-inf'))
        att = F.softmax(att, dim=-1)

        y = att @ v #(B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1,2).contiguous().view(B,T,C)

        #output projection
        y = self.c_proj(y)
        return y
       
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4* config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4* config.n_embd, config.n_embd)
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x



class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        selfln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    


@dataclass
class GPTConfig:
    block_size: int = 1024 #max sequence length
    vocab_size: int = 50257 #number of tokens : 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|>
    n_layer: int = 12 #number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 #embedding dimension

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), #Hidden
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    
    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        
        from transformers import GPT2LMHeadModel
        
        print("loading weights from pretrained gpt: %s" % model_type)

        #n_layer, n_head and n_embd are determined from model_type

        config_args = {
            'gpt2' : dict(n_layer=12, n_head=12, n_embd = 768), #124M params
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd = 1024), #350M params
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd = 1280), #774M params
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd = 1600), #1558M params
        }[model_type]

        config_args['vocab_size'] = 50257 #always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 #always 1024 for GPT model checkpoints

        #create a from-scractch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)

        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] #discord this mask/ buffer 

        #init a huggingface/transformers model

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        

