from dataclasses import dataclass 
import torch 
import torch.nn as nn 
from torch.nn import functional as F 

class SelfAttention(nn.Module): 
 
    def __init__(self, config): 
        super().__init__() 
        assert config.n_embd % config.n_head == 0 
        # key, query, value projections for all heads, but in a batch 
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd) 
        # output projection 
        self.c_proj = nn.Linear(config.n_embd, config.n_embd) 
        self.c_proj.NANOGPT_SCALE_INIT = 1 
        # regularization 
        self.n_head = config.n_head 
        self.n_embd = config.n_embd 
        # not really a 'bias', more of a mask, but following the OpenAI/HF naming though 
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)) 
                                     .view(1, 1, config.block_size, config.block_size)) 
 
    def forward(self, x): 
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd) 
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim 
        qkv = self.c_attn(x) 
        q, k, v = qkv.split(self.n_embd, dim=2) 
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs) 
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs) 
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs) 
        y = F.scaled_dot_product_attention(q, k, v) # non-causal self-attention
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side 
        # output projection 
        y = self.c_proj(y) 
        return y 
 
class MLP(nn.Module): 
 
    def __init__(self, config): 
        super().__init__() 
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd) 
        self.gelu    = torch.nn.GELU(approximate='none') 
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd) 
        self.c_proj.NANOGPT_SCALE_INIT = 1 
 
    def forward(self, x): 
        x = self.c_fc(x) 
        x = self.gelu(x) 
        x = self.c_proj(x) 
        return x 
 
class Block(nn.Module): 
 
    def __init__(self, config): 
        super().__init__() 
        self.ln_1 = nn.LayerNorm(config.n_embd) 
        self.attn = SelfAttention(config) 
        self.ln_2 = nn.LayerNorm(config.n_embd) 
        self.mlp = MLP(config) 
 
    def forward(self, x): 
        x = x + self.attn(self.ln_1(x)) 
        x = x + self.mlp(self.ln_2(x)) 
        return x 
 
@dataclass 
class HTConfig: 
    block_size: int = 32 # max word length is 31, but use 2^5 for optimazied GPU kernel dispatches
    vocab_size: int = 28 # 26 letters + '.' (end symbol) + '_' (ungussed letter symbol) 
    n_layer: int = 16 
    n_head: int = 12 
    n_embd: int = n_head*64
 
class HangmanTransformer(nn.Module): 
 
    def __init__(self, config): 
        super().__init__() 
        self.config = config 
 
        self.transformer = nn.ModuleDict(dict( 
            wte = nn.Embedding(config.vocab_size, config.n_embd), 
            wpe = nn.Embedding(config.block_size, config.n_embd), 
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), 
            ln_f = nn.LayerNorm(config.n_embd), 
        )) 
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) 
 
        # init params 
        self.apply(self._init_weights) 
 
    def _init_weights(self, module): 
        if isinstance(module, nn.Linear): 
            std = 0.02 
            if hasattr(module, 'NANOGPT_SCALE_INIT'): 
                std *= (2 * self.config.n_layer) ** -0.5 
            torch.nn.init.normal_(module.weight, mean=0.0, std=std) 
            if module.bias is not None: 
                torch.nn.init.zeros_(module.bias) 
        elif isinstance(module, nn.Embedding): 
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02) 
 
    def forward(self, idx, targets=None, return_logits=True): 
        device = idx.device 
        b, t = idx.size() 
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}" 
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t) 
 
        # forward the model itself 
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd) 
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd) 
        x = tok_emb + pos_emb 
 
        for block in self.transformer.h: 
            x = block(x) 
        x = self.transformer.ln_f(x) 
        logits = self.lm_head(x) # (B, T, vocab_size) 
        loss = None 
        if targets is not None: 
            loss = F.cross_entropy(logits.mean(1), targets.view(-1)) 
 
        return logits, loss