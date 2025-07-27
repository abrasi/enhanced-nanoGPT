import os
import math
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from hellaswag import render_example, iterate_examples
from zeus.monitor import ZeusMonitor

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--w_load", type=float, default=0.0)
    parser.add_argument("--w_importance", type=float, default=0.0)
    parser.add_argument("--w_penalty", type=float, default=0.0)
    parser.add_argument("--lambda_z", type=float, default=0.0)
    parser.add_argument("--bias", type=bool, default=False)
    
    parser.add_argument("--moe_implementation", type=str, default="OLNNMoE")
    parser.add_argument("--exp_name", type=str, default=None)
    return parser.parse_args()

# -----------------------------------------------------------------------------

 
class FFN(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x  
    
class MoA(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        
        assert config.n_embd % config.n_head == 0
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        
        # Número de expertos
        self.n_experts = config.n_experts
        assert self.n_experts > 0
        
        # Matrices compartidas
        self.Wk = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.Wv = nn.Linear(config.n_embd, config.n_embd, bias=False)

        # Definición dos expertos
        self.Wq = nn.ModuleList(
            [nn.Linear(config.n_embd, config.n_embd, bias=False) for _ in range(self.n_experts)]
        )
        self.Wo = nn.ModuleList(
            [nn.Linear(config.n_embd, config.n_embd, bias=False) for _ in range(self.n_experts)]
        )
            
        # Pesos do gate
        self.gate = nn.Linear(config.n_embd, self.n_experts, bias=False)
        # Pesos de ruido gaussiano
        self.noise_layer = nn.Linear(config.n_embd, self.n_experts, bias=False)
        # Bias de DeepSeek
        if config.bias:
            self.register_buffer("bias", torch.zeros(self.n_experts, dtype=torch.float32))
        else:
            self.bias = None
        
        self.register_buffer("total_assigment_count", torch.zeros(self.n_experts, dtype=torch.long))
        
        # Top-k expertos a seleccionar por token
        self.topk = config.topk
        
        # Peso para o loss da importancia dos expertos
        self.w_importance = config.w_importance
        # Peso para o loss de balanceo de carga dos expertos
        self.w_load = config.w_load
        # Peso para o loss de penalización de logits grandes
        self.lambda_z = config.lambda_z
        # Peso para o loss de Brais
        self.w_penalty = config.w_penalty
    
    @torch.no_grad()
    def reset_stats(self):
        self.total_assigment_count.zero_()
        
    def forward(self, x):

        B,T, _ = x.shape

        auxiliary_loss = 0.0
              
        # Aplanamos as entradas x: (B*T, C)      
        x_squashed = x.view(-1, x.shape[-1])
        
        # (B*T, n_experts) -> por cada token obtéñense os logits de cada experto
        gate_logits = self.gate(x_squashed)
        
        # Calculamos o penalty da implementación de Brais
        if self.w_penalty > 0:
            self.penalty = self.get_penalty(gate_logits)
            auxiliary_loss += self.penalty
        
        # Calculamos o z-loss
        if self.lambda_z > 0:
            self.z_loss = self.get_z_loss(gate_logits)
            auxiliary_loss += self.z_loss
        
        # Introducimos ruido gaussiano ós logits dos expertos
        noise_std = F.softplus(self.noise_layer(x_squashed))
        gate_logits += torch.randn_like(gate_logits) * noise_std
        
        # Engadimos o bias de DeepSeek
        if self.bias is not None:
            gate_logits += self.bias
        
        # Aplicamos a estratexia top-k:
        # weights representa as puntuacións dos top-k expertos
        # selected_experts contén os indices dos topk expertos seleccionados para cada token
                    # selected_experts =
                    # [[0, 2],  # Token 0 → Expertos 0 y 2
                    # [1, 3],  # Token 1 → Expertos 1 y 3
                    # [0, 3],  # Token 2 → Expertos 0 y 3
                    # [1, 2]]  # Token 3 → Expertos 1 y 2 
        weights, selected_experts = torch.topk(gate_logits, self.topk)
    
        # Aplicamos softmax para obter os scores de cada experto, contén a información de canto contribuirá cada experto
        weights = F.softmax(
            weights,
            dim=1,
            dtype=torch.float,
        ).type_as(x)
        
        # Proxeccións compartidas entre os expertos
        k = (self.Wk(x)).view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, T, head_dim)
        v = (self.Wv(x)).view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, T, head_dim)
        
        # Calculamos o loss que equilibra a importancia de cada experto
        if self.w_importance > 0:
            self.importance_loss = self.get_importance_loss(weights, selected_experts)
            auxiliary_loss += self.importance_loss
                    
        # Calculamos o loss que equilibra a carga entre os expertos
        if self.w_load > 0:
            # Para seguir correctamente o paper de Switch Transformer, o loss de carga ten que calcularse cos logits do gate, sen topk
            router_probs = F.softmax(gate_logits, dim=-1, dtype=torch.float).type_as(x)
            top1_experts = selected_experts[:, 0]
            self.load_loss = self.get_load_loss(router_probs, top1_experts)
            auxiliary_loss += self.load_loss
        
        # Engadimos as asignacións dos expertos o reconto total
        with torch.no_grad():
            top1_experts = selected_experts[:, 0].view(-1)
            ones = torch.ones_like(top1_experts, dtype=torch.long)
            # top1_experts contén os indices dos expertos seleccionados para cada token, sumamos 1 para cada indice de experto seleccionado
            self.total_assigment_count.scatter_add_(0, top1_experts, ones)

        # results conterá os resultados finais
        results = torch.zeros_like(x_squashed)
        
        for i in range(self.n_experts):
            # Obtemos batch_idx e nth_expert:
                # batch_idx representa os indices dos tokens que van ao experto i
                # nth_expert representa en que índice dos top-k expertos se atopa o experto i
                        # selected_experts =
                        # [[0, 2],   # token 0 → expertos 0 e 2
                        #  [1, 3],   # token 1 → expertos 1 e 3
                        #  [0, 3],   # token 2 → expertos 0 e 3
                        #  [1, 2]]   # token 3 → expertos 1 e 2
                        
                        # batch_idx = [0, 3]    # Tokens 0 e 3 van ó experto 2
                        # nth_expert = [1, 1]   # Dos tokens 0 e 3, o experto 2 é o segundo, índice 1
            batch_idx, nth_expert = torch.where(selected_experts == i)
            
            if batch_idx.numel() > 0:
                
                b_idx = batch_idx // T
                t_idx = batch_idx % T
                           
                # .view(B, T, self.n_head, self.head_dim).transpose(1, 2)     
                # Pasamos os tokens seleccionados ó experto i
                
                # print(x[b_idx, t_idx].shape)
                q = (self.Wq[i](x_squashed[b_idx, t_idx])).view(-1, self.n_head, self.head_dim).unsqueeze(2)

                # print("shape: ", q.shape, k.shape, v.shape)

                # A memoria das GPUs agótase neste punto, e o adestramento falla
                k_sel = k[b_idx]
                v_sel = v[b_idx]
                
                attention = F.scaled_dot_product_attention(
                    q, k_sel, v_sel,
                    is_causal=True
                ).squeeze(2)
                
                attention = attention.transpose(1, 2).contiguous()
                attention = attention.view(-1, self.n_embd)
                # print("att shape: ", attention.shape)
                attention = self.Wo[i](attention)  # Proxección final do experto i

                # Multiplicamos a saída do experto polo peso que lle corresponde, añádese unha dimensión para sexan compatibles
                # weights[batch_idx, nth_expert, None] contén a contribución do experto i para cada token
                results[batch_idx] += weights[batch_idx, nth_expert, None] * attention
                        
        # Devolvemos á forma orixinal das entradas e retornamos as perdas auxiliares, os expertos seleccionados e os pesos
        return results.view_as(x), auxiliary_loss
        

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = MoA(config)       
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.ffn = FFN(config)

    def forward(self, x):
        # Aquí, na atención, os tokens "comunícanse" entre eles
        x1, auxiliary_loss = self.attn(self.ln_1(x))
        x1 = x + x1 # Conexión residual
        # Aquí os tokens "pensan" individualmente
        x2 = x1 + self.ffn(self.ln_2(x1))
        return x2, auxiliary_loss

@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension
    
    n_experts: int = 4
    topk: int = 2

class GPT(nn.Module):

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

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

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

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        
        
        # ----- Perda auxiliar para MoA -----
        total_auxilary_loss = 0
        num_blocks = len(self.transformer.h)
        
        # Facemos pasar as entradas por cada bloque do transformer
        for block in self.transformer.h:
            x, auxilary_loss = block(x)
            total_auxilary_loss += auxilary_loss
        
        mean_auxilary_loss = total_auxilary_loss / num_blocks
        # ----- Perda auxiliar para MoA -----
        
        
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss += mean_auxilary_loss
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
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

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        if master_process:
            print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

# -----------------------------------------------------------------------------
import tiktoken
import numpy as np

def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32) # added after video
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        # get the shard filenames
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y

# -----------------------------------------------------------------------------
# helper function for HellaSwag eval
# takes tokens, mask, and logits, returns the index of the completion with the lowest loss

def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm

# -----------------------------------------------------------------------------

# Funcion que actuaiza el bias en cada capa MoE del modelo
def update_moe_layer_bias(moe_layer, gamma):
    # Sumar los tokens asignados a cada experto en esta capa
    tokens_per_expert = moe_layer.total_assigment_count.float().to(device)
    
    # Recogemos los tokens de todos los nodos, cuello de botella?
    if dist.is_initialized():
        dist.all_reduce(tokens_per_expert, op=dist.ReduceOp.AVG)
    
    # Promedio local de tokens para este MoE layer
    avg_tokens = tokens_per_expert.mean()
    
    # Definir umbrales locales
    overload_threshold = avg_tokens * 1.2
    underload_threshold = avg_tokens * 0.8

    # Determinar índices de expertos sobrecargados e infracargados
    overloaded = (tokens_per_expert > overload_threshold).nonzero(as_tuple=True)[0]
    underloaded = (tokens_per_expert < underload_threshold).nonzero(as_tuple=True)[0]
    
    with torch.no_grad():
        if overloaded.numel() > 0:
            moe_layer.bias[overloaded] -= gamma
        if underloaded.numel() > 0:
            moe_layer.bias[underloaded] += gamma    
            
def get_lr(it, warmup_steps, max_lr, min_lr, max_steps):
        # 1) linear warmup for warmup_iters steps
        if it < warmup_steps:
            return max_lr * (it+1) / warmup_steps
        # 2) if it > lr_decay_iters, return min learning rate
        if it > max_steps:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
        return min_lr + coeff * (max_lr - min_lr)      

def run_training(gptconfig: GPTConfig, log_dir):
        # create model
        model = GPT(gptconfig)
        # model = GPT.from_pretrained("gpt2") # or init from OpenAI GPT-2
        model.to(device)
        use_compile = True # torch.compile interferes with HellaSwag eval and Generation. TODO fix
        if use_compile:
            model = torch.compile(model)
        if ddp:
            model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=True)
        raw_model = model.module if ddp else model # always contains the "raw" unwrapped model

        max_lr = 6e-4
        min_lr = max_lr * 0.1
        warmup_steps = 715
        max_steps = 19073 # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens

        # Hiperparámetro para o axuste do bias en MoA
        gamma = 0.001
        bias_update_step = int(max_steps * 0.97) # Más o menos en DeepSeek el bias de Gating de MoE pasa a ser 0 a partir de este punto

        # Configurar o optimizador
        optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device_type)

        # create the log directory we will write checkpoints to and log to
        log_file = os.path.join(log_dir, "log.txt")
        with open(log_file, "w") as f: # open for writing to clear the file
            pass

        # Monitorización energia de las GPUs
        monitor = ZeusMonitor(gpu_indices=[torch.cuda.current_device()])

        monitor.begin_window("epoch")
        for step in range(max_steps):
            t0 = time.time()
            last_step = (step == max_steps - 1)
            
            # once in a while evaluate our validation loss
            if step % 250 == 0 or last_step:
                model.eval()
                val_loader.reset()
                with torch.no_grad():
                    val_loss_accum = 0.0
                    val_loss_steps = 20
                    for _ in range(val_loss_steps):
                        x, y = val_loader.next_batch()
                        x, y = x.to(device), y.to(device)
                        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                            logits, loss = model(x, y)
                        loss = loss / val_loss_steps
                        val_loss_accum += loss.detach()
                if ddp:
                    # Aqui se promedian los losses calculados entre las GPUs
                    dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
                if master_process:
                    print(f"validation loss: {val_loss_accum.item():.4f}")
                    with open(log_file, "a") as f:
                        f.write(f"{step} val {val_loss_accum.item():.4f}\n")
                    if step > 0 and (step % 5000 == 0 or last_step):
                        # optionally write model checkpoints
                        checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                        checkpoint = {
                            'model': raw_model.state_dict(),
                            'config': raw_model.config,
                            'step': step,
                            'val_loss': val_loss_accum.item()
                        }
                        # you might also want to add optimizer.state_dict() and
                        # rng seeds etc., if you wanted to more exactly resume training
                        torch.save(checkpoint, checkpoint_path)

            # once in a while evaluate hellaswag
            if (step % 250 == 0 or last_step) and (not use_compile):
                print("evaluating hella swag")
                num_correct_norm = 0
                num_total = 0
                for i, example in enumerate(iterate_examples("val")):
                    # only process examples where i % ddp_world_size == ddp_rank
                    if i % ddp_world_size != ddp_rank:
                        continue
                    # render the example into tokens and labels
                    _, tokens, mask, label = render_example(example)
                    tokens = tokens.to(device)
                    mask = mask.to(device)
                    # get the logits
                    with torch.no_grad():
                        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                            logits, loss = model(tokens)
                        pred_norm = get_most_likely_row(tokens, mask, logits)
                    num_total += 1
                    num_correct_norm += int(pred_norm == label)
                # reduce the stats across all processes
                if ddp:
                    num_total = torch.tensor(num_total, dtype=torch.long, device=device)
                    num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
                    dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
                    dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
                    num_total = num_total.item()
                    num_correct_norm = num_correct_norm.item()
                acc_norm = num_correct_norm / num_total
                if master_process:
                    print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
                    with open(log_file, "a") as f:
                        f.write(f"{step} hella {acc_norm:.4f}\n")

            # once in a while generate from the model (except step 0, which is noise)
            if ((step > 0 and step % 250 == 0) or last_step) and (not use_compile):
                model.eval()
                num_return_sequences = 4
                max_length = 32
                tokens = enc.encode("Hello, I'm a language model,")
                tokens = torch.tensor(tokens, dtype=torch.long)
                tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
                xgen = tokens.to(device)
                sample_rng = torch.Generator(device=device)
                sample_rng.manual_seed(42 + ddp_rank)
                while xgen.size(1) < max_length:
                    # forward the model to get the logits
                    with torch.no_grad():
                        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                            logits, loss = model(xgen) # (B, T, vocab_size)
                        # take the logits at the last position
                        logits = logits[:, -1, :] # (B, vocab_size)
                        # get the probabilities
                        probs = F.softmax(logits, dim=-1)
                        # do top-k sampling of 50 (huggingface pipeline default)
                        # topk_probs here becomes (5, 50), topk_indices is (5, 50)
                        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                        # select a token from the top-k probabilities
                        # note: multinomial does not demand the input to sum to 1
                        ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
                        # gather the corresponding indices
                        xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
                        # append to the sequence
                        xgen = torch.cat((xgen, xcol), dim=1)
                # print the generated text
                for i in range(num_return_sequences):
                    tokens = xgen[i, :max_length].tolist()
                    decoded = enc.decode(tokens)
                    print(f"rank {ddp_rank} sample {i}: {decoded}")

            # do one step of the optimization
            model.train()
            optimizer.zero_grad()
            
            # Actualización de bias para MoE
            if gptconfig.bias:
                
                if step == bias_update_step:
                    gamma = 0.0
                    if master_process:
                        print("Zeroed out the bias update speed in the MoE layer -> 97%% of training done")
                        
                if gamma > 0:
                    
                    for block in model.module.transformer.h: 
                        if hasattr(block, 'attn'): 
                            update_moe_layer_bias(block.attn, gamma)
            #--------------------------------           
            
            # Reseteamos os contadores de asignación de expertos para realizar o conteo despois do gradient accumulation
            for block in model.module.transformer.h:
                block.attn.reset_stats()

            # Inicializamos os acumuladores dos losses auxiliares
            aux_losses = {'importance_loss': 0.0, 'load_loss': 0.0, 'z_loss': 0.0, 'penalty': 0.0}
            
            loss_accum = 0.0
            for micro_step in range(grad_accum_steps):
                x, y = train_loader.next_batch()
                x, y = x.to(device), y.to(device)
                # added after video, this field is also used by the forward pass.
                if ddp:
                    model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                # we have to scale the loss to account for gradient accumulation,
                # because the gradients just add on each successive backward().
                # addition of gradients corresponds to a SUM in the objective, but
                # instead of a SUM we want MEAN. Scale the loss here so it comes out right
                loss = loss / grad_accum_steps
                loss_accum += loss.detach()
                
                with torch.no_grad():
                    for block in model.module.transformer.h:
                        moe = block.attn
                        aux_losses['importance_loss']  += getattr(moe, 'importance_loss', 0.0)
                        aux_losses['load_loss'] += getattr(moe, 'load_loss',       0.0)
                        aux_losses['z_loss']    += getattr(moe, 'z_loss',          0.0)
                        aux_losses['penalty']  += getattr(moe, 'penalty',         0.0)
                
                loss.backward()
                
            if ddp:
                dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
                
        
            # Promediamos as perdas auxiliares
            num_layers = len(model.module.transformer.h)
            for k in aux_losses:
                aux_losses[k] /= (grad_accum_steps * num_layers)
                if not torch.is_tensor(aux_losses[k]):
                    aux_losses[k] = torch.tensor(aux_losses[k], device=device)
                else:
                    aux_losses[k] = aux_losses[k].to(device)

            if ddp:
                for v in aux_losses.values():
                    dist.all_reduce(v, op=dist.ReduceOp.AVG)

            aux_losses = {k: v.item() for k, v in aux_losses.items()}
            #------------------------------------
            
            
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # determine and set the learning rate for this iteration
            lr = get_lr(step, warmup_steps, max_lr, min_lr, max_steps)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
                
                    
            # Conteo os tokens asignados a cada experto
            layer_counts = []
            layer_dropped_counts = []
            for block in model.module.transformer.h:
                
                moe = block.attn
                if hasattr(moe, 'total_assigment_count'):
                    layer_counts.append(moe.total_assigment_count.clone())
                    
                if hasattr(moe, "total_dropped_count"):
                    layer_dropped_counts.append(moe.total_dropped_count.clone())
                    
            # Promediamos os contadores de asignación de expertos e dropped tokens de habelos entre os bloques e logo as GPUs
            mean_assignments = torch.stack(layer_counts).float().mean(dim=0)
            if len(layer_dropped_counts) > 0:
                mean_dropped = torch.stack(layer_dropped_counts).float().mean().unsqueeze(0)
            
            if ddp:
                dist.all_reduce(mean_assignments, op=dist.ReduceOp.AVG)
                if len(layer_dropped_counts) > 0:
                    dist.all_reduce(mean_dropped, op=dist.ReduceOp.AVG)
            
            # Definimos as porcentaxes para mostralas nas métricas     
            assigment_percentage = mean_assignments / (total_batch_size / ddp_world_size)
            if len(layer_dropped_counts) > 0:
                dropped_percentage = mean_dropped / (total_batch_size / ddp_world_size)
            else:
                dropped_percentage = 0
            # ------------------------------------------
            
            
            optimizer.step()
            if device_type == "cuda":
                torch.cuda.synchronize() # wait for the GPU to finish work
            
            t1 = time.time()
            dt = t1 - t0 # time difference in seconds
            tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
            tokens_per_sec = tokens_processed / dt
            if master_process:
                
                parts = [
                    "{k} {v:.3e}".format(k=k, v=v)
                    for k, v in aux_losses.items()
                    if abs(v) > 0
                ]            
                loss_str = " | ".join(parts)
                msg = (f"step {step:5d} | loss: {loss_accum.item():.6f} "
                        f"| lr {lr:.4e} | norm: {norm:.4f} "
                        f"| dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
                
                if loss_str:
                    msg += " | " + loss_str
                    
                print(msg)
                
                with open(log_file, "a") as f:
                    f.write(f"{step} train {loss_accum.item():.6f}")
                    for k, v in aux_losses.items():
                        if abs(v) > 0:
                            f.write(f" {k} {v:.3e}")
                    f.write(f" expert_assignment_percentage {assigment_percentage.tolist()}")
                    if len(layer_dropped_counts) > 0:
                        f.write(f" dropped_percentage {dropped_percentage.item()*100:.2f}%")
                    f.write("\n")

        mes = monitor.end_window("epoch")
        
        time_h   = mes.time / 3600
        energy_j = torch.tensor(mes.total_energy, device=device)
        
        if ddp:
            dist.all_reduce(energy_j, op=dist.ReduceOp.SUM)
        
        energy_kwh = energy_j / 3.6e6

        if master_process:
            print(f"Epoch time: {time_h:.3f} h  |  energy: {energy_kwh:.6f} kWh")
            with open(log_file, "a") as f:
                f.write(f"Epoch time: {time_h:.3f} h  |  energy: {energy_kwh:.6f} kWh\n")

        if ddp:
            destroy_process_group()


if __name__ == "__main__":
    
    # simple launch:
    # python train_gpt2.py
    # DDP launch for e.g. 8 GPUs:
    # torchrun --standalone --nproc_per_node=8 train_gpt2.py

    # run the training loop
    from torch.distributed import init_process_group, destroy_process_group
    from torch.nn.parallel import DistributedDataParallel as DDP
    import torch.distributed as dist

    # set up DDP (distributed data parallel).
    # torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
    ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
    if ddp:
        # use of DDP atm demands CUDA, we set the device appropriately according to rank
        assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    else:
        # vanilla, non-DDP run
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        # attempt to autodetect device
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        print(f"using device: {device}")

    # added after video, pytorch can be serious about it's device vs. device_type distinction
    device_type = "cuda" if device.startswith("cuda") else "cpu"

    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)

    enc = tiktoken.get_encoding("gpt2")

    total_batch_size = 524288 # 2**19, ~0.5M, in number of tokens, esto viene 64 * 1024 * 8 GPUS
    B = 2 # micro batch size
    T = 256 # sequence length
    assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
    # Aplicamos gradient accumulation por que no podemos hacer un batch de 524288 tokens
    grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
    if master_process:
        print(f"total desired batch size: {total_batch_size}")
        print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")


    # Paralelización de datos
    train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")
    val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")

    torch.set_float32_matmul_precision('high') # BF16?
    
    args = parse_args()
    
    config = GPTConfig(
        vocab_size=50304,
        w_load=args.w_load,
        w_importance=args.w_importance,
        w_penalty=args.w_penalty,
        lambda_z=args.lambda_z,
        bias=args.bias,
        moe_implementation=args.moe_implementation,
    )
    
    exp_name = args.exp_name or f"{args.moe_implementation}_load{args.w_load}_imp{args.w_importance}"
    log_dir  = os.path.join("logMoA", exp_name)
    
    run_training(config, log_dir)
