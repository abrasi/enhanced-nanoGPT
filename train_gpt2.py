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
# -----------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):

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

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y
    
    
# Experto
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

    
# num decayed parameter tensors: 146, with 294,297,600 parameters
# num non-decayed parameter tensors: 170, with 259,584 parameters
class MoE(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.n_experts = config.n_experts
        assert self.n_experts > 0
        self.experts = nn.ModuleList([FFN(config) for _ in range(self.n_experts)])
        self.gate = nn.Linear(config.n_embd, self.n_experts, bias=False)
        self.noise_layer = nn.Linear(config.n_embd, self.n_experts, bias=False)
        
        self.n_experts_per_tok = config.n_experts_per_tok
        # Peso para la importancia de los expertos
        self.w_importance = config.w_importance
        # Peso para la carga de los expertos, hacer que reciban nº similar de tokens
        self.w_load = config.w_load
        # Peso para la penalización de logits grandes
        self.lambda_z = config.lambda_z
        
        self.importance_loss = 0.0
        self.load_loss = 0.0
        self.z_loss = 0.0
        
        # self.bias = nn.Parameter(torch.empty(self.n_experts))
        
    
    # load_loss asegura que todos los expertos reciben ejemplos.
    # auxiliary_loss asegura que todos los expertos son útiles y no quedan expertos inactivos.
    def get_kth_excluding(self, gate_logits):
        """
        Calcula kth_excluding(H(x), k, i) para cada experto i en el gating network.

        gate_logits: [batch*seq_len, n_experts] (logits antes de softmax)
        k: número de expertos a seleccionar por token
        """
        
        # Expandir logits para comparación (agregar una dimensión extra para simular exclusión)
        expanded_logits = gate_logits.unsqueeze(1).expand(-1, self.n_experts, -1)  # [B*T, n_experts, n_experts]

        # Crear una máscara para excluir cada experto i
        mask = torch.eye(self.n_experts, device=gate_logits.device).bool()  # [n_experts, n_experts]
        masked_logits = expanded_logits.masked_fill(mask, float('-inf'))  # Reemplazar la diagonal con -inf

        # Obtener el k-ésimo mayor excluyendo cada experto i
        kth_values, _ = torch.topk(masked_logits, self.n_experts_per_tok, dim=2)  # [B*T, n_experts, k]
        kth_excluding_values = kth_values[:, :, -1]  # Última columna es el k-ésimo mayor excluyendo i

        return kth_excluding_values
        
    
    # Ensure that each expert receives a similar number of tokens 
    def get_load_loss(self, gate_logits, noise_std):
                        
        kth_excluding = self.get_kth_excluding(gate_logits)
        
        normal_cdf = torch.distributions.Normal(0, 1).cdf
        P_xi = normal_cdf((gate_logits - kth_excluding) / noise_std)
        
        Load_X = P_xi.sum(dim=0)

        mean_load = Load_X.mean()
        std_load = Load_X.std()
        CV = std_load / mean_load
        
        # Compute load balancing loss
        L_load = self.w_load * (CV ** 2)
        return L_load       
        
    # Importance-based loss
    def get_auxiliary_loss(self, probs, selected_experts):
    
        # Tenemos un tensor probs de forma (B*T, n_experts_per_tok) y necesitamos uno de forma (B*T, n_experts)
        B_T, _ = probs.shape
        probs_expanded = torch.zeros(B_T, self.n_experts, device=probs.device, dtype=probs.dtype)
        
        # Esta operacion introduce a la dimensión 1, en los índices selected_experts, los pesos de los expertos probs
        probs_expanded.scatter_(1, selected_experts, probs)
        
        importance = probs_expanded.sum(dim=0)
        
        cv_squared = torch.var(importance) / torch.mean(importance) ** 2
        
        return self.w_importance * cv_squared
    
    # St-MoE, penaliza logits muy grandes
    
    def get_z_loss(self, gate_logits):
        
        logsumexp = torch.logsumexp(gate_logits, dim=-1)
        loss = torch.mean(logsumexp ** 2)
        return self.lambda_z * loss

        
    def forward(self, x):
        # Se convierte la dimension a (B*T, C) -> (B*T, n_embd)        
        x_squashed = x.view(-1, x.shape[-1])
        
        # (B*T, n_experts) -> por cada token se obtiene un vector de n_experts dimensiones, cada experto tiene una puntuación
        gate_logits = self.gate(x_squashed)
        
        # z_loss = self.get_z_loss(gate_logits)
        
        noise_std = F.softplus(self.noise_layer(x_squashed))
        gate_logits += torch.randn_like(gate_logits) * noise_std
        
        # BIAS de DeepSeek
        # gate_logits += self.bias
        
        self.load_loss = self.get_load_loss(gate_logits, noise_std)
                
        # Se seleccionan las top n_experts_per_tok puntuaciones de expertos por token
        weights, selected_experts = torch.topk(gate_logits, self.n_experts_per_tok)
        # weights nos dice las top n_experts_per_tok puntuaciones de expertos por token
        # selected_experts nos dice que top n_experts_per_tok expertos son los seleccionados para cada token
                    # selected_experts =
                    # [[0, 2],  # Token 0 → Expertos 0 y 2
                    # [1, 3],  # Token 1 → Expertos 1 y 3
                    # [0, 3],  # Token 2 → Expertos 0 y 3
                    # [1, 2]]  # Token 3 → Expertos 1 y 2
    
        # Aplicamos softmax en la dimension de las filas, por cada token se obtiene un multiplicador para cada experto,
        # ¿cuanto pondera cada experto en la representación de un token?
        weights = F.softmax(
            weights,
            dim=1,
            dtype=torch.float,
        ).type_as(x)    
        
        # self.importance_loss = self.get_auxiliary_loss(weights, selected_experts)
        
        self.total_assigment_count = torch.zeros(self.n_experts, device=x.device, dtype=torch.int32)

        results = torch.zeros_like(x_squashed)
        for i, expert in enumerate(self.experts):
            batch_idx, nth_expert = torch.where(selected_experts == i)

            self.total_assigment_count[i] += batch_idx.shape[0]
            # batch_idx nos dice a que experto va cada fila, cada token
            # nth_expert nos dice qué indice en cada fila, cada token, corresponde al experto i
            
                    # batch_idx = [0, 3]    # Tokens 0 y 3 van al experto 2
                    # nth_expert = [1, 1]   # En cada fila, el experto 2 es el segundo seleccionado (índice 1)
            
            if len(batch_idx) > 0:
                # Pasamos los tokens seleccionados al experto i
                expert_logits = expert(x_squashed[batch_idx])
                # Multiplicamos la salida del experto por el peso que le corresponde, se añade una dimensión para que las dimensiones sean compatibles
                # weights[batch_idx, nth_expert, None] contiene las contribuciones del experto i para cada token
                results[batch_idx] += weights[batch_idx, nth_expert, None] * expert_logits
                        
        return results.view_as(x), self.load_loss
        

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        if config.expert_implementation == "MoE":
            self.moe_layer = MoE(config)
        elif config.expert_implementation == "SwitchMoE":
            self.moe_layer = SwitchMoE(config)
        # self.mlp = MLP(config)

    def forward(self, x):
        # Aqui en la atención los tokens se "comunican" entre ellos
        x = x + self.attn(self.ln_1(x)) # sumamos la salida de una capa a la entrada para evitar el problema del GRADIENT VANISHING -> conexión residual
        # Aqui los tokens "piensan" individualmente
        x2, auxiliary_loss = self.moe_layer(self.ln_2(x))
        return x + x2, auxiliary_loss

@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension
    n_experts: int = 4
    n_experts_per_tok: int = 2
    w_importance: float = 0.1
    w_load: float = 0.1
    expert_implementation: str = "MoE"
    lambda_z: float = 0.001
    # expert_capacity: int = 1536 # = expert_capacity =  (B*T / n_experts) * capacity_factor = 1.5, usamos T en vez de B*T por la forma en que se hace forward en SwitchMoE

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
        
        
        # ----- Auxilary loss for MoE -----
        total_auxilary_loss = 0
        num_blocks = len(self.transformer.h)
        
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x, auxilary_loss = block(x)
            total_auxilary_loss += auxilary_loss
        
        mean_auxilary_loss = total_auxilary_loss / num_blocks
        # ----- Auxilary loss for MoE -----
        
        
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
    B = 16 # micro batch size
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

    torch.set_float32_matmul_precision('high')

    # Funcion que actuaiza el bias en cada capa MoE del modelo
    def update_moe_layer_bias(moe_layer, gamma):
        # Sumar los tokens asignados a cada experto en esta capa
        tokens_per_expert = moe_layer.total_assigment_count.float().to(device)
        
        # Recogemos los tokens de todos los expertos
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


    def get_auxiliary_losses():
        # importance_loss = []
        load_loss = []
        for block in model.module.transformer.h:
            
            if hasattr(block, 'moe_layer') and hasattr(block.moe_layer, 'load_loss'):
                # importance_loss.append(block.moe_layer.importance_loss.item())
                load_loss.append(block.moe_layer.load_loss.item())
            
            return None, torch.tensor(load_loss).mean().item()            

    # create model
    model = GPT(GPTConfig(vocab_size=50304))
    # model = GPT.from_pretrained("gpt2") # or init from OpenAI GPT-2
    model.to(device)
    use_compile = False # torch.compile interferes with HellaSwag eval and Generation. TODO fix
    if use_compile:
        model = torch.compile(model)
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=False)
    raw_model = model.module if ddp else model # always contains the "raw" unwrapped model

    max_lr = 6e-4
    min_lr = max_lr * 0.1
    warmup_steps = 715
    max_steps = 19073 # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens

    # gamma = 0.001 # Hiperparametro para el ajuste del bias del Gate de MoE
    # bias_update_step = int(max_steps * 0.97) # Más o menos en DeepSeek el bias de Gating de MoE pasa a ser 0 a partir de este punto

    def get_lr(it):
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

    # optimize!
    optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device_type)

    # create the log directory we will write checkpoints to and log to
    log_dir = "log"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"log.txt")
    with open(log_file, "w") as f: # open for writing to clear the file
        pass

    # Monitorización energia de las GPUs
    monitor = ZeusMonitor(gpu_indices=[torch.cuda.current_device()])
    steps = []

    monitor.begin_window("epoch")
    for step in range(max_steps):
        t0 = time.time()
        last_step = (step == max_steps - 1)

        monitor.begin_window("step")
        
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
        # if step == bias_update_step:
        #     gamma = 0.0
        #     if master_process:
        #         print("Zeroed out the bias update speed in the MoE layer -> 97% of training done")
                
        # if gamma > 0:
            
        #     for block in model.module.transformer.h: 
        #         if hasattr(block, 'moe_layer'): 
        #             update_moe_layer_bias(dist, device, block.moe_layer, gamma)
        #--------------------------------           
        
        # Expert assigment count
        # first_assigment_counts = []
        total_assigment_count = []
        # Promedio de conteo de asignaciones a expertos en todas los bloques transformer
        # fist_assigment_mean = None
        # Recogemos los conteos de asignaciones a expertos de todas las capas MoE 
        for block in model.module.transformer.h:
            
            if hasattr(block, 'moe_layer') and hasattr(block.moe_layer, 'total_assigment_count'):
                # first_assigment_counts.append(block.moe_layer.first_expert_assigment_count)
                total_assigment_count.append(block.moe_layer.total_assigment_count)
                
        if total_assigment_count:
            # Stack: obtiene un tensor de forma (n_layer, n_experts)
            # 2 tensores: uno para la asignación del primer experto y otro para la asignación a la primera y segunda opción
            # first_assigment_counts_stack = torch.stack(first_assigment_counts, dim=0)
            total_assigment_count_stack = torch.stack(total_assigment_count, dim=0)
            # Promedio a lo largo de los bloques transformer (dim=0)
            # fist_assigment_mean = first_assigment_counts_stack.float().mean(dim=0)
            total_mean = total_assigment_count_stack.float().mean(dim=0)
                    
        if total_assigment_count is not None:
            if ddp:
                # Se promedian los tokens recibidos por los expertos entre las GPUs
                # dist.all_reduce(fist_assigment_mean, op=dist.ReduceOp.AVG)
                dist.all_reduce(total_mean, op=dist.ReduceOp.AVG)
                total_tokens = B * T * 2 # Expertos activados = 2
                # total_assigment = total_mean.sum()
                # dropped_tokens = total_tokens - total_assigment
                assigment_percentage = total_mean / float(total_tokens)
                # first_assignment_percentage = fist_assigment_mean.float() / float(total_tokens)
                # dropped_percentage = float(dropped_tokens) / total_tokens
        # -----------------------
        _, load_loss = get_auxiliary_losses()
                
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
            loss.backward()
        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # determine and set the learning rate for this iteration
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()
        if device_type == "cuda":
            torch.cuda.synchronize() # wait for the GPU to finish work
            
        result = monitor.end_window("step")
        steps.append(result)
        
        t1 = time.time()
        dt = t1 - t0 # time difference in seconds
        tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
        tokens_per_sec = tokens_processed / dt
        if master_process:
            print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
            with open(log_file, "a") as f:
                # f.write(f"{step} train {loss_accum.item():.6f}, first expert assigment percentage: {first_assignment_percentage.tolist()}, dropped tokens: {dropped_tokens} ({dropped_percentage * 100:.2f}%)\n")
                f.write(f"{step} train {loss_accum.item():.6f}, load loss {load_loss:.6f} expert assigment percentage: {assigment_percentage.tolist()}\n")

    mes = monitor.end_window("epoch")

    avg_time = sum(map(lambda m: m.time, steps)) / len(steps)
    avg_energy = sum(map(lambda m: m.total_energy, steps)) / len(steps)
    with open(log_file, "a") as f:
        f.write(f"Epoch consumed {mes.time} s and {mes.total_energy} J.")
        f.write(f"One step took {avg_energy} J on average.")

    if ddp:
        destroy_process_group()