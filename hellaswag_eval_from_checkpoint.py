import torch
import torch.distributed as dist
from torch.serialization import safe_globals
from train_gpt2 import GPT, GPTConfig, iterate_examples, render_example, get_most_likely_row

# Cargar modelo
checkpoint_path = "./log/model_19072.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ddp = torch.distributed.is_initialized()
ddp_rank = dist.get_rank() if ddp else 0
ddp_world_size = dist.get_world_size() if ddp else 1
master_process = (ddp_rank == 0)

checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

# Ajustar claves para quitar el prefijo "_orig_mod."
state_dict = {k.replace("_orig_mod.", ""): v for k, v in checkpoint["model"].items()}

# Crear modelo con la misma configuración que en el checkpoint
config = checkpoint["config"]  
model = GPT(config)

# Cargar los pesos corregidos
model.load_state_dict(state_dict, strict=True)

# Enviar modelo a GPU y cambiar a modo evaluación
model.to(device)
model.eval()

print(f"Modelo cargado desde el checkpoint en {checkpoint_path}")

# Evaluar en HellaSwag
num_correct_norm = 0
num_total = 0
for i, example in enumerate(iterate_examples("val")):
    if i % ddp_world_size != ddp_rank:
        continue
    
    _, tokens, mask, label = render_example(example)
    tokens = tokens.to(device)
    mask = mask.to(device)
    
    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits, _ = model(tokens)
        pred_norm = get_most_likely_row(tokens, mask, logits)
    
    num_total += 1
    num_correct_norm += int(pred_norm == label)

# Reducir stats en DDP
if ddp:
    num_total = torch.tensor(num_total, dtype=torch.long, device=device)
    num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
    dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
    dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
    num_total = num_total.item()
    num_correct_norm = num_correct_norm.item()

# Calcular accuracy
acc_norm = num_correct_norm / num_total
if master_process:
    print(f"HellaSwag accuracy: {num_correct_norm}/{num_total} = {acc_norm:.4f}")
