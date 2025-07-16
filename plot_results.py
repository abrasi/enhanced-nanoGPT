import matplotlib.pyplot as plt
import numpy as np
import re

# Leer el archivo de log
import matplotlib.pyplot as plt
import numpy as np
import ast

# Leer el archivo de log
log_file = "./log.txt"
train_losses = []
val_losses = []
z_losses = []
penalty_losses = []
importance_losses = []
load_losses = []
steps_train = []
steps_val = []
steps_z = []
dropped_tokens = []
expert_assignments = []

# pattern = re.compile(
#     r"^(\d+)\s+train\s+([\d.]+)\s+expert_assignment_percentage\s+(\[.*?\])\s+dropped_percentage\s+([\d.]+)%"
# )
pattern = re.compile(
    r"^(\d+)\s+train\s+([\d.]+)(?:\s+importance_loss\s+([\d.eE+-]+))?\s+expert_assignment_percentage\s+(\[.*?\])\s+dropped_percentage\s+([\d.]+)%"  # "
)


with open(log_file, "r") as f:
    for line in f:
        # Buscar línea de entrenamiento con load_loss
        match = pattern.search(line)
        if match:
            step = int(match.group(1))
            train_loss = float(match.group(2))
            # load_loss = float(match.group(3))
            # z_loss = float(match.group(3))
            importance_loss = float(match.group(3))
            # penalty = float(match.group(3))
            percentages = ast.literal_eval(match.group(4))
            dropped = float(match.group(5))

            steps_train.append(step)
            train_losses.append(train_loss)
            expert_assignments.append(percentages)
            importance_losses.append(importance_loss)
            dropped_tokens.append(dropped)
            # load_losses.append(load_loss)
            # z_losses.append(z_loss)
            # penalty_losses.append(penalty)
            
        # Línea de validación
        elif "val" in line:
            parts = line.strip().split()
            if len(parts) >= 3:
                step = int(parts[0])
                val_loss = float(parts[2])
                steps_val.append(step)
                val_losses.append(val_loss)
        

# Convertir listas a arrays para graficar
train_losses = np.array(train_losses)
val_losses = np.array(val_losses)
importance_losses = np.array(importance_losses)
# load_losses = np.array(load_losses)
# z_losses = np.array(z_losses)
# penalty_losses = np.array(penalty_losses)
dropped_tokens = np.array(dropped_tokens)
expert_assignments = np.array(expert_assignments)

# Encontrar los mínimos
min_train_loss = np.min(train_losses)
min_train_epoch = steps_train[np.argmin(train_losses)]
min_val_loss = np.min(val_losses)
min_val_epoch = steps_val[np.argmin(val_losses)]

# Graficar pérdidas
plt.figure(figsize=(10, 5))
plt.plot(steps_train, train_losses, label="Train Loss", color='blue')
plt.plot(steps_val, val_losses, label="Validation Loss", color='red')
plt.scatter(min_train_epoch, min_train_loss, color='blue', marker='o', label=f'Min Train Loss: {min_train_loss:.4f}')
plt.scatter(min_val_epoch, min_val_loss, color='red', marker='o', label=f'Min Val Loss: {min_val_loss:.4f}')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Evolution")
plt.legend()
plt.grid()
plt.show()


# Importance loss
plt.figure(figsize=(10, 5))
plt.plot(steps_train, importance_losses, label="Importance Loss", color='orange')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Importance Loss Evolution")
plt.legend()
plt.grid()
plt.show()

# Load loss
# plt.figure(figsize=(10, 5))
# plt.plot(steps_train, load_losses, label="Load Loss", color='green')
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.title("Load Loss Evolution")
# plt.legend()
# plt.grid()
# plt.show()

# z-loss
# plt.figure(figsize=(10, 5))
# plt.plot(steps_train, z_losses, label="z-Loss", color='green')
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.title("z-Loss Evolution")
# plt.legend()
# plt.grid()
# plt.show()

# penalty loss
# plt.figure(figsize=(10, 5))
# plt.plot(steps_train, penalty_losses, label="Penalty Loss", color='green')
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.title("Penalty Loss Evolution")
# plt.legend()
# plt.grid()
# plt.show()

# Gráfico de dropped tokens
plt.figure(figsize=(10, 5))
plt.plot(steps_train, dropped_tokens, label="Dropped Tokens (%)", color='purple')
plt.title("Dropped Tokens Percentage")
plt.xlabel("Epochs")
plt.ylabel("Percentage (%)")
plt.legend()
plt.grid()
plt.show()

# Gráfico de asignación a expertos con stacked area plot
plt.figure(figsize=(10, 5))
plt.stackplot(steps_train, expert_assignments.T, labels=[f"Expert {i}" for i in range(expert_assignments.shape[1])], alpha=0.7)
plt.title("Expert Assignment Proportions")
plt.xlabel("Epochs")
plt.ylabel("Proportion")
plt.legend()
plt.grid()
plt.show()

print(f"Minimum Train Loss: {min_train_loss:.4f} at epoch {min_train_epoch}")
print(f"Minimum Validation Loss: {min_val_loss:.4f} at epoch {min_val_epoch}")
