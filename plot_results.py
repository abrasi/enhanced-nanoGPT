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
# importance_losses = []
load_losses = []
steps_train = []
steps_val = []
# dropped_tokens = []
expert_assignments = []

# train_pattern = re.compile(r"(\d+) train ([\d.]+).+dropped tokens: [\d.]+ \((\d+.\d+)%\)")
# expert_pattern = re.compile(r"expert assigment percentage: \[([\d., ]+)\]")
pattern = re.compile(r"(\d+) train ce ([\d.]+)")


with open(log_file, "r") as f:
    for line in f:
        # train_match = train_pattern.search(line)
        # expert_match = expert_pattern.search(line)
        match = pattern.search(line)
        if match:
            step = int(match.group(1))
            train_loss = float(match.group(2))
            # importance_loss = float(match.group(3))
            load_loss = float(match.group(4))
            expert_values = list(map(float, match.group(5).split(',')))
            
            steps_train.append(step)
            train_losses.append(train_loss)
            # importance_losses.append(importance_loss)
            load_losses.append(load_loss)
            expert_assignments.append(expert_values)
        
        parts = line.strip().split(',', maxsplit=1)
        meta_info = parts[0].split()
        epoch, loss_type, loss_value = meta_info[:3]
        epoch = int(epoch)
        loss_value = float(loss_value)
        
        if loss_type == "val":
            val_losses.append(loss_value)
            steps_val.append(epoch)

# Convertir listas a arrays para graficar
train_losses = np.array(train_losses)
val_losses = np.array(val_losses)
# importance_losses = np.array(importance_losses)
load_losses = np.array(load_losses)
# dropped_tokens = np.array(dropped_tokens)
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
# plt.figure(figsize=(10, 5))
# plt.plot(steps_train, importance_losses, label="Importance Loss", color='orange')
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.title("Importance Loss Evolution")
# plt.legend()
# plt.grid()
# plt.show()

# Load loss
# plt.figure(figsize=(10, 5))
# plt.plot(steps_train, load_losses, label="Load Loss", color='green')
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.title("Load Loss Evolution")
# plt.legend()
# plt.grid()
# plt.show()

# Gráfico de dropped tokens
# plt.figure(figsize=(10, 5))
# plt.plot(steps_train, dropped_tokens, label="Dropped Tokens (%)", color='purple')
# plt.title("Dropped Tokens Percentage")
# plt.xlabel("Epochs")
# plt.ylabel("Percentage (%)")
# plt.legend()
# plt.grid()
# plt.show()

# Gráfico de asignación a expertos con stacked area plot
# plt.figure(figsize=(10, 5))
# plt.stackplot(steps_train, expert_assignments.T, labels=[f"Expert {i}" for i in range(expert_assignments.shape[1])], alpha=0.7)
# plt.title("Expert Assignment Proportions")
# plt.xlabel("Epochs")
# plt.ylabel("Proportion")
# plt.legend()
# plt.grid()
# plt.show()

print(f"Minimum Train Loss: {min_train_loss:.4f} at epoch {min_train_epoch}")
print(f"Minimum Validation Loss: {min_val_loss:.4f} at epoch {min_val_epoch}")
