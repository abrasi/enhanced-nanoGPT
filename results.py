import matplotlib.pyplot as plt
import numpy as np

# Leer el archivo de log
log_file = "./log.txt"
train_losses = []
val_losses = []
steps_train = []
steps_val = []

with open(log_file, "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 3:
            epoch, loss_type, loss_value = parts
            epoch = int(epoch)
            loss_value = float(loss_value)
            
            if loss_type == "train":
                train_losses.append(loss_value)
                steps_train.append(epoch)
            elif loss_type == "val":
                val_losses.append(loss_value)
                steps_val.append(epoch)

# Encontrar los mínimos
min_train_loss = min(train_losses)
min_train_epoch = steps_train[np.argmin(train_losses)]

min_val_loss = min(val_losses)
min_val_epoch = steps_val[np.argmin(val_losses)]

# Graficar
plt.figure(figsize=(10, 5))
plt.plot(steps_train, train_losses, label="Train Loss", color='blue')
plt.plot(steps_val, val_losses, label="Validation Loss", color='red')
plt.scatter(min_train_epoch, min_train_loss, color='blue', marker='o', label=f'Min Train Loss: {min_train_loss:.4f}')
plt.scatter(min_val_epoch, min_val_loss, color='red', marker='o', label=f'Min Val Loss: {min_val_loss:.4f}')
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Evolution")
plt.legend()
plt.grid()
plt.show()

print(f"Minimum Train Loss: {min_train_loss:.4f} at epoch {min_train_epoch}")
print(f"Minimum Validation Loss: {min_val_loss:.4f} at epoch {min_val_epoch}")