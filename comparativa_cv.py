import matplotlib.pyplot as plt
import numpy as np

# Datos (exemplo) - substituír polos reais
implementacions = ["Liña base", "OLNNMoE", "SwitchMoE", "GShardMoE", "CondMLP"]
consumo_zeus = [12.14, 19.88, 16.01, 17.21, 13.52]  
consumo_calc = [21.60, 39.21, 31.55, 34.02, 25.20]  
co2 = [3.69, 6.76, 7.12, 7.09, 4.31]  

# --- Histograma Consumo (Zeus e Calculadora) ---
x = np.arange(len(implementacions))
width = 0.35

plt.figure(figsize=(7,5))
bars1 = plt.bar(x - width/2, consumo_zeus, width, label="Zeus (kWh)")
bars2 = plt.bar(x + width/2, consumo_calc, width, label="Calc. Online (kWh)")

plt.ylabel("Consumo enerxético (kWh)")
plt.xticks(x, implementacions)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.legend()

for bars in [bars1, bars2]:
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.05, f"{yval:.2f}", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
plt.show()

# --- Histograma CO2 ---
plt.figure(figsize=(7,5))
bars = plt.bar(implementacions, co2, color="seagreen")

plt.ylabel("Emisións de CO₂ (kgCO₂eq)")
plt.xticks(rotation=15)
plt.grid(axis="y", linestyle="--", alpha=0.7)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.05, f"{yval:.2f}", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
plt.show()











