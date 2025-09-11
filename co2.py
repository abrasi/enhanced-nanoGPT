import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------
# Datos (orde de arriba a abaixo)
# -----------------------
values = [502, 352, 291.42, 153.9, 70, 63, 62.44, 31.22,
          25, 22.23, 18.08, 16.68, 11.95, 5.51, 3.17, 0.99]

labels = ["GPT-3 (175B)",
          "Gopher (280B)",
          "Llama 2 (70B)",
          "Llama 2 (34B)",
          "OPT (175B)",
          "Coche medio, vida útil completa",
          "Llama (13B)", "Llama 2 (7B)", "BLOOM (176B)", "Granite (13B)",
          "Vida media estadounidense (1 ano)",
          "Starcoder (15.5B)", "Luminous Extended (30B)",
          "Vida media global (1 ano)", "Luminous Base (13B)",
          "Viaxe en avión, 1 pasaxeiro, NY–SF"]

# -----------------------
# Axustes rápidos
# -----------------------
title = ("Emisións equivalentes de CO₂ (toneladas) en modelos de aprendizaxe automática\n"
         "seleccionados e exemplos da vida real, 2020–23")
xlabel = "Toneladas de CO₂ eq."
figsize = (10, 6)
dpi = 200
outfile = Path("grafico_emisions_galego.png")
top_to_bottom = True

# Texto do pé de imaxe (fonte)
footnote = ("Fonte: AI Index, 2024; Luccioni et al., 2022; Strubell et al., 2019  |  "
            "Gráfico: AI Index Report 2024")

# -----------------------
# Construír gráfico
# -----------------------
fig = plt.figure(figsize=figsize)
ax = fig.add_subplot(111)

# Posicións en eixe Y
if top_to_bottom:
    y_pos = list(range(len(values)))[::-1]   # deixa a 1ª arriba
    labels_to_show = labels
else:
    y_pos = list(range(len(values)))
    labels_to_show = labels

ax.barh(y_pos, values)  # non poñas cores explícitas

ax.set_yticks(y_pos)
ax.set_yticklabels(labels_to_show)
ax.set_title(title)
ax.set_xlabel(xlabel)

# Etiquetas numéricas sobre cada barra
for y, v in zip(y_pos, values):
    txt = f"{int(v)}" if abs(v - int(v)) < 1e-9 else f"{v:.2f}"
    ax.text(v, y, " " + txt, va="center")

# Marxe inferior extra para o pé
plt.subplots_adjust(bottom=0.18)  # aumenta se o pé corta
# Pé de imaxe, aliñado á esquerda no bordo inferior
fig.text(0.01, 0.01, footnote, ha='left', va='bottom', fontsize=9)

plt.tight_layout(rect=[0, 0.05, 1, 1])  # deixa sitio ao pé
# Garda a imaxe (ou usa show)
# fig.savefig(outfile, dpi=dpi, bbox_inches="tight")
plt.show()

print(f"Gráfico gardado en: {outfile.resolve()}")