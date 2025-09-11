import argparse
import csv
from pathlib import Path
import matplotlib.pyplot as plt

def ler_csv(csv_path):
    models, scores = [], []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if "model" not in reader.fieldnames or "score" not in reader.fieldnames:
            raise ValueError("O CSV debe ter columnas 'model' e 'score'.")
        for row in reader:
            m = row["model"].strip()
            try:
                s = float(row["score"])
            except ValueError:
                continue
            models.append(m)
            scores.append(s)
    return models, scores

def ordenar_desc(models, scores):
    pares = sorted(zip(models, scores), key=lambda x: x[1], reverse=True)
    return [m for m, _ in pares], [s for _, s in pares]

def debuxar_barras(models, scores, titulo, outfile, ancho=12, alto=6, dpi=200, mostrar_valores=True):
    # Unha soa figura, sen estilos nin cores específicos (requisito)
    plt.figure(figsize=(ancho, alto))
    x = range(len(models))
    plt.bar(x, scores)

    plt.xticks(list(x), models, rotation=30, ha='right')
    plt.title(titulo)
    plt.xlabel("Modelo")
    plt.ylabel("Mellor puntuación en HellaSwag")

    if mostrar_valores:
        for i, v in enumerate(scores):
            # etiqueta enriba de cada barra
            txt = f"{v:.4f}" if v < 1 else f"{v:.2f}"
            plt.text(i, v, txt, ha='center', va='bottom')

    plt.tight_layout()
    if outfile:
        Path(outfile).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(outfile, dpi=dpi, bbox_inches="tight")
        print(f"Gráfico gardado en: {Path(outfile).resolve()}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Barras HellaSwag por modelo")
    parser.add_argument("--csv", type=str, default=None, help="Ruta a CSV con columnas model,score")
    parser.add_argument("--out", type=str, default="hellaswag_barras.png", help="Arquivo de saída (png/svg/pdf)")
    parser.add_argument("--titulo", type=str,
                        default="Mellores puntuacións en HellaSwag por modelo",
                        help="Título do gráfico")
    args = parser.parse_args()

    if args.csv:
        models, scores = ler_csv(args.csv)
        if not models:
            raise SystemExit("Non se leron datos válidos do CSV.")
    else:
        # 🔧 MODELO RÁPIDO: edita aquí as túas mellores puntuacións
        models = ["nanoGPT 355M", "OLNNMoE + Nesgo balanceo"]
        scores = [0.3636, 0.3974]

    models, scores = ordenar_desc(models, scores)
    debuxar_barras(models, scores, args.titulo, args.out)

if __name__ == "__main__":
    main()