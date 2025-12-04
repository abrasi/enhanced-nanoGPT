# enhanced-nanoGPT

Fork of [karpathy/build-nanogpt] focused on **Mixture-of-Experts (MoE)** variants and **Green AI** experiments (energy / CO₂ vs. quality).

> Repo maintained by Artur Gil – experiments for a BSc thesis on MoE-based LLMs and energy efficiency.

---

## Features

- Fully compatible with the original **nanoGPT** training loop.
- Several expert-style blocks:
  - Standard dense FFN (baseline)
  - `OLNNMoE`
  - `GShardMoE`
  - `SwitchMoE`
  - `CondMLP` (lightweight conditional MLP-style block)
- Training on **FineWeb EDU**–style datasets (small educational corpus).
- Evaluation on **HellaSwag** via checkpoints.
- `experiment_loop.py` for running batches of experiments (different MoE configs, seeds, etc.).
- Plot scripts (`plot_results.py`, `comparativa_cv.py`, `hellaplot.py`) to generate figures.
- Optional **energy monitoring** with `ZeusMonitor` to log:
  - Training time
  - Energy (Wh / kWh)
---

## Repository overview

Current key files in this fork (top level):

- `train_MoAgpt2.py`, `train_MoEgpt2.py` – training scripts for different MoE / MoA variants.
- `fineweb.py` – dataset download / preprocessing helpers (FineWeb EDU).
- `hellaswag.py`, `hellaswag_eval_from_checkpoint.py` – HellaSwag evaluation utilities.
- `experiment_loop.py` – automation of multiple runs / sweeps.
- `plot_results.py`, `comparativa_cv.py`, `hellaplot.py` – analysis and plotting utilities.
- (Plus all the original `build-nanogpt` code and configs.)

---

## Installation

```bash
git clone https://github.com/<your-user>/enhanced-nanoGPT.git
cd enhanced-nanoGPT

# (Recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

pip install -r requirements.txt
