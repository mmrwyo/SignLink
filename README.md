# SignLink: bilingual Bengali–English sign language recognition

Code and results for the paper:

> N. H. Samin, M. I. Ferdous, R. A. Suity, M. F. I. Rafsan, M. N. Rahman, M. M. Rahman, A. Shafkat, A. Torabi, and Y. Majeed, "Integrating Deep CNN Models for Multilingual Sign Language Recognition: A SignLink-Based Approach for Bengali and English," 11th IEEE Asia-Pacific Conference on Computer Science and Data Engineering (IEEE CSDE 2026), Cox's Bazar, Bangladesh, 2026.

## Contents

| Path | What it is |
|---|---|
| `signlink_rerun_colab.ipynb` | Full pipeline as a Colab notebook (download data, de-duplicate, split, train 8 CNNs, fusion experiments, figures) |
| `signlink_arcc.py`, `signlink_arcc.slurm` | Same pipeline as a resumable script and SLURM job (used on an NVIDIA H100) |
| `signlink_class_names_cell.py` | Maps the numbered Bengali classes to letter names using the merged BN–EN dataset |
| `results/tables/` | Single-model metrics, fusion results, fixed/broken case counts, per-class gains, confusions, run info |
| `results/figures/`, `results/histories/` | Figures and per-epoch training histories |
| `requirements.txt` | Package versions |

## Reproduce

1. Get a Kaggle API token (`kaggle.json`).
2. Colab: open the notebook, choose a GPU runtime, Runtime > Run all. Cluster: `sbatch signlink_arcc.slurm` from a folder containing `signlink_arcc.py`.
3. Finished models are skipped on re-run, so an interrupted run can be resumed.

Protocol in brief: images resized to 50×50 and exact duplicates removed; original images split 64/16/20 by class (seed 42) before augmentation; only Bengali training images augmented; eight CNNs fine-tuned at 224×224; fusion layers fitted on frozen embeddings using half of the validation split; McNemar tests with Holm correction.

Trained weights are not included because of file size; they are available from the corresponding author on request.

## Data

- Bengali Sign Language Dataset: https://www.kaggle.com/datasets/muntakimrafi/bengali-sign-language-dataset
- Sign Language Gesture Images Dataset: https://www.kaggle.com/datasets/ahmedkhanak1995/sign-language-gesture-images-dataset
