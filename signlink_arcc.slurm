#!/bin/bash
#SBATCH --job-name=signlink
#SBATCH --account=cvanddl-cea
#SBATCH --partition=mb-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=signlink_%j.out

# Run from the folder that contains signlink_arcc.py and signlink_rerun/
export PYTHONNOUSERSITE=1                      # ignore packages in ~/.local
PY="$HOME/.conda/envs/signlink/bin/python"     # the signlink env's own Python 3.11
cd "$SLURM_SUBMIT_DIR"
nvidia-smi
"$PY" --version
"$PY" signlink_arcc.py --out "$SLURM_SUBMIT_DIR/signlink_rerun"
