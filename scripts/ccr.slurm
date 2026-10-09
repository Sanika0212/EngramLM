#!/bin/bash
# Run an EngramLM language-modelling experiment on a Slurm GPU cluster (for example UB CCR).
# Usage:  sbatch --account=<your_account> scripts/ccr.slurm <config.yaml> <output_dir>
#SBATCH --job-name=engramlm
#SBATCH --partition=general-compute
#SBATCH --qos=general-compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail
CONFIG=${1:?usage: sbatch scripts/ccr.slurm <config.yaml> <output_dir>}
OUT=${2:?usage: sbatch scripts/ccr.slurm <config.yaml> <output_dir>}

# Keep model and dataset caches on project storage instead of the home quota.
export HF_HOME=${HF_HOME:-$PWD/.hf-cache}

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

uv sync --frozen --extra train
uv run engramlm audit --tiny
uv run engramlm lm --config "$CONFIG" --out "$OUT" --device cuda
