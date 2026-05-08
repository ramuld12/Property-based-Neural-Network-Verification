#!/usr/bin/env bash
#SBATCH --job-name=thesis-sweep
#SBATCH --output=outputs/slurm/%x_%A_%a.out
#SBATCH --error=outputs/slurm/%x_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --array=0-63

set -euo pipefail

SWEEP_FILE="${1:?Usage: sbatch run_array.sbatch <sweep.txt>}"
LINE_NUMBER=$((SLURM_ARRAY_TASK_ID + 1))
COMMAND=$(sed -n "${LINE_NUMBER}p" "$SWEEP_FILE")

mkdir -p outputs/slurm
module purge
module load python/3.11

source .venv/bin/activate
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

echo "Running: $COMMAND"
eval "$COMMAND"
