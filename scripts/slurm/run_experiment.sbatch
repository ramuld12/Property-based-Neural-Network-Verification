#!/usr/bin/env bash
#SBATCH --job-name=thesis-exp
#SBATCH --output=outputs/slurm/%x_%j.out
#SBATCH --error=outputs/slurm/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

set -euo pipefail

CONFIG_PATH="${1:?Usage: sbatch run_experiment.sbatch <config.yaml> <baseline|properties>}"
METHOD="${2:?Usage: sbatch run_experiment.sbatch <config.yaml> <baseline|properties>}"

mkdir -p outputs/slurm
module purge
module load python/3.11

source .venv/bin/activate
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

python -m thesis.cli run "$METHOD" --config "$CONFIG_PATH"
