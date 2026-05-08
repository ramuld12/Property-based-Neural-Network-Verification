#!/usr/bin/env bash
#SBATCH --job-name=thesis-exp
#SBATCH --output=outputs/slurm/%x_%j.out
#SBATCH --error=outputs/slurm/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

set -eo pipefail

CONFIG_PATH="${1:?Usage: sbatch run_experiment.sbatch <config.yaml> <baseline|properties>}"
METHOD="${2:?Usage: sbatch run_experiment.sbatch <config.yaml> <baseline|properties>}"

echo "Running on $(hostname)"
echo "Working directory: $(pwd)"

mkdir -p outputs/slurm

module purge
module load Anaconda3

set +u
eval "$(conda shell.bash hook)"
conda activate /opt/itu/condaenvs/pytorch
set -u

echo "Python version: $(python --version)"
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

python -m thesis.cli run "$METHOD" --config "$CONFIG_PATH"
