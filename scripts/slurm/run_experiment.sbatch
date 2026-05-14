#!/usr/bin/env bash
#SBATCH --job-name=ciciot_yager
#SBATCH --output=outputs/slurm/%x_%j.out
#SBATCH --error=outputs/slurm/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

export TMPDIR=/home/username/tmp
export TMP=/home/username/tmp
export TEMP=/home/username/tmp

CONFIG_PATH="${1:?Usage: sbatch run_experiment.sbatch <config.yaml> <baseline|properties>}"
METHOD="${2:?Usage: sbatch run_experiment.sbatch <config.yaml> <baseline|properties>}"

echo "Running on $(hostname)"
echo "Working directory: $(pwd)"
nvidia-smi

mkdir -p outputs/slurm

module load Anaconda3  # Load anaconda

# Init anaconda (in the HPC, prefer to use the following instead of 'conda init')
eval "$(conda shell.bash hook)"

ENV_NAME='thesis'

conda activate $ENV_NAME
echo "Python version: $(python --version)"

python -c "import torch; print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"


python -m thesis.cli run "$METHOD" --config "$CONFIG_PATH" 
