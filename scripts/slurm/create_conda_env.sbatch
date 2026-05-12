#!/usr/bin/env bash
#SBATCH --job-name=thesis-exp
#SBATCH --output=outputs/slurm/%x_%j.out
#SBATCH --error=outputs/slurm/%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

export TMPDIR=/home/username/tmp
export TMP=/home/username/tmp
export TEMP=/home/username/tmp

echo "Running on $(hostname)"
echo "Working directory: $(pwd)"
nvidia-smi

mkdir -p outputs/slurm

module load Anaconda3  # Load anaconda

# Accept terms and conditions
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Init anaconda (in the HPC, prefer to use the following instead of 'conda init')
eval "$(conda shell.bash hook)"

ENV_NAME='thesis'

if ! conda env list | grep -q "$ENV_NAME"; then
    echo 'Creating new conda env...'
    conda create --name "$ENV_NAME" python=3.11 -y
    pip install --index-url https://download.pytorch.org/whl/cu126 torch torchvision torchaudio torchmetrics
    python -c "import torch; print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
    pip install -r requirements.txt
    echo 'Created'
else
    echo "Environment '$ENV_NAME' already exists."
fi
conda activate $ENV_NAME
echo "Python version: $(python --version)"

pip install -e .