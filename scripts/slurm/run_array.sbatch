#!/usr/bin/env bash
#SBATCH --job-name=cicids_small_grid
#SBATCH --output=outputs/slurm/%x_task-%a_job-%A.out
#SBATCH --error=outputs/slurm/%x_task-%a_job-%A.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --array=0-31

export TMPDIR=/home/username/tmp
export TMP=/home/username/tmp
export TEMP=/home/username/tmp

SWEEP_FILE="${1:?Usage: sbatch run_array.sbatch <sweep.txt>}"
LINE_NUMBER=$((SLURM_ARRAY_TASK_ID + 1))
COMMAND=$(sed -n "${LINE_NUMBER}p" "$SWEEP_FILE")

ENV_NAME="thesis"

module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"
echo "Running on $(hostname)"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"

echo "Python version: $(python --version)"
echo "Running: $COMMAND"
eval "$COMMAND"