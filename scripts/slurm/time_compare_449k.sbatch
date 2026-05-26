#!/usr/bin/env bash
#SBATCH --job-name=time_compare_449k
#SBATCH --output=outputs/%x_job-%j.out
#SBATCH --error=outputs/%x_job-%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100:1
#SBATCH --array=0-27%10

export TMPDIR=/home/username/tmp
export TMP=/home/username/tmp
export TEMP=/home/username/tmp

echo "Running on $(hostname)"
echo "Working directory: $(pwd)"
nvidia-smi
echo "CPU information:"
lscpu | grep -E 'Model name|CPU\(s\)|Thread|Core|Socket'

SWEEP_FILE="configs/time_compare_449k.txt"
LINE_NUMBER=$((SLURM_ARRAY_TASK_ID + 1))
COMMAND=$(grep -v '^[[:space:]]*$' "$SWEEP_FILE" | sed -n "${LINE_NUMBER}p")

if [[ -z "${COMMAND}" ]]; then
    echo "No command found for task ${SLURM_ARRAY_TASK_ID} in ${SWEEP_FILE}"
    exit 1
fi

ENV_NAME="thesis"

module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

echo "Running on $(hostname)"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "Sweep file: $SWEEP_FILE"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running: $COMMAND"

eval "$COMMAND"
