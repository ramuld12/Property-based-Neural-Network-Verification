#!/usr/bin/env bash
#SBATCH --job-name=ex2_prop
#SBATCH --output=outputs/ex2/properties/slurm/%x_task-%a_job-%A.out
#SBATCH --error=outputs/ex2/properties/slurm/%x_task-%a_job-%A.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --array=0-29


export TMPDIR=/home/username/tmp
export TMP=/home/username/tmp
export TEMP=/home/username/tmp

echo "Running on $(hostname)"
echo "Working directory: $(pwd)"
nvidia-smi

SWEEP_FILE="configs/properties/ex2_prop_commands.txt"
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
