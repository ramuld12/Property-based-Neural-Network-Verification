#!/usr/bin/env bash
#SBATCH --job-name=eval_high_lambda
#SBATCH --output=outputs/%x_job-%j.out
#SBATCH --error=outputs/%x_job-%j.err
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

SWEEP_FILE="configs/properties/eval_high_lambda_models.txt"
COMMAND=$(grep -v '^[[:space:]]*$' "$SWEEP_FILE" | sed -n "1p")

if [[ -z "${COMMAND}" ]]; then
    echo "No command found in ${SWEEP_FILE}"
    exit 1
fi

RUN_SLURM_DIR="outputs/high_lambda/slurm"
mkdir -p "$RUN_SLURM_DIR"
RUN_STDOUT="${RUN_SLURM_DIR}/${SLURM_JOB_NAME}_job-${SLURM_JOB_ID}.out"
RUN_STDERR="${RUN_SLURM_DIR}/${SLURM_JOB_NAME}_job-${SLURM_JOB_ID}.err"

ENV_NAME="thesis"

module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

echo "Running on $(hostname)"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "Sweep file: $SWEEP_FILE"
echo "Run stdout: $RUN_STDOUT"
echo "Run stderr: $RUN_STDERR"
echo "Running: $COMMAND"

eval "$COMMAND" >"$RUN_STDOUT" 2>"$RUN_STDERR"
