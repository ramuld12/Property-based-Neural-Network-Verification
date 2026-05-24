#!/usr/bin/env bash
#SBATCH --job-name=eval_high_lambda
#SBATCH --output=outputs/%x_job-%j.out
#SBATCH --error=outputs/%x_job-%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --array=0-179%10

export TMPDIR=/home/username/tmp
export TMP=/home/username/tmp
export TEMP=/home/username/tmp

echo "Running on $(hostname)"
echo "Working directory: $(pwd)"
nvidia-smi

MODEL_ROOT="outputs/high_lambda"
CROSS_DATA_GOOD="data/ciciot2023_preprocessed_good.tsv"
CROSS_DATA_BAD="data/ciciot2023_preprocessed_bad.tsv"
mapfile -t MODEL_PATHS < <(find "$MODEL_ROOT" -type f -name "model.joblib" | sort)

if [[ "${#MODEL_PATHS[@]}" -eq 0 ]]; then
    echo "No model.joblib files found below ${MODEL_ROOT}"
    exit 1
fi

if [[ "$SLURM_ARRAY_TASK_ID" -ge "${#MODEL_PATHS[@]}" ]]; then
    echo "Task ${SLURM_ARRAY_TASK_ID} is beyond discovered model count ${#MODEL_PATHS[@]}; nothing to evaluate."
    exit 0
fi

MODEL_PATH="${MODEL_PATHS[$SLURM_ARRAY_TASK_ID]}"
RUN_SLURM_DIR="outputs/high_lambda/slurm"
mkdir -p "$RUN_SLURM_DIR"
RUN_STDOUT="${RUN_SLURM_DIR}/${SLURM_JOB_NAME}_task-${SLURM_ARRAY_TASK_ID}_job-${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}.out"
RUN_STDERR="${RUN_SLURM_DIR}/${SLURM_JOB_NAME}_task-${SLURM_ARRAY_TASK_ID}_job-${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}.err"

ENV_NAME="thesis"

module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

echo "Running on $(hostname)"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "Model root: $MODEL_ROOT"
echo "Discovered models: ${#MODEL_PATHS[@]}"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Model path: $MODEL_PATH"
echo "Run stdout: $RUN_STDOUT"
echo "Run stderr: $RUN_STDERR"
echo "Cross data: $CROSS_DATA_GOOD $CROSS_DATA_BAD"

python -m thesis.cli evaluate \
    --model "$MODEL_PATH" \
    --cross-data "$CROSS_DATA_GOOD" "$CROSS_DATA_BAD" \
    >"$RUN_STDOUT" 2>"$RUN_STDERR"
