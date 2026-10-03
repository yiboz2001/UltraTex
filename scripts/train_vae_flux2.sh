#!/usr/bin/env bash
# Train the Foreground-Aware VAE decoder for the FLUX.2 backbone.
# The FLUX.2 AE encoder stays frozen; only the decoder is trained, on foreground pixels.
#
# Required: AE_MODEL_PATH  FLUX.2 ae.safetensors (default: checkpoints/base/flux2/ae.safetensors)
# Optional: TRAIN_JSON / EVAL_JSON (default: bundled demo data), OUTPUT_DIR, LR,
#           MAX_TRAIN_STEPS, RESOLUTION, NUM_GPUS, MAIN_PROCESS_PORT
set -Eeuo pipefail

cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && source .venv/bin/activate || true

export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"
export WANDB_MODE=${WANDB_MODE:-offline}
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}

export AE_MODEL_PATH=${AE_MODEL_PATH:-checkpoints/base/flux2/ae.safetensors}
export TRAIN_JSON=${TRAIN_JSON:-data/train_demo.json}
export EVAL_JSON=${EVAL_JSON:-data/eval_demo.json}
export OUTPUT_DIR=${OUTPUT_DIR:-outputs/train_vae_flux2}
export LR=${LR:-1e-5}
export MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-100000}
export RESOLUTION=${RESOLUTION:-2048}

for path in "${AE_MODEL_PATH}" "${TRAIN_JSON}" "${EVAL_JSON}"; do
    if [[ ! -f "${path}" ]]; then
        echo "Required file is missing: ${path}" >&2
        exit 1
    fi
done

# torch.cuda.device_count() respects CUDA_VISIBLE_DEVICES and container GPU
# isolation, so it is safer than counting the host's physical GPUs.
if [[ -n "${NUM_GPUS:-}" ]]; then
    GPU_COUNT=${NUM_GPUS}
else
    GPU_COUNT=$(python -c 'import torch; print(torch.cuda.device_count())')
fi
if ! [[ "${GPU_COUNT}" =~ ^[1-9][0-9]*$ ]]; then
    echo "No CUDA GPU is visible (detected GPU_COUNT=${GPU_COUNT})." >&2
    exit 1
fi

LAUNCH_ARGS=(--num_processes "${GPU_COUNT}" --num_machines 1 --mixed_precision bf16 --dynamo_backend no)
if (( GPU_COUNT > 1 )); then
    LAUNCH_ARGS=(--multi_gpu "${LAUNCH_ARGS[@]}")
fi
if [[ -n "${MAIN_PROCESS_PORT:-}" ]]; then
    LAUNCH_ARGS+=(--main_process_port "${MAIN_PROCESS_PORT}")
    export MASTER_PORT=${MAIN_PROCESS_PORT} MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
    # DeepSpeed skips MPI port discovery only when all of these are set; a
    # multi-GPU launch overrides them per process.
    export RANK=${RANK:-0} LOCAL_RANK=${LOCAL_RANK:-0} WORLD_SIZE=${WORLD_SIZE:-1}
fi

mkdir -p "${OUTPUT_DIR}"
echo "AE:            ${AE_MODEL_PATH}"
echo "Train JSON:    ${TRAIN_JSON}"
echo "Eval JSON:     ${EVAL_JSON}"
echo "Visible GPUs:  ${GPU_COUNT}"
echo "Resolution:    ${RESOLUTION}, LR: ${LR}, steps: ${MAX_TRAIN_STEPS}"
echo "Output:        ${OUTPUT_DIR}"

accelerate launch "${LAUNCH_ARGS[@]}" train_vae/train_decoder_flux2.py "$@"
