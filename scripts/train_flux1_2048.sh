#!/bin/bash
# UltraTex training on the FLUX.1-dev backbone at 2048 resolution.
#
# Required model weights (download from black-forest-labs/FLUX.1-dev):
#   FLUX_DEV_MODEL_PATH   flux1-dev.safetensors
#   AE_MODEL_PATH         ae.safetensors
# Optional:
#   DECODER_CKPT          foreground-aware VAE decoder (see train_vae/)
set -e

cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH}:${PWD}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_IB_DISABLE=0

# Local weight paths (README "Weights" layout); override to point elsewhere.
export FLUX_DEV_MODEL_PATH=${FLUX_DEV_MODEL_PATH:-checkpoints/base/FLUX.1-dev/flux1-dev.safetensors}
export AE_MODEL_PATH=${AE_MODEL_PATH:-checkpoints/base/FLUX.1-dev/ae.safetensors}
export T5=${T5:-checkpoints/base/xflux_text_encoders}
export CLIP=${CLIP:-checkpoints/base/clip-vit-large-patch14}

LR=${LR:-1e-5}
RESOLUTION=${RESOLUTION:-2048}
TRAIN_JSON=${TRAIN_JSON:-data/train_demo.json}
EVAL_JSON=${EVAL_JSON:-data/eval_demo.json}
DECODER_CKPT=${DECODER_CKPT:-checkpoints/flux1/decoder.pt}
PROJECT_DIR=${PROJECT_DIR:-outputs/train_flux1_res=${RESOLUTION}_lr=${LR}}

LAUNCH_ARGS=()
# Set MAIN_PROCESS_PORT when another job on the machine already uses 29500.
if [ -n "${MAIN_PROCESS_PORT:-}" ]; then
    LAUNCH_ARGS+=(--main_process_port "${MAIN_PROCESS_PORT}")
    # Single-process run: DeepSpeed would otherwise run MPI discovery and fall
    # back to port 29500 regardless of the launcher's port.
    export MASTER_PORT=${MAIN_PROCESS_PORT} MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
    # DeepSpeed skips MPI port discovery only when all of these are set; a
    # multi-GPU launch overrides them per process.
    export RANK=${RANK:-0} LOCAL_RANK=${LOCAL_RANK:-0} WORLD_SIZE=${WORLD_SIZE:-1}
fi

accelerate launch "${LAUNCH_ARGS[@]}" train_flux1.py \
    --learning_rate "${LR}" \
    --resolution "${RESOLUTION}" \
    --train_data_json "${TRAIN_JSON}" \
    --eval_data_json "${EVAL_JSON}" \
    --decoder_ckpt "${DECODER_CKPT}" \
    --project_dir "${PROJECT_DIR}" \
    "$@"
