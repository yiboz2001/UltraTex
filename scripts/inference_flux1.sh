#!/bin/bash
# UltraTex inference on the FLUX.1-dev backbone.
#   CHECKPOINT  LoRA checkpoint directory (see the Hugging Face release)
set -e

cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH}:${PWD}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_NVLS_ENABLE=0

# Local weight paths. Point these at your download of FLUX.1-dev and the
# text encoders so nothing is fetched from the Hub at run time.
export FLUX_DEV_MODEL_PATH=${FLUX_DEV_MODEL_PATH:-checkpoints/base/FLUX.1-dev/flux1-dev.safetensors}
export AE_MODEL_PATH=${AE_MODEL_PATH:-checkpoints/base/FLUX.1-dev/ae.safetensors}
export T5=${T5:-checkpoints/base/xflux_text_encoders}
export CLIP=${CLIP:-checkpoints/base/clip-vit-large-patch14}

CHECKPOINT=${CHECKPOINT:-checkpoints/flux1/lora}
DECODER_CKPT=${DECODER_CKPT:-checkpoints/flux1/decoder.pt}
EVAL_JSON=${EVAL_JSON:-data/demo.json}
# inference_flux1.py also builds a train dataset for logging; point it at the
# same JSON unless you need a separate split.
TRAIN_JSON=${TRAIN_JSON:-$EVAL_JSON}

LAUNCH_ARGS=()
# Set MAIN_PROCESS_PORT when another job on the machine already uses 29500.
[ -n "${MAIN_PROCESS_PORT:-}" ] && LAUNCH_ARGS+=(--main_process_port "${MAIN_PROCESS_PORT}")

accelerate launch "${LAUNCH_ARGS[@]}" inference_flux1.py \
    --resume_from_checkpoint "${CHECKPOINT}" \
    --decoder_ckpt "${DECODER_CKPT}" \
    --eval_data_json "${EVAL_JSON}" \
    --train_data_json "${TRAIN_JSON}" \
    "$@"
