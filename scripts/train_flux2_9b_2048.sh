#!/bin/bash
set -e

cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && source .venv/bin/activate || true

export PYTHONPATH=${PYTHONPATH}:${PWD}
export HF_HOME=${HF_HOME:-checkpoints}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_NVLS_ENABLE=0

FLUX2_CACHE=${FLUX2_CACHE:-checkpoints/base/flux2}
export AE_MODEL_PATH=${FLUX2_CACHE}/ae.safetensors

# Usage:
#   bash train_flux2_9b_2048.sh [4b|9b] [extra trainer args]
#   TASK=mr bash train_flux2_9b_2048.sh   # metallic-roughness LoRA (train_flux2_mr.py)
# Defaults to 9B. All training hyperparameters below can also be overridden by
# environment variables without editing this file.
MODEL_VARIANT=${MODEL_VARIANT:-9b}
if [ "${1:-}" = "4b" ] || [ "${1:-}" = "9b" ]; then
    MODEL_VARIANT=$1
    shift
fi

if [ "${MODEL_VARIANT}" = "9b" ]; then
    MODEL_NAME=flux.2-klein-base-9b
    export KLEIN_9B_BASE_MODEL_PATH=${FLUX2_CACHE}/flux-2-klein-base-9b.safetensors
    export QWEN3_8B_PATH=${FLUX2_CACHE}/qwen3-8b
    MODEL_PATH=${KLEIN_9B_BASE_MODEL_PATH}
    TEXT_ENCODER_PATH=${QWEN3_8B_PATH}
else
    MODEL_NAME=flux.2-klein-base-4b
    export KLEIN_4B_BASE_MODEL_PATH=${FLUX2_CACHE}/flux-2-klein-base-4b.safetensors
    export QWEN3_4B_PATH=${FLUX2_CACHE}/qwen3-4b
    MODEL_PATH=${KLEIN_4B_BASE_MODEL_PATH}
    TEXT_ENCODER_PATH=${QWEN3_4B_PATH}
fi

for required_path in "${AE_MODEL_PATH}" "${MODEL_PATH}"; do
    if [ ! -f "${required_path}" ]; then
        echo "Required local model file is missing: ${required_path}" >&2
        exit 1
    fi
done
if [ ! -d "${TEXT_ENCODER_PATH}" ]; then
    echo "Required local text encoder is missing: ${TEXT_ENCODER_PATH}" >&2
    exit 1
fi

# TASK=albedo (default) trains the albedo LoRA with train_flux2.py;
# TASK=mr trains the metallic-roughness LoRA with train_flux2_mr.py.
TASK=${TASK:-albedo}
case "${TASK}" in
    albedo) TRAIN_SCRIPT=train_flux2.py; DEFAULT_LORA=checkpoints/flux2/lora ;;
    mr)     TRAIN_SCRIPT=train_flux2_mr.py; DEFAULT_LORA=checkpoints/flux2_mr/lora ;;
    *) echo "Unknown TASK=${TASK}; expected albedo or mr" >&2; exit 2 ;;
esac

RESOLUTION=${RESOLUTION:-2048}
LEARNING_RATE=${LEARNING_RATE:-5e-5}
MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-1000000}
CHECKPOINTING_STEPS=${CHECKPOINTING_STEPS:-2000}
WORKERS_PER_GPU=${WORKERS_PER_GPU:-12}
SPARSE_ATTENTION=${SPARSE_ATTENTION:-true}
BUCKET_METADATA_JSON=${BUCKET_METADATA_JSON:-}
DEGRADE_SINGLE_RENDER=${DEGRADE_SINGLE_RENDER:-true}
DEGRADE_SINGLE_RENDER_RESOLUTION=${DEGRADE_SINGLE_RENDER_RESOLUTION:-512}

TRAIN_JSON=${TRAIN_JSON:-data/train_demo.json}
EVAL_JSON=${EVAL_JSON:-data/eval_demo.json}
PROJECT_DIR=${PROJECT_DIR:-outputs/train_flux2_${TASK}_${MODEL_VARIANT}_res=${RESOLUTION}_lr=${LEARNING_RATE}}
DECODER_CKPT=${DECODER_CKPT:-checkpoints/flux2/decoder.pt}
# Resume: an explicit RESUME_FROM_CHECKPOINT wins; otherwise continue from the
# newest checkpoint in PROJECT_DIR; otherwise start from the released LoRA if it
# is present (checkpoints/flux2/lora); otherwise train the LoRA from scratch.
BOOTSTRAP_CHECKPOINT=${BOOTSTRAP_CHECKPOINT:-${DEFAULT_LORA}}

if [ -z "${RESUME_FROM_CHECKPOINT+x}" ]; then
    LATEST_CHECKPOINT=
    LATEST_STEP=-1
    for CANDIDATE in "${PROJECT_DIR}"/checkpoint-*; do
        [ -f "${CANDIDATE}/dit_lora.safetensors" ] || continue
        CANDIDATE_STEP=${CANDIDATE##*-}
        case "${CANDIDATE_STEP}" in
            ''|*[!0-9]*) continue ;;
        esac
        if [ "${CANDIDATE_STEP}" -gt "${LATEST_STEP}" ]; then
            LATEST_STEP=${CANDIDATE_STEP}
            LATEST_CHECKPOINT=${CANDIDATE}
        fi
    done
    if [ -n "${LATEST_CHECKPOINT}" ]; then
        RESUME_FROM_CHECKPOINT=${LATEST_CHECKPOINT}
    elif [ -f "${BOOTSTRAP_CHECKPOINT}/dit_lora.safetensors" ]; then
        RESUME_FROM_CHECKPOINT=${BOOTSTRAP_CHECKPOINT}
    else
        RESUME_FROM_CHECKPOINT=
    fi
fi

if [ -n "${RESUME_FROM_CHECKPOINT}" ] && [ ! -f "${RESUME_FROM_CHECKPOINT}/dit_lora.safetensors" ]; then
    echo "Resume checkpoint is missing dit_lora.safetensors: ${RESUME_FROM_CHECKPOINT}" >&2
    exit 1
fi

if [ ! -f "${DECODER_CKPT}" ]; then
    echo "FLUX.2 decoder checkpoint is missing: ${DECODER_CKPT}" >&2
    exit 1
fi

# Optional: per-object foreground fractions, used to bucket samples of similar
# token cost together. Without it every sample gets the same cost.
if [ -n "${BUCKET_METADATA_JSON}" ] && [ ! -f "${BUCKET_METADATA_JSON}" ]; then
    echo "Bucket metadata JSON is missing: ${BUCKET_METADATA_JSON}" >&2
    exit 1
fi

EXTRA_ARGS=()
# Single-render degradation is an albedo-only augmentation.
if [ "${TASK}" = albedo ]; then
    EXTRA_ARGS+=(--degrade_single_render "${DEGRADE_SINGLE_RENDER}"
                 --degrade_single_render_resolution "${DEGRADE_SINGLE_RENDER_RESOLUTION}")
fi
if [ -n "${RESUME_FROM_CHECKPOINT}" ]; then
    EXTRA_ARGS+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi
if [ -n "${BUCKET_METADATA_JSON}" ]; then
    EXTRA_ARGS+=(--bucket_metadata_json "${BUCKET_METADATA_JSON}")
fi

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
    CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((GPU_COUNT - 1)))
    export CUDA_VISIBLE_DEVICES
else
    GPU_COUNT=$(awk -F, '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")
fi

LAUNCH_ARGS=(--num_processes "${GPU_COUNT}")
if [ "${GPU_COUNT}" -gt 1 ]; then
    LAUNCH_ARGS=(--multi_gpu "${LAUNCH_ARGS[@]}")
fi
if [ -n "${MAIN_PROCESS_PORT:-}" ]; then
    LAUNCH_ARGS+=(--main_process_port "${MAIN_PROCESS_PORT}")
fi

echo "Task: ${TASK} (${TRAIN_SCRIPT})"
echo "Model: ${MODEL_NAME}"
echo "GPUs: ${GPU_COUNT} (${CUDA_VISIBLE_DEVICES})"
echo "Train JSON: ${TRAIN_JSON}"
echo "Eval JSON: ${EVAL_JSON}"
echo "Resolution: ${RESOLUTION}, LR: ${LEARNING_RATE}, steps: ${MAX_TRAIN_STEPS}"
echo "Output: ${PROJECT_DIR}"
echo "Resume: ${RESUME_FROM_CHECKPOINT:-<none, LoRA trained from scratch>}"
echo "Decoder: ${DECODER_CKPT}"
echo "Bucket metadata: ${BUCKET_METADATA_JSON:-<none>}"
echo "Sparse attention: ${SPARSE_ATTENTION}"
echo "Single-render degradation: ${DEGRADE_SINGLE_RENDER} (${DEGRADE_SINGLE_RENDER_RESOLUTION} -> ${RESOLUTION})"

accelerate launch "${LAUNCH_ARGS[@]}" "${TRAIN_SCRIPT}" \
    --model_name "${MODEL_NAME}" \
    --project_dir "${PROJECT_DIR}" \
    --train_data_json "${TRAIN_JSON}" \
    --eval_data_json "${EVAL_JSON}" \
    --learning_rate "${LEARNING_RATE}" \
    --max_train_steps "${MAX_TRAIN_STEPS}" \
    --checkpointing_steps "${CHECKPOINTING_STEPS}" \
    --num_workers "${WORKERS_PER_GPU}" \
    --resolution "${RESOLUTION}" \
    --decoder_ckpt "${DECODER_CKPT}" \
    --batch_size 1 \
    --gradient_accumulation_steps 1 \
    --lora_rank 64 \
    --sparse_attention "${SPARSE_ATTENTION}" \
    --sparse_attention_topk 0.20 \
    --sparse_attention_blkq 128 \
    --sparse_attention_blkk 64 \
    --inference_num_steps 25 \
    --inference_drop_background_tokens true \
    --inference_remove_background true \
    "${EXTRA_ARGS[@]}" \
    "$@"
