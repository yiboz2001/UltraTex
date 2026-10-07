#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"
[ -f .venv/bin/activate ] && source .venv/bin/activate || true

export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export NCCL_NVLS_ENABLE=0
export AE_MODEL_PATH=${AE_MODEL_PATH:-checkpoints/base/flux2/ae.safetensors}
export KLEIN_9B_BASE_MODEL_PATH=${KLEIN_9B_BASE_MODEL_PATH:-checkpoints/base/flux2/flux-2-klein-base-9b.safetensors}
export QWEN3_8B_PATH=${QWEN3_8B_PATH:-checkpoints/base/flux2/qwen3-8b}

# Usage: bash scripts/inference_flux2.sh [--task albedo|mr] [extra Python args]
# --task mr predicts the metallic-roughness map with the MR LoRA.
TASK=${TASK:-albedo}
ARGS=()
while (( $# )); do
    case "$1" in
        --task) TASK=$2; shift 2 ;;
        --task=*) TASK=${1#--task=}; shift ;;
        *) ARGS+=("$1"); shift ;;
    esac
done
case "${TASK}" in
    albedo|mr) ;;
    *) echo "Unknown --task ${TASK}; expected albedo or mr" >&2; exit 2 ;;
esac
set -- "${ARGS[@]+"${ARGS[@]}"}"

if [[ "${TASK}" == mr ]]; then
    CHECKPOINT=${CHECKPOINT:-checkpoints/flux2_mr/lora}
else
    CHECKPOINT=${CHECKPOINT:-checkpoints/flux2/lora}
fi
DECODER_CKPT=${DECODER_CKPT:-checkpoints/flux2/decoder.pt}
EVAL_JSON=${EVAL_JSON:-data/eval_demo.json}
ARCHIVE_ROOT=${ARCHIVE_ROOT:-outputs/inference_flux2}
OUTPUT_DIR=${OUTPUT_DIR:-${ARCHIVE_ROOT}/${TASK}}

for path in "${AE_MODEL_PATH}" "${KLEIN_9B_BASE_MODEL_PATH}" "${CHECKPOINT}/dit_lora.safetensors" "${DECODER_CKPT}"; do
    if [[ ! -f "${path}" ]]; then
        echo "Required file is missing: ${path}" >&2
        exit 1
    fi
done
if [[ ! -d "${QWEN3_8B_PATH}" ]]; then
    echo "Required text encoder directory is missing: ${QWEN3_8B_PATH}" >&2
    exit 1
fi

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
    CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((GPU_COUNT - 1)))
    export CUDA_VISIBLE_DEVICES
else
    GPU_COUNT=$(awk -F, '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")
fi
NUM_MACHINES=${PET_NNODES:-${NNODES:-1}}
MACHINE_RANK=${PET_NODE_RANK:-${NODE_RANK:-0}}
MAIN_IP=${MASTER_ADDR:-127.0.0.1}
MAIN_PORT=${MASTER_PORT:-${MAIN_PROCESS_PORT:-29500}}
TOTAL_PROCESSES=$((NUM_MACHINES * GPU_COUNT))
LAUNCH_ARGS=(--num_processes "${TOTAL_PROCESSES}" --num_machines "${NUM_MACHINES}" --machine_rank "${MACHINE_RANK}" --mixed_precision bf16)
if (( TOTAL_PROCESSES > 1 )); then
    LAUNCH_ARGS=(--multi_gpu "${LAUNCH_ARGS[@]}")
fi
if (( NUM_MACHINES > 1 )); then
    LAUNCH_ARGS+=(--main_process_ip "${MAIN_IP}" --main_process_port "${MAIN_PORT}")
elif [[ -n "${MAIN_PROCESS_PORT:-}" ]]; then
    LAUNCH_ARGS+=(--main_process_port "${MAIN_PROCESS_PORT}")
fi

if [[ ! -f "${EVAL_JSON}" ]]; then
    echo "Inference JSON is missing: ${EVAL_JSON}" >&2
    exit 1
fi

accelerate launch "${LAUNCH_ARGS[@]}" inference_flux2.py \
    --task "${TASK}" \
    --eval_data_json "${EVAL_JSON}" \
    --resume_from_checkpoint "${CHECKPOINT}" \
    --decoder_ckpt "${DECODER_CKPT}" \
    --project_dir "${OUTPUT_DIR}" \
    --save_raw_result true \
    --save_composite true \
    "$@"
