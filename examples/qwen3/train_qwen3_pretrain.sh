#!/bin/bash
set -e

NNODES=${1:-1}
MASTER_ADDR=${2:-127.0.0.1}
MODEL=${3:-Qwen/Qwen3-8B}
MODEL_NAME=${4:-${MODEL_NAME:-$(basename "${MODEL}")}}
MODEL_NAME=$(echo "${MODEL_NAME}" | tr '[:upper:]' '[:lower:]')

NPROC_PER_NODE=${NPROC_PER_NODE:-8}
HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
OUTPUT_DIR=${OUTPUT_DIR:-megatron_output/${MODEL_NAME}-pretrain}
DATASET=${DATASET:-swift/chinese-c4}

# ====== 模型切分：比较模型名自动设置 ======
case ${MODEL_NAME} in
    *30b*a3b*|*30ba3b*)
        TP=1; CP=1; PP=2; EP=8; ETP=1 ;;
    *32b*)
        TP=4; CP=1; PP=2; EP=1; ETP=1 ;;
    *8b*)
        TP=2; CP=1; PP=1; EP=1; ETP=1 ;;
    *)
        echo "未知模型名: ${MODEL_NAME}，支持 qwen3-8b / qwen3-32b / qwen3-30ba3b" >&2
        exit 1 ;;
esac

WORLD_SIZE=$(( NNODES * NPROC_PER_NODE ))
PARALLEL=$(( TP * CP * PP ))
if [ $(( WORLD_SIZE % PARALLEL )) -ne 0 ]; then
    echo "world_size=${WORLD_SIZE} 不能被 TP*CP*PP=${PARALLEL} 整除" >&2
    exit 1
fi
DP=$(( WORLD_SIZE / PARALLEL ))
MBS=1
GBS=$(( DP * PP * 8 ))

echo "[train_qwen3_pretrain] model=${MODEL} name=${MODEL_NAME} world=${WORLD_SIZE}" \
     "TP=${TP} CP=${CP} PP=${PP} EP=${EP} DP=${DP} MBS=${MBS} GBS=${GBS}"

# ====== 并行参数 ======
PARALLEL_ARGS=(
    --tensor_model_parallel_size ${TP}
    --pipeline_model_parallel_size ${PP}
    --context_parallel_size ${CP}
)
if [ "${EP}" -gt 1 ]; then
    PARALLEL_ARGS+=(
        --expert_model_parallel_size ${EP}
        --expert_tensor_parallel_size ${ETP}
    )
fi

# ====== 优化项开关 ======
OPT_ARGS=()
if [ "${RECOMPUTE:-0}" = "1" ]; then
    OPT_ARGS+=(
        --recompute_granularity full
        --recompute_method uniform
        --recompute_num_layers 1
    )
fi
if [ "${OPTIMIZER_CPU_OFFLOAD:-0}" = "1" ]; then
    OPT_ARGS+=(
        --optimizer_cpu_offload true
        --use_precision_aware_optimizer
        --optimizer_offload_fraction 1
    )
fi

EXTRA_KWARGS=${MEGATRON_EXTRA_KWARGS:-}
if [ "${EP}" -gt 1 ]; then
    OPT_ARGS+=(
        --moe_permute_fusion true
        --moe_grouped_gemm true
        --moe_shared_expert_overlap true
        --moe_aux_loss_coeff 1e-6
        --moe_router_fusion true
    )
    if [ "${DEEPEP:-0}" = "1" ]; then
        EXTRA_KWARGS='{"moe_router_fusion": true, "moe_token_dispatcher_type": "flex", "moe_flex_dispatcher_backend": "deepep", "moe_deepep_num_sms": 60}'
    fi
fi
if [ -n "${EXTRA_KWARGS}" ]; then
    OPT_ARGS+=(--megatron_extra_kwargs "${EXTRA_KWARGS}")
fi

source /opt/dtk/env.sh
source "${RCCLENV:-path/to/env.sh}"

export GPU_MAX_HW_QUEUES=4
export NVTE_USE_HIPBLASLT_GROUPEDGEMM=1
export HSA_NO_SCRATCH_RECLAIM=1

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NNODES=${NNODES} \
NODE_RANK=${OMPI_COMM_WORLD_RANK:-0} \
MASTER_ADDR=${MASTER_ADDR} \
MASTER_PORT=${MASTER_PORT:-29510} \
NPROC_PER_NODE=${NPROC_PER_NODE} \
HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES} \
megatron pt \
    --model ${MODEL} \
    --save_safetensors true \
    --dataset ${DATASET} \
    --streaming true \
    --load_from_cache_file true \
    --finetune true \
    "${PARALLEL_ARGS[@]}" \
    --sequence_parallel true \
    --micro_batch_size ${MBS} \
    --global_batch_size ${GBS} \
    --train_iters ${TRAIN_ITERS:-10000} \
    --lr ${LR:-1e-5} \
    --lr_warmup_iters ${WARMUP_ITERS:-300} \
    --min_lr ${MIN_LR:-1e-6} \
    --max_length ${MAX_LENGTH:-8192} \
    --packing true \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --cross_entropy_loss_fusion true \
    --use_distributed_optimizer true \
    --overlap_grad_reduce true \
    --attention_backend flash \
    --no_save_optim true \
    --no_save_rng true \
    --output_dir ${OUTPUT_DIR} \
    "${OPT_ARGS[@]}"
