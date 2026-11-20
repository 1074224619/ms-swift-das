#!/bin/bash
# Qwen3-30B-A3B 全参数 SFT
#
# 用法：
#   bash train_qwen3_30ba3b_sft.sh <模型路径> <数据集路径> <RCCLENV 路径>
# 例：
#   bash train_qwen3_30ba3b_sft.sh \
#       Qwen/Qwen3-30B-A3B /cfs/ms-swift/data/sft.jsonl /cfs/megatron-code/ms-swift/env.sh
#
# 多机由 run_train.sh 调起，NNODES / MASTER_ADDR / NODE_RANK / MASTER_PORT
# 从环境变量读取。
#
# 预调参数（2 节点 × 8 卡）：TP=1 CP=1 PP=2 EP=8，DP=8，MBS=1，GBS=64
# EP=8 要求 DP 能被 8 整除，单机 8 卡（DP=4）不满足，至少需要 2 个节点，
# 请通过 run_train.sh 启动。
set -e

MODEL=${1:-Qwen/Qwen3-30B-A3B}
DATASET=${2:-}
RCCLENV=${3:-env.sh}
if [ -z "${DATASET}" ]; then
    DATASET='AI-ModelScope/alpaca-gpt4-data-zh#500 AI-ModelScope/alpaca-gpt4-data-en#500 swift/self-cognition#500'
fi
read -r -a DATASETS <<< "${DATASET}"

NNODES=${NNODES:-1}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
NODE_RANK=${NODE_RANK:-${OMPI_COMM_WORLD_RANK:-0}}
MASTER_PORT=${MASTER_PORT:-29510}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
OUTPUT_DIR=${OUTPUT_DIR:-megatron_output/qwen3-30ba3b-sft}

source /opt/dtk/env.sh
source "${RCCLENV}"

export GPU_MAX_HW_QUEUES=4
export NVTE_USE_HIPBLASLT_GROUPEDGEMM=1
export HSA_NO_SCRATCH_RECLAIM=1

WORLD_SIZE=$(( NNODES * NPROC_PER_NODE ))
DP=$(( WORLD_SIZE / (1 * 1 * 2) ))
if [ $(( WORLD_SIZE % 2 )) -ne 0 ] || [ $(( DP % 8 )) -ne 0 ]; then
    echo "切分不匹配: world_size=${WORLD_SIZE} DP=${DP}，EP=8 需要 DP 能整除 8（至少 2 节点）" >&2
    exit 1
fi

echo "[qwen3-30ba3b-sft] model=${MODEL} nodes=${NNODES} TP=1 CP=1 PP=2 EP=8 DP=${DP} MBS=1 GBS=64"

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NNODES=${NNODES} \
NODE_RANK=${NODE_RANK} \
MASTER_ADDR=${MASTER_ADDR} \
MASTER_PORT=${MASTER_PORT} \
NPROC_PER_NODE=${NPROC_PER_NODE} \
HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES} \
megatron sft \
    --model ${MODEL} \
    --save_safetensors true \
    --dataset "${DATASETS[@]}" \
    --load_from_cache_file true \
    --split_dataset_ratio 0.01 \
    --tuner_type full \
    --finetune true \
    --tensor_model_parallel_size 1 \
    --context_parallel_size 1 \
    --pipeline_model_parallel_size 2 \
    --expert_model_parallel_size 8 \
    --expert_tensor_parallel_size 1 \
    --moe_permute_fusion true \
    --moe_grouped_gemm true \
    --moe_shared_expert_overlap true \
    --moe_aux_loss_coeff 1e-6 \
    --moe_router_fusion true \
    --micro_batch_size 1 \
    --global_batch_size 64 \
    --num_train_epochs 1 \
    --cross_entropy_loss_fusion true \
    --lr 1e-5 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-6 \
    --sequence_parallel true \
    --packing true \
    --attention_backend flash \
    --max_length 2048 \
    --system 'You are a helpful assistant.' \
    --model_author swift \
    --model_name swift-robot \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --use_distributed_optimizer true \
    --overlap_grad_reduce true \
    --optimizer_cpu_offload true \
    --use_precision_aware_optimizer \
    --optimizer_offload_fraction 1 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --no_save_optim true \
    --no_save_rng true \
    --output_dir ${OUTPUT_DIR}

# deepep（可选）：
# --megatron_extra_kwargs '{"moe_router_fusion": true, "moe_token_dispatcher_type": "flex", "moe_flex_dispatcher_backend": "deepep", "moe_deepep_num_sms": 60}'
