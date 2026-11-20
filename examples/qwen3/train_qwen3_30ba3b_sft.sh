#!/bin/bash
# Qwen3-30B-A3B 全参数 SFT（配置与 train_qwen3_8b_sft.sh 一致，MoE 并行部分除外）
#
# 用法：
#   bash train_qwen3_30ba3b_sft.sh <模型路径> <数据集路径> <RCCLENV 路径>
#
# 预调参数（2 机 16 卡）：TP=1 CP=1 PP=2 EP=8，DP=8，MBS=1，GBS=32
set -e

MODEL=${1:-Qwen/Qwen3-30B-A3B}
DATASET=${2:-train.jsonl}
RCCLENV=${3:-env.sh}

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

echo "[qwen3-30ba3b-sft] model=${MODEL} nodes=${NNODES} TP=1 CP=1 PP=2 EP=8 DP=8 MBS=1 GBS=32"

# PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NNODES=${NNODES} \
NODE_RANK=${NODE_RANK} \
MASTER_ADDR=${MASTER_ADDR} \
MASTER_PORT=${MASTER_PORT} \
NPROC_PER_NODE=${NPROC_PER_NODE} \
HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES} \
megatron sft \
    --model ${MODEL} \
    --dataset ${DATASET} \
    --load_from_cache_file true \
    --split_dataset_ratio 0.01 \
    --packing true \
    --max_length 4096 \
    --system 'You are a helpful assistant.' \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --tuner_type full \
    --finetune true \
    --tensor_model_parallel_size 1 \
    --sequence_parallel true \
    --context_parallel_size 1 \
    --pipeline_model_parallel_size 2 \
    --expert_model_parallel_size 8 \
    --expert_tensor_parallel_size 1 \
    --micro_batch_size 1 \
    --global_batch_size 32 \
    --train_iters 20 \
    --cross_entropy_loss_fusion true \
    --lr 1e-5 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-6 \
    --attention_backend flash \
    --use_distributed_optimizer true \
    --overlap_grad_reduce true \
    --logging_steps 1 \
    --no_save_optim true \
    --no_save_rng true \
    --save_safetensors false \
    --output_dir ${OUTPUT_DIR}
