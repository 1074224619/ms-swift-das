#!/bin/bash
# Qwen3-8B 全参数 SFT —— 由 run_train.sh 通过 mpirun 调起
#
# 一般通过 run_train.sh 启动：
#   bash run_train.sh <hostfile> <node_num> sft
# 单机调试也可以直接执行：
#   bash train_qwen3_sft.sh 1 127.0.0.1 Qwen/Qwen3-8B
#
# 参数：$1=NNODES  $2=MASTER_ADDR  $3=模型路径
# 可用环境变量：NPROC_PER_NODE / HIP_VISIBLE_DEVICES / OUTPUT_DIR
#               RCCLENV / MASTER_PORT
# 多机额外设置：NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME（网卡名用 ifconfig 查看）
#               NCCL_IB_HCA NCCL_IB_GID_INDEX（IB/RoCE 环境）
#
# 基于 ms-swift 的 examples/megatron/sft.sh 整理，默认单机 8 卡、TP=2。
# 换用 Qwen3-32B / Qwen3-30B-A3B 等规格时，请相应调整 TP/PP/EP。
set -e

NNODES=${1:-1}
MASTER_ADDR=${2:-127.0.0.1}
MODEL=${3:-Qwen/Qwen3-8B}
RCCLENV=${RCCLENV:-/cfs/megatron-code/ms-swift/env.sh}

NPROC_PER_NODE=${NPROC_PER_NODE:-8}
HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
OUTPUT_DIR=${OUTPUT_DIR:-megatron_output/Qwen3-8B-sft}


source /opt/dtk/env.sh
source "${RCCLENV}"



# HCU 环境建议，参考 ms-swift/docs/source/BestPractices/AMD-support.md
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
megatron sft \
    --model ${MODEL} \
    --save_safetensors true \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#500' \
              'AI-ModelScope/alpaca-gpt4-data-en#500' \
              'swift/self-cognition#500' \
    --load_from_cache_file true \
    --split_dataset_ratio 0.01 \
    --tuner_type full \
    --finetune true \
    --tensor_model_parallel_size 2 \
    --micro_batch_size 4 \
    --global_batch_size 16 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --num_train_epochs 1 \
    --cross_entropy_loss_fusion true \
    --lr 1e-5 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-6 \
    --packing true \
    --sequence_parallel true \
    --attention_backend flash \
    --max_length 2048 \
    --system 'You are a helpful assistant.' \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --no_save_optim true \
    --no_save_rng true \
    --output_dir ${OUTPUT_DIR}
