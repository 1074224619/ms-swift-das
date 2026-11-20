#!/bin/bash
# Qwen3-32B 预训练（CPT）
#
# 用法：
#   bash train_qwen3_32b_pretrain.sh <模型路径> <数据集路径> <RCCLENV 路径>
# 例：
#   bash train_qwen3_32b_pretrain.sh \
#       Qwen/Qwen3-32B-Base /cfs/ms-swift/data/pretrain /cfs/megatron-code/ms-swift/env.sh
#
# 多机由 run_train.sh 调起，NNODES / MASTER_ADDR / NODE_RANK / MASTER_PORT
# 从环境变量读取；单机调试：
#   NNODES=1 MASTER_ADDR=127.0.0.1 bash train_qwen3_32b_pretrain.sh ...
#
# 预调参数（单机 8 卡）：TP=4 CP=1 PP=2 EP=1，DP=1，MBS=1，GBS=16
# 优化器状态 CPU offload（fraction=1），显存不够可保持，CPU 内存紧张可去掉。
# 如果数据集是本地文件且 --streaming 报错，删掉该行即可。
set -e

MODEL=${1:-Qwen/Qwen3-32B-Base}
DATASET=${2:-swift/chinese-c4}
RCCLENV=${3:-/cfs/megatron-code/ms-swift/env.sh}

NNODES=${NNODES:-1}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
NODE_RANK=${NODE_RANK:-${OMPI_COMM_WORLD_RANK:-0}}
MASTER_PORT=${MASTER_PORT:-29510}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
OUTPUT_DIR=${OUTPUT_DIR:-megatron_output/qwen3-32b-base-pretrain}

source /opt/dtk/env.sh
source "${RCCLENV}"

export GPU_MAX_HW_QUEUES=4
export NVTE_USE_HIPBLASLT_GROUPEDGEMM=1
export HSA_NO_SCRATCH_RECLAIM=1

echo "[qwen3-32b-pretrain] model=${MODEL} nodes=${NNODES} TP=4 CP=1 PP=2 EP=1 MBS=1 GBS=16"

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NNODES=${NNODES} \
NODE_RANK=${NODE_RANK} \
MASTER_ADDR=${MASTER_ADDR} \
MASTER_PORT=${MASTER_PORT} \
NPROC_PER_NODE=${NPROC_PER_NODE} \
HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES} \
megatron pt \
    --model ${MODEL} \
    --save_safetensors true \
    --dataset ${DATASET} \
    --streaming true \
    --packing true \
    --load_from_cache_file true \
    --finetune true \
    --tensor_model_parallel_size 4 \
    --context_parallel_size 1 \
    --pipeline_model_parallel_size 2 \
    --micro_batch_size 1 \
    --global_batch_size 16 \
    --train_iters 10000 \
    --cross_entropy_loss_fusion true \
    --lr 1e-5 \
    --lr_warmup_iters 300 \
    --min_lr 1e-6 \
    --sequence_parallel true \
    --use_distributed_optimizer true \
    --overlap_grad_reduce true \
    --optimizer_cpu_offload true \
    --use_precision_aware_optimizer \
    --optimizer_offload_fraction 1 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --attention_backend flash \
    --max_length 8192 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --no_save_optim true \
    --no_save_rng true \
    --output_dir ${OUTPUT_DIR}
