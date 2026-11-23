#!/bin/bash
# Qwen3.5-27B 预训练（CPT）—— 由 run_train.sh 通过 mpirun 调起
#
# 用法：
#   bash train_qwen3_5_27b_pretrain.sh <模型路径> <数据集路径> <RCCLENV 路径>
# 例：
#   bash train_qwen3_5_27b_pretrain.sh \
#       Qwen/Qwen3.5-27B swift/chinese-c4 /path/to/env.sh
#
# 多机由 run_train.sh 调起，NNODES / MASTER_ADDR / NODE_RANK / MASTER_PORT
# 从环境变量读取；单机调试：
#   NNODES=1 MASTER_ADDR=127.0.0.1 bash train_qwen3_5_27b_pretrain.sh ...
#
# 预调参数（单机 8 卡）：TP=4 CP=1 PP=2，DP=1，MBS=1，GBS=32
# GBS 必须能被 DP*MBS 整除；多机时 DP 随节点数增大，请同步调大 GBS。
# 注意：Qwen3.5-27B 官方未发布 Base 版本，默认从 Instruct 权重继续预训练；
# 如有 Base 权重请通过 $1 传入。
set -e

MODEL=${1:-Qwen/Qwen3.5-27B}
DATASET=${2}
RCCLENV=${3:-env.sh}

NNODES=${NNODES:-1}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
NODE_RANK=${NODE_RANK:-${OMPI_COMM_WORLD_RANK:-0}}
MASTER_PORT=${MASTER_PORT:-29510}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
OUTPUT_DIR=${OUTPUT_DIR:-megatron_output/Qwen3.5-27B-pretrain}

source /opt/dtk/env.sh
source "${RCCLENV}"

# HCU 环境建议，参考 ms-swift/docs/source/BestPractices/AMD-support.md
export GPU_MAX_HW_QUEUES=4
export NVTE_USE_HIPBLASLT_GROUPEDGEMM=1
export HSA_NO_SCRATCH_RECLAIM=1
export USE_MCORE_GDN=1                    # Qwen3.5 的 GatedDeltaNet 走 mcore 实现
export SKIP_MULTIMODAL_MTP_VALIDATION=1

echo "[qwen3.5-27b-pretrain] model=${MODEL} nodes=${NNODES} TP=4 CP=1 PP=2 DP=1 MBS=1 GBS=32"

# 输出目录存在旧 checkpoint 时 Megatron 会自动续训；下面一行每次都会强制从零开始。
rm -rf ${OUTPUT_DIR}

# PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NNODES=${NNODES} \
NODE_RANK=${NODE_RANK} \
MASTER_ADDR=${MASTER_ADDR} \
MASTER_PORT=${MASTER_PORT} \
NPROC_PER_NODE=${NPROC_PER_NODE} \
HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES} \
megatron pt \
    --model ${MODEL} \
    --language_model_only true \
    --save_safetensors true \
    --dataset ${DATASET} \
    --streaming true \
    --packing true \
    --max_length 4096 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --load_from_cache_file true \
    --finetune true \
    --tensor_model_parallel_size 4 \
    --sequence_parallel true \
    --context_parallel_size 1 \
    --pipeline_model_parallel_size 2 \
    --micro_batch_size 1 \
    --global_batch_size 32 \
    --train_iters 20 \
    --cross_entropy_loss_fusion true \
    --lr 1e-5 \
    --lr_warmup_iters 3 \
    --min_lr 1e-6 \
    --use_distributed_optimizer true \
    --overlap_grad_reduce true \
    --attention_backend flash \
    --logging_steps 1 \
    --no_save_optim true \
    --no_save_rng true \
    --output_dir ${OUTPUT_DIR} \
    --external_plugins ../plugin/torch_profiler_plugin.py \
    --callbacks torch_profiler
