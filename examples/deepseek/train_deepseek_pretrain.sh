#!/bin/bash
# DeepSeek-V4-Flash-Base 预训练（CPT）—— 由 run_train.sh 通过 mpirun 调起
#
# 一般通过 run_train.sh 启动：
#   bash run_train.sh <hostfile> <node_num> pretrain
# 单机调试也可以直接执行：
#   bash train_deepseek_pretrain.sh 1 127.0.0.1 deepseek-ai/DeepSeek-V4-Flash-Base
#
# 参数：$1=NNODES  $2=MASTER_ADDR  $3=模型路径
# 可用环境变量：NPROC_PER_NODE / HIP_VISIBLE_DEVICES / OUTPUT_DIR
#               RCCLENV / MASTER_PORT
# 多机额外设置：NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME（网卡名用 ifconfig 查看）
#               NCCL_IB_HCA NCCL_IB_GID_INDEX（IB/RoCE 环境）
#
# 环境依赖：Megatron-Core dev（或 >=0.20）、最新 mcore-bridge 和 ms-swift，
#           安装见 ms-swift/docs/source/Megatron-SWIFT/Quick-start.md
#
# 注意：
#   - DeepSeek-V4 暂不支持 TP，tensor_model_parallel_size 固定为 1；
#     支持 CP / PP / EP / DP。PP>1 时必须同时设置
#     --pipeline_model_parallel_layout，例如 PP=2 时：
#       --pipeline_model_parallel_size 2 \
#       --pipeline_model_parallel_layout 'Et*22|t*21mL'
#   - 如需 FP8 训练，追加：
#       --fp8_recipe blockwise --fp8_format e4m3 --fp8_param_gather true
#   - DeepSeek-V4.1 需要专用 Megatron 分支，且同样不支持 TP/SP，
#     参考 ms-swift/examples/models/deepseek_v41/megatron_sft.sh。
#
# 基于 ms-swift 的 examples/megatron/pretrain.sh 和
# docs/source/BestPractices/deepseek-v4.md 整理。
# --finetune true 表示从模型权重继续预训练；从零初始化请去掉该参数。
set -e

NNODES=${1:-1}
MASTER_ADDR=${2:-127.0.0.1}
MODEL=${3:-deepseek-ai/DeepSeek-V4-Flash-Base}

NPROC_PER_NODE=${NPROC_PER_NODE:-8}
HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
OUTPUT_DIR=${OUTPUT_DIR:-megatron_output/DeepSeek-V4-Flash-Base-pretrain}

# 集群环境（DTK / RCCL），按实际路径修改
if [ -f /opt/dtk/env.sh ]; then
    source /opt/dtk/env.sh
fi
RCCLENV=${RCCLENV:-/cfs/megatron-code/ms-swift/env.sh}
if [ -f "${RCCLENV}" ]; then
    source "${RCCLENV}"
fi

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
megatron pt \
    --model ${MODEL} \
    --save_safetensors true \
    --dataset swift/chinese-c4 \
    --streaming true \
    --load_from_cache_file true \
    --finetune true \
    --tensor_model_parallel_size 1 \
    --expert_model_parallel_size 8 \
    --mtp_num_layers 1 \
    --moe_permute_fusion true \
    --moe_grouped_gemm true \
    --moe_shared_expert_overlap true \
    --moe_aux_loss_coeff 1e-3 \
    --micro_batch_size 1 \
    --global_batch_size 16 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --train_iters 10000 \
    --cross_entropy_loss_fusion true \
    --masked_softmax_fusion false \
    --lr 1e-5 \
    --lr_warmup_iters 300 \
    --min_lr 1e-6 \
    --attention_backend flash \
    --max_length 8192 \
    --bf16 true \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --no_save_optim true \
    --no_save_rng true \
    --output_dir ${OUTPUT_DIR}
