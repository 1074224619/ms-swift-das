#!/bin/bash
# DeepSeek-V4-Flash LoRA SFT —— 由 run_train.sh 通过 mpirun 调起
#
# 一般通过 run_train.sh 启动：
#   bash run_train.sh <hostfile> <node_num> lora
# 单机调试也可以直接执行：
#   bash train_deepseek_lora.sh 1 127.0.0.1 deepseek-ai/DeepSeek-V4-Flash
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
#     --pipeline_model_parallel_layout。
#   - 如需 FP8 + LoRA，追加 --fp8_recipe blockwise --fp8_format e4m3
#     --fp8_param_gather true，并保持 --merge_lora false（FP8 精度下
#     LoRA delta 会被舍入为 0，需用 BF16 权重做 Merge-LoRA）。
#   - DeepSeek-V4.1 需要专用 Megatron 分支，参考
#     ms-swift/examples/models/deepseek_v41/megatron_sft.sh。
#
# 基于 ms-swift 的 docs/source/BestPractices/deepseek-v4.md 整理，
# 是 V4-Flash 单机最推荐的微调方式。
# 训练后可用 megatron export --merge_lora true 合并权重。
set -e

NNODES=${1:-1}
MASTER_ADDR=${2:-127.0.0.1}
MODEL=${3:-deepseek-ai/DeepSeek-V4-Flash}

NPROC_PER_NODE=${NPROC_PER_NODE:-8}
HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
OUTPUT_DIR=${OUTPUT_DIR:-megatron_output/DeepSeek-V4-Flash-lora}

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
megatron sft \
    --model ${MODEL} \
    --save_safetensors true \
    --merge_lora false \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#1000' \
              'AI-ModelScope/alpaca-gpt4-data-en#1000' \
              'swift/self-cognition#1000' \
    --load_from_cache_file true \
    --add_non_thinking_prefix true \
    --loss_scale ignore_empty_think \
    --split_dataset_ratio 0.01 \
    --tuner_type lora \
    --lora_rank 16 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --finetune true \
    --tensor_model_parallel_size 1 \
    --expert_model_parallel_size 8 \
    --mtp_num_layers 1 \
    --moe_permute_fusion true \
    --moe_grouped_gemm true \
    --moe_shared_expert_overlap true \
    --moe_aux_loss_coeff 1e-3 \
    --micro_batch_size 4 \
    --global_batch_size 32 \
    --padding_free false \
    --group_by_length true \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --num_train_epochs 1 \
    --cross_entropy_loss_fusion true \
    --lr 1e-4 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-5 \
    --max_length 4096 \
    --bf16 true \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --no_save_optim true \
    --no_save_rng true \
    --output_dir ${OUTPUT_DIR}
