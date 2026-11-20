#!/bin/bash
# GLM-5.3-Flash 预训练（CPT）—— 由 run_train.sh 通过 mpirun 调起
#
# 一般通过 run_train.sh 启动：
#   bash run_train.sh <hostfile> <node_num> pretrain
# 单机调试也可以直接执行：
#   bash train_glm_pretrain.sh 1 127.0.0.1 ZhipuAI/GLM-5.3-Flash
#
# 参数：$1=NNODES  $2=MASTER_ADDR  $3=模型路径
# 可用环境变量：NPROC_PER_NODE / HIP_VISIBLE_DEVICES / OUTPUT_DIR
#               RCCLENV / MASTER_PORT
# 多机额外设置：NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME（网卡名用 ifconfig 查看）
#               NCCL_IB_HCA NCCL_IB_GID_INDEX（IB/RoCE 环境）
#
# 首次运行前需要给 Megatron-LM 打补丁（只需执行一次）：
#   python -m mcore_bridge.tools.apply_megatron_patch
#
# 环境依赖：transformers>=5.16.1、Megatron-LM dev、mcore-bridge main，
#           安装见 ms-swift/docs/source/Megatron-SWIFT/Quick-start.md
#
# 基于 ms-swift 的 examples/megatron/pretrain.sh 和
# examples/models/glm5_3/megatron_sft.sh 整理，默认单机 8 卡、TP=2/PP=2/EP=4。
# 这里只训练语言模型部分（--language_model_only true），并使用纯文本语料。
# 如需训练 MTP 头，把 --mtp_num_layers 0 改为 1 并加上
# --mtp_loss_scaling_factor 0.1。
#
# --finetune true 表示从模型权重继续预训练；从零初始化请去掉该参数。
set -e

NNODES=${1:-1}
MASTER_ADDR=${2:-127.0.0.1}
MODEL=${3:-ZhipuAI/GLM-5.3-Flash}

NPROC_PER_NODE=${NPROC_PER_NODE:-8}
HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
OUTPUT_DIR=${OUTPUT_DIR:-megatron_output/GLM-5.3-Flash-pretrain}

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
    --language_model_only true \
    --vit_attn_impl sdpa \
    --mtp_num_layers 0 \
    --tensor_model_parallel_size 2 \
    --pipeline_model_parallel_size 2 \
    --expert_model_parallel_size 4 \
    --context_parallel_size 1 \
    --sequence_parallel true \
    --moe_grouped_gemm true \
    --moe_permute_fusion true \
    --moe_aux_loss_coeff 1e-3 \
    --micro_batch_size 1 \
    --global_batch_size 16 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --train_iters 10000 \
    --cross_entropy_loss_fusion true \
    --lr 1e-5 \
    --lr_warmup_iters 300 \
    --min_lr 1e-6 \
    --max_length 4096 \
    --padding_free true \
    --bf16 true \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --no_save_optim true \
    --no_save_rng true \
    --output_dir ${OUTPUT_DIR}
