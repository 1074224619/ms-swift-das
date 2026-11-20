#!/bin/bash
# GLM-5.3-Flash LoRA SFT —— 由 run_train.sh 通过 mpirun 调起
#
# 一般通过 run_train.sh 启动：
#   bash run_train.sh <hostfile> <node_num> lora
# 单机调试也可以直接执行：
#   bash train_glm_lora.sh 1 127.0.0.1 ZhipuAI/GLM-5.3-Flash
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
# 基于 ms-swift 的 examples/models/glm5_3/megatron_sft.sh 整理。
# GLM-5.3 的 LoRA 目标模块沿用官方示例的 MLA + MoE 投影层列表；
# 也可以直接写 --target_modules all-linear。
# 训练后可用 megatron export --merge_lora true 合并权重。
set -e

NNODES=${1:-1}
MASTER_ADDR=${2:-127.0.0.1}
MODEL=${3:-ZhipuAI/GLM-5.3-Flash}

NPROC_PER_NODE=${NPROC_PER_NODE:-8}
HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
OUTPUT_DIR=${OUTPUT_DIR:-megatron_output/GLM-5.3-Flash-lora}

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
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#500' \
              'AI-ModelScope/alpaca-gpt4-data-en#500' \
              'swift/self-cognition#500' \
    --load_from_cache_file true \
    --split_dataset_ratio 0.01 \
    --tuner_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules linear_q_down_proj linear_q_up_proj linear_kv_down_proj linear_kv_up_proj linear_proj f_a_proj f_b_proj g_a_proj g_b_proj beta_proj out_proj \
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
    --num_train_epochs 1 \
    --cross_entropy_loss_fusion true \
    --lr 1e-4 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-5 \
    --max_length 4096 \
    --padding_free true \
    --bf16 true \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --no_save_optim true \
    --no_save_rng true \
    --output_dir ${OUTPUT_DIR}
