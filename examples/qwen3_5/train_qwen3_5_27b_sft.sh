#!/bin/bash
# Qwen3.5-27B 全参数 SFT —— 由 run_train.sh 通过 mpirun 调起
#
# 用法：
#   bash train_qwen3_5_27b_sft.sh <模型路径> <数据集路径> <RCCLENV 路径>
# 例：
#   bash train_qwen3_5_27b_sft.sh \
#       Qwen/Qwen3.5-27B train.jsonl /path/to/env.sh
#
# 多机由 run_train.sh 调起，NNODES / MASTER_ADDR / NODE_RANK / MASTER_PORT
# 从环境变量读取；单机调试：
#   NNODES=1 MASTER_ADDR=127.0.0.1 bash train_qwen3_5_27b_sft.sh ...
#
# 预调参数（单机 8 卡）：TP=4 CP=1 PP=2，DP=1，MBS=1，GBS=32
# GBS 必须能被 DP*MBS 整除；多机时 DP 随节点数增大，请同步调大 GBS。
set -e

MODEL=${1:-Qwen/Qwen3.5-27B}
RCCLENV=${3:-env.sh}
# 不传数据集时使用 Qwen3.5 官方示例的文本 SFT 数据集组合。
if [ -n "${2:-}" ]; then
    DATASET_ARGS=(--dataset "${2}")
else
    DATASET_ARGS=(--dataset 'AI-ModelScope/alpaca-gpt4-data-zh#500'
                            'AI-ModelScope/alpaca-gpt4-data-en#500'
                            'swift/self-cognition#500')
fi

NNODES=${NNODES:-1}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
NODE_RANK=${NODE_RANK:-${OMPI_COMM_WORLD_RANK:-0}}
MASTER_PORT=${MASTER_PORT:-29510}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
OUTPUT_DIR=${OUTPUT_DIR:-megatron_output/Qwen3.5-27B-sft}
# 纯文本训练：只加载/训练/保存语言模型部分（不加载 ViT）。
# 需要图文/视频多模态训练时设置 LANGUAGE_MODEL_ONLY=false，
# 并把 --freeze_vit / --freeze_aligner 改为 false。
LANGUAGE_MODEL_ONLY=${LANGUAGE_MODEL_ONLY:-true}

# 集群环境（DTK / RCCL），按实际路径修改
if [ -f /opt/dtk/env.sh ]; then
    source /opt/dtk/env.sh
fi
source "${RCCLENV}"

# HCU 环境建议，参考 ms-swift/docs/source/BestPractices/AMD-support.md
export GPU_MAX_HW_QUEUES=4
export NVTE_USE_HIPBLASLT_GROUPEDGEMM=1
export HSA_NO_SCRATCH_RECLAIM=1
export USE_MCORE_GDN=1                    # Qwen3.5 的 GatedDeltaNet 走 mcore 实现
export SKIP_MULTIMODAL_MTP_VALIDATION=1

echo "[qwen3.5-27b-sft] model=${MODEL} nodes=${NNODES} TP=4 CP=1 PP=2 DP=1 MBS=1 GBS=32"

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NNODES=${NNODES} \
NODE_RANK=${NODE_RANK} \
MASTER_ADDR=${MASTER_ADDR} \
MASTER_PORT=${MASTER_PORT} \
NPROC_PER_NODE=${NPROC_PER_NODE} \
HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES} \
megatron sft \
    --model ${MODEL} \
    --language_model_only ${LANGUAGE_MODEL_ONLY} \
    --save_safetensors true \
    "${DATASET_ARGS[@]}" \
    --load_from_cache_file true \
    --add_non_thinking_prefix true \
    --loss_scale ignore_empty_think \
    --split_dataset_ratio 0.01 \
    --tuner_type full \
    --finetune true \
    --tensor_model_parallel_size 4 \
    --sequence_parallel true \
    --context_parallel_size 1 \
    --pipeline_model_parallel_size 2 \
    --micro_batch_size 1 \
    --global_batch_size 32 \
    --num_train_epochs 1 \
    --freeze_llm false \
    --freeze_vit true \
    --freeze_aligner true \
    --cross_entropy_loss_fusion true \
    --lr 1e-5 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-6 \
    --packing true \
    --optimizer_cpu_offload false \
    --attention_backend flash \
    --max_length 2048 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --logging_steps 1 \
    --no_save_optim true \
    --no_save_rng true \
    --output_dir ${OUTPUT_DIR}
