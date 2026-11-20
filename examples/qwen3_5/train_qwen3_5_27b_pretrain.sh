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
# 注意：Qwen3.5-27B 官方未发布 Base 版本，默认从 Instruct 权重继续预训练；
# 如有 Base 权重请通过 $1 传入。
# GBS 必须能被 DP*MBS 整除；多机时 DP 随节点数增大，请同步调大 GBS。
# 冒烟测试可用 TRAIN_ITERS=10 WARMUP_ITERS=3 覆盖迭代数。
set -e

MODEL=${1:-Qwen/Qwen3.5-27B}
# 默认使用 swift/chinese-c4（streaming）；传入本地 jsonl 时自动改为非 streaming，
# 也可用 STREAMING=true/false 覆盖。
if [ -n "${2:-}" ]; then
    DATASET=${2}
    STREAMING=${STREAMING:-false}
else
    DATASET=swift/chinese-c4
    STREAMING=${STREAMING:-true}
fi
RCCLENV=${3:-env.sh}

NNODES=${NNODES:-1}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
NODE_RANK=${NODE_RANK:-${OMPI_COMM_WORLD_RANK:-0}}
MASTER_PORT=${MASTER_PORT:-29510}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
OUTPUT_DIR=${OUTPUT_DIR:-megatron_output/Qwen3.5-27B-pretrain}
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

echo "[qwen3.5-27b-pretrain] model=${MODEL} nodes=${NNODES} TP=4 CP=1 PP=2 DP=1 MBS=1 GBS=32"

# 输出目录存在旧 checkpoint 时 Megatron 会自动续训；需要强制从零开始时取消下一行注释。
# rm -rf ${OUTPUT_DIR}

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NNODES=${NNODES} \
NODE_RANK=${NODE_RANK} \
MASTER_ADDR=${MASTER_ADDR} \
MASTER_PORT=${MASTER_PORT} \
NPROC_PER_NODE=${NPROC_PER_NODE} \
HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES} \
megatron pt \
    --model ${MODEL} \
    --language_model_only ${LANGUAGE_MODEL_ONLY} \
    --save_safetensors true \
    --dataset ${DATASET} \
    --streaming ${STREAMING} \
    --load_from_cache_file true \
    --finetune true \
    --tensor_model_parallel_size 4 \
    --sequence_parallel true \
    --context_parallel_size 1 \
    --pipeline_model_parallel_size 2 \
    --micro_batch_size 1 \
    --global_batch_size 32 \
    --train_iters ${TRAIN_ITERS:-10000} \
    --cross_entropy_loss_fusion true \
    --lr 1e-5 \
    --lr_warmup_iters ${WARMUP_ITERS:-300} \
    --min_lr 1e-6 \
    --use_distributed_optimizer true \
    --overlap_grad_reduce true \
    --optimizer_cpu_offload false \
    --attention_backend flash \
    --packing true \
    --max_length 8192 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --logging_steps 1 \
    --no_save_optim true \
    --no_save_rng true \
    --output_dir ${OUTPUT_DIR}
