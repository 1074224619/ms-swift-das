#!/bin/bash
# Qwen3-32B LoRA SFT
#
# 用法：
#   bash train_qwen3_32b_lora.sh <模型路径> <数据集路径> <RCCLENV 路径>
# 例：
#   bash train_qwen3_32b_lora.sh \
#       Qwen/Qwen3-32B /cfs/ms-swift/data/sft.jsonl /cfs/megatron-code/ms-swift/env.sh
#
# 多机由 run_train.sh 调起，NNODES / MASTER_ADDR / NODE_RANK / MASTER_PORT
# 从环境变量读取；单机调试：
#   NNODES=1 MASTER_ADDR=127.0.0.1 bash train_qwen3_32b_lora.sh ...
#
# 预调参数（单机 8 卡）：TP=4 CP=1 PP=1 EP=1，DP=2，MBS=2，GBS=16
# LoRA 只训练低秩适配器，显存占用低，不开 CPU offload。
set -e

MODEL=${1:-Qwen/Qwen3-32B}
DATASET=${2:-}
RCCLENV=${3:-env.sh}
if [ -z "${DATASET}" ]; then
    DATASET='AI-ModelScope/alpaca-gpt4-data-zh#500 AI-ModelScope/alpaca-gpt4-data-en#500 swift/self-cognition#500'
fi
read -r -a DATASETS <<< "${DATASET}"

NNODES=${NNODES:-1}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
NODE_RANK=${NODE_RANK:-${OMPI_COMM_WORLD_RANK:-0}}
MASTER_PORT=${MASTER_PORT:-29510}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
OUTPUT_DIR=${OUTPUT_DIR:-megatron_output/qwen3-32b-lora}

source /opt/dtk/env.sh
source "${RCCLENV}"

export GPU_MAX_HW_QUEUES=4
export NVTE_USE_HIPBLASLT_GROUPEDGEMM=1
export HSA_NO_SCRATCH_RECLAIM=1

echo "[qwen3-32b-lora] model=${MODEL} nodes=${NNODES} TP=4 CP=1 PP=1 EP=1 DP=2 MBS=2 GBS=16"

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NNODES=${NNODES} \
NODE_RANK=${NODE_RANK} \
MASTER_ADDR=${MASTER_ADDR} \
MASTER_PORT=${MASTER_PORT} \
NPROC_PER_NODE=${NPROC_PER_NODE} \
HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES} \
megatron sft \
    --model ${MODEL} \
    --save_safetensors true \
    --dataset "${DATASETS[@]}" \
    --load_from_cache_file true \
    --split_dataset_ratio 0.01 \
    --tuner_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --merge_lora false \
    --finetune true \
    --tensor_model_parallel_size 4 \
    --sequence_parallel true \
    --context_parallel_size 1 \
    --pipeline_model_parallel_size 1 \
    --micro_batch_size 2 \
    --global_batch_size 16 \
    --num_train_epochs 1 \
    --cross_entropy_loss_fusion true \
    --lr 1e-4 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-5 \
    --packing true \
    --attention_backend flash \
    --max_length 2048 \
    --system 'You are a helpful assistant.' \
    --model_author swift \
    --model_name swift-robot \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --no_save_optim true \
    --no_save_rng true \
    --output_dir ${OUTPUT_DIR}
