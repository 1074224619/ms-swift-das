#!/bin/bash
# Qwen3.5-35B-A3B LoRA SFT —— 由 run_train.sh 通过 mpirun 调起
#
# 用法：
#   bash train_qwen3_5_35ba3b_lora.sh <模型路径> <数据集路径> <RCCLENV 路径>
# 例：
#   bash train_qwen3_5_35ba3b_lora.sh \
#       Qwen/Qwen3.5-35B-A3B dataset.jsonl /path/to/env.sh
#
# 多机由 run_train.sh 调起，NNODES / MASTER_ADDR / NODE_RANK / MASTER_PORT
# 从环境变量读取；单机调试：
#   NNODES=1 MASTER_ADDR=127.0.0.1 bash train_qwen3_5_35ba3b_lora.sh ...
#
# 预调参数（单机 8 卡）：TP=1 CP=1 PP=1 EP=8，DP=8，MBS=1，GBS=16
# 训练后可用 megatron export --merge_lora true 合并权重。
set -e

MODEL=${1:-Qwen/Qwen3.5-35B-A3B}
DATASET=${2}
RCCLENV=${3:-env.sh}

NNODES=${NNODES:-1}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
NODE_RANK=${NODE_RANK:-${OMPI_COMM_WORLD_RANK:-0}}
MASTER_PORT=${MASTER_PORT:-29510}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
OUTPUT_DIR=${OUTPUT_DIR:-megatron_output/Qwen3.5-35B-A3B-lora}

source /opt/dtk/env.sh
source "${RCCLENV}"

# HCU 环境建议，参考 ms-swift/docs/source/BestPractices/AMD-support.md
export GPU_MAX_HW_QUEUES=4
export NVTE_USE_HIPBLASLT_GROUPEDGEMM=1
export HSA_NO_SCRATCH_RECLAIM=1
export USE_MCORE_GDN=1                    # Qwen3.5 的 GatedDeltaNet 走 mcore 实现
export SKIP_MULTIMODAL_MTP_VALIDATION=1

echo "[qwen3.5-35b-a3b-lora] model=${MODEL} nodes=${NNODES} TP=1 CP=1 PP=1 EP=8 DP=8 MBS=1 GBS=16"

# PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NNODES=${NNODES} \
NODE_RANK=${NODE_RANK} \
MASTER_ADDR=${MASTER_ADDR} \
MASTER_PORT=${MASTER_PORT} \
NPROC_PER_NODE=${NPROC_PER_NODE} \
HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES} \
megatron sft \
    --model ${MODEL} \
    --language_model_only true \
    --save_safetensors true \
    --merge_lora false \
    --dataset ${DATASET} \
    --max_length 4096 \
    --packing true \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --load_from_cache_file true \
    --add_non_thinking_prefix true \
    --loss_scale ignore_empty_think \
    --split_dataset_ratio 0.01 \
    --tuner_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules linear_qkv linear_proj in_proj out_proj \
    --finetune true \
    --tensor_model_parallel_size 1 \
    --sequence_parallel true \
    --context_parallel_size 1 \
    --pipeline_model_parallel_size 1 \
    --expert_model_parallel_size 8 \
    --moe_permute_fusion true \
    --moe_grouped_gemm true \
    --moe_shared_expert_overlap true \
    --moe_aux_loss_coeff 1e-6 \
    --megatron_extra_kwargs '{"moe_router_fusion": true,}' \
    --micro_batch_size 1 \
    --global_batch_size 32 \
    --train_iters 20 \
    --freeze_llm false \
    --freeze_vit true \
    --freeze_aligner true \
    --cross_entropy_loss_fusion true \
    --lr 1e-4 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-5 \
    --attention_backend flash \
    --logging_steps 1 \
    --no_save_optim true \
    --no_save_rng true \
    --output_dir ${OUTPUT_DIR} \
    --external_plugins ../plugin/torch_profiler_plugin.py \
    --callbacks torch_profiler
