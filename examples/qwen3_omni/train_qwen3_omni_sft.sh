#!/bin/bash
# Qwen3-Omni-30B-A3B-Instruct 全参数 SFT —— 由 run_train.sh 通过 mpirun 调起
#
# 一般通过 run_train.sh 启动：
#   bash run_train.sh <hostfile> <node_num> sft
# 单机调试也可以直接执行：
#   bash train_qwen3_omni_sft.sh 1 127.0.0.1 /path/to/Qwen3-Omni-30B-A3B-Instruct
#
# 参数：$1=NNODES  $2=MASTER_ADDR  $3=模型路径
# 可用环境变量：NPROC_PER_NODE / HIP_VISIBLE_DEVICES / OUTPUT_DIR
#               DATA_DIR / SHAREGPT4V_N / TULU_N / VIDEO_N
#               RCCLENV / MASTER_PORT
# 多机额外设置：NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME（网卡名用 ifconfig 查看）
#               NCCL_IB_HCA NCCL_IB_GID_INDEX（IB/RoCE 环境）
#
# 基于 ms-swift 的 examples/megatron/multimodal/omni/moe.sh（LoRA）改为全参数，
# 使用 VeOmni 格式的混合数据（ShareGPT4V 图文描述 / tulu-3 纯文本 / LLaVA-Video 视频）。
# 数据准备见 guide.md；video.json 需先用 prepare_data.py 转成 video_swift.json。
#
# 注意：30B 全参数 SFT 对显存要求较高，显存不足时请改用 train_qwen3_omni_lora.sh。
set -e

NNODES=${1:-1}
MASTER_ADDR=${2:-127.0.0.1}
MODEL=${3:-/public/home/wangxj3/Downloads/model_datasets/models/Qwen3-Omni-30B-A3B-Instruct}

NPROC_PER_NODE=${NPROC_PER_NODE:-8}
HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
OUTPUT_DIR=${OUTPUT_DIR:-megatron_output/Qwen3-Omni-30B-A3B-Instruct-sft}

# VeOmni 数据集目录（数据集中的 images/videos 为相对路径，以该目录为根）
DATA_DIR=${DATA_DIR:-/public/home/wangxj3/Downloads/model_datasets/datasets/VeOmni}
# 各数据集采样条数，可按配比调整
SHAREGPT4V_N=${SHAREGPT4V_N:-10000}
TULU_N=${TULU_N:-5000}
VIDEO_N=${VIDEO_N:-5753}

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
# 相对路径的多媒体文件以 DATA_DIR 为根
export ROOT_IMAGE_DIR="${DATA_DIR}"

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NNODES=${NNODES} \
NODE_RANK=${OMPI_COMM_WORLD_RANK:-0} \
MASTER_ADDR=${MASTER_ADDR} \
MASTER_PORT=${MASTER_PORT:-29510} \
NPROC_PER_NODE=${NPROC_PER_NODE} \
HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES} \
ENABLE_AUDIO_OUTPUT=0 \
IMAGE_MAX_TOKEN_NUM=1024 \
VIDEO_MAX_TOKEN_NUM=128 \
FPS_MAX_FRAMES=12 \
megatron sft \
    --model ${MODEL} \
    --save_safetensors true \
    --dataset "${DATA_DIR}/sharegpt4v_instruct_gpt4-vision_cap100k_coco.json#${SHAREGPT4V_N}" \
              "${DATA_DIR}/tulu-3-sft-mixture/data#${TULU_N}" \
              "${DATA_DIR}/video_swift.json#${VIDEO_N}" \
    --load_from_cache_file true \
    --split_dataset_ratio 0.01 \
    --tuner_type full \
    --finetune true \
    --freeze_llm false \
    --freeze_vit true \
    --freeze_aligner true \
    --expert_model_parallel_size 8 \
    --moe_permute_fusion true \
    --moe_grouped_gemm true \
    --moe_shared_expert_overlap true \
    --moe_aux_loss_coeff 1e-3 \
    --micro_batch_size 1 \
    --global_batch_size 8 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --num_train_epochs 1 \
    --cross_entropy_loss_fusion true \
    --lr 1e-5 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-6 \
    --packing true \
    --sequence_parallel true \
    --attention_backend flash \
    --max_length 4096 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --no_save_optim true \
    --no_save_rng true \
    --output_dir ${OUTPUT_DIR}
