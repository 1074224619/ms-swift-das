#!/bin/bash
set -e

MODEL=${1:-Qwen/Qwen3-32B-Base}
DATASET=${2:-dataset.jsonl}
RCCLENV=${3:-env.sh}

NNODES=${NNODES:-1}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
NODE_RANK=${NODE_RANK:-${OMPI_COMM_WORLD_RANK:-0}}
MASTER_PORT=${MASTER_PORT:-29511}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
OUTPUT_DIR=${OUTPUT_DIR:-megatron_output/qwen3-32b-base-pretrain}

source /opt/dtk/env.sh
source "${RCCLENV}"

export GPU_MAX_HW_QUEUES=4
export NVTE_USE_HIPBLASLT_GROUPEDGEMM=1

echo "[qwen3-32b-pretrain] model=${MODEL} nodes=${NNODES} TP=4 CP=1 PP=2 EP=1 MBS=1 GBS=32"

rm -rf ${OUTPUT_DIR}
python3 - <<'PY'
import swift.megatron.trainers.base as m
p = m.__file__
s = open(p).read()
old = """        if args.mcore_model is None:
            self.bridge.load_weights(models, args.model_dir)"""
new = """        if args.mcore_model is None and not args.perform_initialization:
            self.bridge.load_weights(models, args.model_dir)"""
if old in s:
    open(p, 'w').write(s.replace(old, new, 1))
    print('[patch]', p)
else:
    print('[patch] no change:', p)
PY

# PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NNODES=${NNODES} \
NODE_RANK=${NODE_RANK} \
MASTER_ADDR=${MASTER_ADDR} \
MASTER_PORT=${MASTER_PORT} \
NPROC_PER_NODE=${NPROC_PER_NODE} \
HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES} \
megatron pt \
    --model ${MODEL} \
    --perform_initialization false \
    --dataset ${DATASET} \
    --max_length 4096 \
    --streaming false \
    --packing true \
    --truncation_strategy split \
    --load_from_cache_file true \
    --finetune true \
    --tensor_model_parallel_size 4 \
    --sequence_parallel true \
    --context_parallel_size 1 \
    --pipeline_model_parallel_size 2 \
    --micro_batch_size 1 \
    --global_batch_size 32 \
    --train_iters 10 \
    --cross_entropy_loss_fusion true \
    --lr 1e-5 \
    --lr_warmup_iters 3 \
    --min_lr 1e-6 \
    --use_distributed_optimizer true \
    --overlap_grad_reduce true \
    --attention_backend flash \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --logging_steps 1 \
    --save_steps 0 \
    --no_save_optim true \
    --no_save_rng true \
    --save_safetensors false \
    --output_dir ${OUTPUT_DIR}
