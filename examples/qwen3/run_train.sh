#!/bin/bash
# 多机训练启动器：生成 slots hostfile，并通过 mpirun 在每个节点调起训练脚本。
#
set -e
set -o pipefail

hostfile_input=${1:?用法: bash run_train.sh <hostfile> <node_num> [pretrain|sft|lora]}
node_num=${2:?用法: bash run_train.sh <hostfile> <node_num> [pretrain|sft|lora]}
TASK=${3:-sft}

case ${TASK} in
    pretrain|sft|lora) ;;
    *) echo "未知任务: ${TASK}（可选 pretrain / sft / lora）" >&2; exit 1 ;;
esac

# ===多节点设置===
HOSTFILE="${hostfile_input}_slots"
rm -f ${HOSTFILE}
sed -n "1,${node_num}p" "${hostfile_input}" | sed 's/$/ slots=1/' > ${HOSTFILE}
HOST="$(sed -n '1p' ${HOSTFILE} | awk -F ' ' '{print $1}')"
SSH_PORT=${SSH_PORT:-29510}

# ===模型, 数据集和rccl配置===
MODEL=${MODEL:-/public/opendas/DL_DATA/llm-models/qwen3/Qwen3-8B}
DATASET=${DATASET:-/public/home/wangxj3/Downloads/model_datasets/datasets/oscar-en-10k/oscar-en-10k.jsonl}
RCCLENV=${RCCLENV:-/public/home/wangxj3/Projects/ms-swift-das/examples/env.sh}
LOG_DIR=${LOG_DIR:-logs}
mkdir -p ${LOG_DIR}

# ===任务分发：train_qwen3_<模型>_<任务>.sh===
MODEL_KEY=${MODEL_KEY:-qwen3_8b}
script=train_${MODEL_KEY}_${TASK}.sh
if [ ! -f "${script}" ]; then
    echo "找不到 ${script}（MODEL_KEY 可选 qwen3_8b / qwen3_32b / qwen3_30ba3b）" >&2
    exit 1
fi

echo "[run_train] task=${TASK} script=${script} nodes=${node_num} master=${HOST}" \
     "model=${MODEL:-<脚本内默认>} dataset=${DATASET:-<脚本内默认>}"
TRAIN_CMD="bash ${script} '${MODEL}' '${DATASET}' '${RCCLENV}'"
MPIRUN_ENV_ARGS=(-x NNODES=${node_num} -x MASTER_ADDR=${HOST})

mpirun -np ${node_num} --hostfile ${HOSTFILE} \
    --allow-run-as-root \
    --bind-to none \
    --mca plm_rsh_no_tree_spawn 1 \
    --mca plm_rsh_args "-p ${SSH_PORT}" \
    "${MPIRUN_ENV_ARGS[@]}" \
    bash -c "${TRAIN_CMD}" 2>&1 | tee ${LOG_DIR}/${script}_$(date +%Y-%m-%d_%H:%M:%S).log
