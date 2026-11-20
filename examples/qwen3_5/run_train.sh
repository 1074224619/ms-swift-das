#!/bin/bash
# 多机训练启动器：生成 slots hostfile，并通过 mpirun 在每个节点调起
# train_qwen3_5_<task>.sh。
#
# 用法：
#   bash run_train.sh <hostfile> <node_num> [pretrain|sft|lora]
# 例如：
#   bash run_train.sh hosts.txt 2 sft
#
# hostfile 每行一个主机名/IP，node_num 为使用的节点数。
# 可用环境变量：MODEL、RCCLENV、SSH_PORT、LOG_DIR
set -e
set -o pipefail

hostfile_input=${1:?用法: bash run_train.sh <hostfile> <node_num> [pretrain|sft|lora]}
node_num=${2:?用法: bash run_train.sh <hostfile> <node_num> [pretrain|sft|lora]}
TASK=${3:-sft}

case ${TASK} in
    pretrain|sft|lora) ;;
    *) echo "未知任务: ${TASK}（可选 pretrain / sft / lora）" >&2; exit 1 ;;
esac

script=train_qwen3_5_${TASK}.sh
MODEL=${MODEL:-Qwen/Qwen3.5-35B-A3B}
RCCLENV=${RCCLENV:-/cfs/megatron-code/ms-swift/env.sh}
export RCCLENV
LOG_DIR=${LOG_DIR:-logs}
mkdir -p ${LOG_DIR}

HOSTFILE="${hostfile_input}_slots"
rm -f ${HOSTFILE}
sed -n "1,${node_num}p" ${hostfile_input} | sed 's/$/ slots=1/' > ${HOSTFILE}

HOST="$(sed -n '1p' ${HOSTFILE} | awk -F ' ' '{print $1}')"

mpirun -np ${node_num} --hostfile ${HOSTFILE} \
    --allow-run-as-root \
    --bind-to none \
    --mca plm_rsh_no_tree_spawn 1 \
    --mca plm_rsh_args "-p ${SSH_PORT:-29511}" \
    bash -c "bash ${script} ${node_num} ${HOST} ${MODEL}" 2>&1 | tee ${LOG_DIR}/${script}_$(date +%Y-%m-%d_%H:%M:%S).log
