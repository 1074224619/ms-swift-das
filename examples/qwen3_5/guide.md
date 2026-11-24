# ms-swift-das Qwen3 guide

## 创建容器
```bash

docker run -it \
    --shm-size=64G \
    --device=/dev/kfd \
    --device=/dev/mkfd \
    --device=/dev/dri \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --ulimit memlock=-1:-1 \
    --ipc=host \
    --network=host \
    --group-add video \
    --privileged \
    --name name \
    -v /opt/hyhal:/opt/hyhal:ro \
    -v /root/.ssh:/root/.ssh:ro \
    -v /workspace:/workspace \
    10.16.1.152:5000/jenkins/model_test_env/megatron:0.19.2-ubuntu22.04-dtk2610-py3.12-20261119-0654 \
    /bin/bash
```
## 下载模型

|模型|下载路径|
|--|--|
|qwen3.5-9b| `modelscope download --model Qwen/Qwen3.5-9B`|
|qwen3.5-27b| `modelscope download --model Qwen/Qwen3.5-27B`|
|qwen3.5-35ba3b| `modelscope download --model Qwen/Qwen3.5-35B-A3B`|

## 下载数据集并处理
--dataset 支持本地 jsonl / json / csv / txt / parquet 文件, 训练时在线 tokenize + packing. 

- 预训练:
    
    只要是类似`{"text": "一段预训练文本"}`格式的就行, 以`oscar-en-10k`为例:
    1. 下载数据集: `modelscope download --dataset kanliu/oscar-en-10k-megatron --local_dir ./oscar-en-10k`
    2. 直接将jsonl文本传给dataset即可 `--dataset oscar-en-10k/oscar-en-10k.jsonl`

    使用pt训练时会`use_chat_template=false`

- sft/lora:
    
    普通的sft格式即可, 以`Chinese-DeepSeek-R1-Distill-data-110k-SFT`为例: 
    1. 下载数据集: `modelscope download --dataset liucong/Chinese-DeepSeek-R1-Distill-data-110k-SFT --local_dir ./Chinese-DeepSeek-R1-Distill-data-110k-SFT`
    2. 直接将jsonl文本传给dataset即可 `--dataset Chinese-DeepSeek-R1-Distill-data-110k-SFT/distill_r1_110k_sft.jsonl`

- grpo/RL(暂无):

    以`DAPO-Math-17k`为例:
    1. 下载数据集: `modelscope download --dataset AI-ModelScope/DAPO-Math-17k  --local_dir ./DAPO-Math-17k`
    2. 转成支持的格式

## 配置节点环境

### 填写hostfile下的可用节点列表: 

编辑当前的hostfile文件, `vi hostfile`, 示例:
```
192.168.0.1
192.168.0.2
...
```

### 配置rcclenv.sh并测试rccltest是否正常 

编辑rcclenv.sh文件, `vi examples/tools/rcclenv.sh`

执行单机和多机rccltest
- 单机
    ```
    cd examples/tools
    bash test_rccl.sh single
    ```
- 多机
    ```
    cd examples/tools
    SSH_PORT=25901 bash test_rccl.sh multi path/to/hostfile
    ```

## 启动训练

通过run_train.sh启动训练, 参数设置通常为:
`MODEL=模型路径 DATASET=数据集路径 RCCLENV=rcclenv.sh路径 MODEL_KEY=训练脚本 bash run_train.sh hosfile路径 节点数 训练方案`

- 预训练:
    
    以qwen3 8b为例: `MODEL=path/to/Qwen3.5-9B DATASET=path/to/oscar-en-10k.jsonl RCCLENV=/path/to/rcclenv.sh MODEL_KEY=qwen3_5_8b bash run_train.sh hostfile 1 pretrain`

- sft:

    以qwen3 8b为例: `MODEL=path/to/Qwen3.5-9B DATASET=path/to/Chinese-DeepSeek-R1-Distill-data-110k-SFT/distill_r1_110k_sft.jsonl RCCLENV=/path/to/rcclenv.sh MODEL_KEY=qwen3_5_8b bash run_train.sh hostfile 1 sft`

- lora:

    以qwen3 30ba3b为例: `MODEL=path/to/Qwen3.5-35B-A3B DATASET=path/to/Chinese-DeepSeek-R1-Distill-data-110k-SFT/distill_r1_110k_sft.jsonl RCCLENV=/path/to/rcclenv.sh MODEL_KEY=qwen3_5_35ba3b bash run_train.sh hostfile 1 lora`

## 简单调参
ms-swift自有参数和megatron-lm参数相互搭配组合, 
- ms-swift在swift/megatron/arguments/megatron_args.py中定义参数
- 对于没有定义的, 可以通过megatron_extra_kwargs直接映射到transformerconfig

常见的参数设置方法如下所示:

- 切分配置相关参数
    ```python
    --tensor_model_parallel_size 1
    --sequence_parallel true
    --context_parallel_size 1
    --pipeline_model_parallel_size 1
    --expert_model_parallel_size 1
    --expert_tensor_parallel_size 1
    ```
- 显存优化相关参数
    ```python
    # offload
    # 用于sft/pretrain
    --optimizer_cpu_offload true 
    --optimizer_offload_fraction 1.0 
    --use_precision_aware_optimizer true
    # 用于grpo/gkd等 rollout 场景
    --offload_optimizer true
    --offload_mode true
    --offload_bridge true # 仅 GRPO/GKD：导出的 HF 权重放 CPU 主存
    --offload_teacher_model true # 仅 GKD：teacher 放 CPU

    # recompute
    --recompute_granularity full # selective 配合--recompute_modules, 默认 core_attn；另有 moe_act layernorm mla_up_proj mlp moe
    --recompute_method uniform # uniform / block 
    --recompute_num_layers 1 #需要被整除
    --mhc_recompute_layer_num 1 # 仅dsv4的mhc

    # 分布式ddp优化器
    --use_distributed_optimizer true

    ```
- 性能优化相关参数
    ```python
    --overlap_grad_reduce true 
    --attention_backend flash
    --moe_permute_fusion true 
    --moe_grouped_gemm true 
    --moe_shared_expert_overlap true 
    --moe_aux_loss_coeff 1e-6 
    --megatron_extra_kwargs '{"moe_router_fusion": true,}' # 无语义校验，绕过检查, 字段名必须在 mcore 里存在, 给 swift 没暴露的 mcore 字段开口子
    ```
- 数据集配置相关参数
    ```python
    --finetune #断点续训
    --max_length 8192 #最大的seq, 超出会截断
    --packing true
    --truncation_strategy #对于超长数据处理: delete(默认, 跳过数据), left留末尾丢开头, right留开头丢末尾, split(切成多个batch, 只能pretrain用)
    --streaming true # 流式, 配合train_iter使用
    --split_dataset_ratio 0.01 #未设置--val_dataset时从训练集中抽部分做eval
    --system # 给样本加上system prompt, 在样本没有system字段时补上.
    --add_non_thinking_prefix true #对于没有think数据集, 自动补上一个空的think的标识
    --loss_scale ignore_empty_think  #将空think块的loss权重设置为0. 可选参数:
        # - default: SFT, 只对 assistant 回复算 loss（含历史轮，权重 1）；system/user/多模态 token 不算	
        # - last_round: RLHF（DPO/GRPO 等）, 只对最后一轮 assistant 回复算 loss（最后一个 user 之后的内容）
        # - all: swift pt, 所有 token 都算 loss，包括 system/user
    ```
- 训练方案相关参数
    ```python
    --tuner_type full #可选 full(全参训练) / lora(冻结基座) / lora_llm(只对llm)
        # - lora时: --target_modules [all-linear]在指定层使用低秩矩阵, --lora_rank/--lora_alpha 输出adapter, --merge_lora 控制导出时是否合并
        # - lora_llm时: merge_lora=true

    --num_train_epochs # 训练集过几遍, 会覆盖train_iters
    --language_model_only true # 多模态模型只跑文本模型
    # 加载vit但是只训练纯文本: --language_model_only false, --freeze_vit true, --freeze_llm false, freeze_aligner true
    ```