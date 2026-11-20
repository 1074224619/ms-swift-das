# ms-swift-das

1. 创建容器


2. 下载模型


2. 处理数据集

预训练:
--dataset 支持本地 jsonl / json / csv / txt / parquet 文件, 训练时在线 tokenize + packing. 
只要是类似`{"text": "一段预训练文本"}`格式的就行
以`oscar-en-10k`为例:
1. 下载数据集: modelscope download --dataset kanliu/oscar-en-10k-megatron --local_dir ./oscar-en-10k
2. 直接将jsonl文本传给dataset即可 --dataset oscar-en-10k/oscar-en-10k.jsonl

sft:
--data
1. 下载数据集: modelscope download --dataset liucong/Chinese-DeepSeek-R1-Distill-data-110k-SFT --local_dir ./Chinese-DeepSeek-R1-Distill-data-110k-SFT
2. 直接将jsonl文本传给dataset即可 --dataset Chinese-DeepSeek-R1-Distill-data-110k-SFT/distill_r1_110k_sft.jsonl

grpo/RL
以`DAPO-Math-17k`为例:

1. 下载数据集: modelscope download --dataset AI-ModelScope/DAPO-Math-17k  --local_dir ./DAPO-Math-17k
2. 转成支持的格式

- pretrain: 使用pt训练时会use_chat_template=false

3. 配置节点环境

5. 启动训练
超参数配置:
--streaming true # 流式, 配合train_iter使用, 
--finetune 断点续训
--max_length 最大的seq, 超出会截断
--packing true
--truncation_strategy 超长数据支持: delete(默认, 跳过数据), left留末尾丢开头, right留开头丢末尾, split(切成多个batch, 只能pretrain用)
--split_dataset_ratio 0.01 (没--val_dataset从训练集中抽部分做eval)

--tuner_type 支持 full(全参训练) / lora(冻结基座) / lora_llm(只对llm)
    - lora时: --target_modules [all-linear]在指定层使用低秩矩阵, --lora_rank/--lora_alpha 输出adapter, --merge_lora 控制导出时是否合并
    - lora_llm时: merge_lora=true
--system 给样本加上system prompt, 在样本没有system字段时补上.
--model_author / --model_name 让模型知道叫什么, 谁开发的因为self-cognition数据集产生的字段
--load_from_cache_file true 从缓存加载ckpt生效
--num_train_epochs 训练集过几遍, 会覆盖train_iters


预训练:
MODEL=/public/opendas/DL_DATA/llm-models/qwen3/Qwen3-8B DATASET=/public/home/wangxj3/Downloads/model_datasets/datasets/oscar-en-10k/oscar-en-10k.jsonl RCCLENV=/public/home/wangxj3/Projects/ms-swift-das/examples/env.sh MODEL_KEY=qwen3_8b bash run_train.sh hostfile 1 pretrain

sft:
MODEL=/public/opendas/DL_DATA/llm-models/qwen3/Qwen3-8B DATASET=/public/home/wangxj3/Downloads/model_datasets/datasets/Chinese-DeepSeek-R1-Distill-data-110k-SFT/distill_r1_110k_sft.jsonl RCCLENV=/public/home/wangxj3/Projects/ms-swift-das/examples/env.sh MODEL_KEY=qwen3_8b bash run_train.sh hostfile 1 sft

lora:
MODEL=/public/opendas/DL_DATA/llm-models/qwen3/Qwen3-8B DATASET=/public/home/wangxj3/Downloads/model_datasets/datasets/Chinese-DeepSeek-R1-Distill-data-110k-SFT/distill_r1_110k_sft.jsonl RCCLENV=/public/home/wangxj3/Projects/ms-swift-das/examples/env.sh MODEL_KEY=qwen3_8b bash run_train.sh hostfile 1 lora

