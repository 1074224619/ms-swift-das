# ms-swift-das Qwen3-Omni guide

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
|Qwen3-Omni-30B-A3B| `modelscope download --model Qwen/Qwen3-Omni-30B-A3B-Instruct`|

本机已下载到 `/public/home/wangxj3/Downloads/model_datasets/models/Qwen3-Omni-30B-A3B-Instruct`，
下面 ms-swift 的训练脚本默认使用该路径（可用 `MODEL` 环境变量覆盖）。

## 下载数据集并处理
qwen3 omni属于全模态, 输入包含图像, 视频, 音频, 文本, 输出为文本.
数据集处理流程参考: https://veomni.readthedocs.io/en/latest/examples/qwen3_omni_moe.html

### sharegpt4v_cap_100k + COCO2017
sharegpt4v_cap_100k + COCO2017 是一个图像数据集, sharegpt4v_cap_100k中的sharegpt4v_instruct_gpt4-vision_cap100k.json就是对coco数据集的描述
```bash
mkdir VeOmni
cd VeOmni
# 下载数据集
modelscope download --dataset anyforge/coco-2017  --local_dir ./coco-2017
mv coco-2017 coco # 需要改一下目录名字
modelscope download --dataset AI-ModelScope/ShareGPT4V --local_dir ./ShareGPT4V
```
解压coco后与`sharegpt4v_instruct_gpt4-vision_cap100k.json`配对, 生成`sharegpt4v_instruct_gpt4-vision_cap100k_coco.json`.
```python
import json
with open('sharegpt4v_instruct_gpt4-vision_cap100k.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
filtered_data = []
for item in data:
    if item.get('image', '').startswith('coco'):
        new_item = item.copy()
        image_path = new_item.pop('image')
        new_item['images'] = [image_path]
        filtered_data.append(new_item)
with open('sharegpt4v_instruct_gpt4-vision_cap100k_coco.json', 'w', encoding='utf-8') as f:
    json.dump(filtered_data, f, ensure_ascii=False, indent=4)
```

处理完成后目录结构如下所示:
> ```
> VeOmni
> ├—— sharegpt4v_instruct_gpt4-vision_cap100k.json
> ├—— sharegpt4v_instruct_gpt4-vision_cap100k_coco.json
> ├—— coco/
> ├   └—— train2017/
> ├       ├—— 000000000009.jpg
> ├       ├—— 000000000026.jpg
> ├       └—— ... (more images)
> └—— ...(code files)
> ```

### tulu-3-sft-mixture
tulu-3-sft-mixture 是一个  数据集, parquet格式可以直接被swift加载
```bash
modelscope download --dataset allenai/tulu-3-sft-mixture --local_dir ./tulu-3-sft-mixture
```

目录结构如下所示:
> ```
> VeOmni
> ├—— tulu-3-sft-mixture/
> ├   └—— data/
> ├       ├—— train-00000-of-00006.parquet
> ├       ├—— train-00001-of-00006.parquet
> ├       ├—— train-00002-of-00006.parquet
> ├       ├—— train-00003-of-00006.parquet
> ├       ├—— train-00004-of-00006.parquet
> ├       └—— train-00005-of-00006.parquet
> └—— ...(code files)
> ```

### LLaVA-Video-178K
LLaVA-Video-178K 是一个视频数据集, 我们只需要其中的 LLaVA-Video-178K/0_30_s_academic_v0_1 一部分即可, 需要转成json格式
```bash
modelscope download --dataset lmms-lab/LLaVA-Video-178K   --include '0_30_s_academic_v0_1/*'   --local_dir ./LLaVA-Video-178
```
解压所有的 tar.gz 文件 并将解压出来的文件放到 VeOmni 目录下.
并将`0_30_s_academic_mc_v0_1_qa_processed.json` 处理成 `video.json`.

```python
import json
with open('0_30_s_academic_mc_v0_1_qa_processed.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
new_data = []
for item in data:
    new_item = item.copy()
    image_path = new_item.pop('video')
    new_item['videos'] = [image_path]
    new_data.append(new_item)  
with open('video.json', 'w', encoding='utf-8') as f:
    json.dump(new_data, f, ensure_ascii=False, indent=4)
```

目录结构如下所示:
> ```
> VeOmni
> ├—— 0_30_s_academic_mc_v0_1_qa_processed.json
> ├—— video.json
> ├—— academic_source/
> ├   ├—— activitynet/
> ├   ├   └—— ...
> ├   ├—— Charades/
> ├   ├   └—— ...
> ├   ├—— ego4d/
> ├   ├   └—— ...
> ├   ├—— NextQA/
> ├   ├   └—— ...
> ├   └—— youcook2/
> ├       └—— ...
> └—— ...(code files)
> ```

VeOmni 处理好的数据可以直接给 ms-swift 使用：`--dataset` 支持本地 json / parquet 目录，媒体相对路径通过 `ROOT_IMAGE_DIR` 解析。唯一需要额外处理的是LLaVA-Video 的标注：它用 `<image>` 作为视频占位符，而 ms-swift 按媒体类型分发占位符（`videos` 字段只替换 `<video>`），所以要先转换出 `video_swift.json`：

```bash
cd examples/qwen3_omni
python prepare_data.py --data_dir path/to/VeOmni
```

三个数据集对应 VeOmni 的设置：
| 数据 | 文件 | 默认采样 |
|---|---|---|
| ShareGPT4V caption | `sharegpt4v_instruct_gpt4-vision_cap100k_coco.json` | 10000 |
| tulu-3-sft-mixture | `tulu-3-sft-mixture/data`（parquet 目录） | 5000 |
| LLaVA-Video | `video_swift.json` | 5753（全量） |

采样条数可通过 `SHAREGPT4V_N` / `TULU_N` / `VIDEO_N` 调整。默认为 VeOmni的 0.4 / 0.2 / 0.4 配比，可设置 `SHAREGPT4V_N=10000 TULU_N=5000 VIDEO_N=5753`。

## 启动训练
镜像中可能会缺些依赖, 手动装上就行
```bash
pip install qwen_omni_utils
pip install audioread
pip install decord   # 视频解码；缺了它视频样本会被丢弃（容器里的 torchvision 没编译 read_video）
```

通过run_train.sh启动训练, 参数设置通常为:
`MODEL=模型路径 DATASET=数据集路径 RCCLENV=rcclenv.sh路径 MODEL_KEY=训练脚本 bash run_train.sh hosfile路径 节点数 训练方案`

- LoRA：
    ```bash
    MODEL=/path/to/Qwen3-Omni-30B-A3B-Instruct DATA_DIR=/path/to/VeOmni RCCLENV=/path/to/rcclenv.sh bash run_train.sh hostfile 1 lora
    ```

MODEL=/public/home/wangxj3/Downloads/model_datasets/models/Qwen3-Omni-30B-A3B-Instruct DATA_DIR=/public/home/wangxj3/Downloads/model_datasets/datasets/VeOmni RCCLENV=/public/home/wangxj3/Projects/ms-swift-das/examples/tools/rcclenv.sh bash run_train.sh hostfile 1 lora

日志中告警信息解释:
- 零星出现 `module 'torchvision.io' has no attribute 'read_video'`：`decord` 对个别损坏/编码不支持的视频解码失败（`DECORDError: avcodec_send_packet ...`）后回退到 torchvision 的报错，该样本会按数据集错误被丢弃（VeOmni 全量约 1% 的视频），属于坏样本，不是环境缺少依赖。

- `[transformers] Qwen2VL video processing does not apply the per-frame pixel cap ...` 是transformers 的提示：ms-swift 读取视频时已按参考实现（qwen-vl-utils 的 `VIDEO_MAX_PIXELS`）对每帧做过 resize，这里不再 cap 也不会多耗 token，可忽略。


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
    --dataloader_pin_memory false # 数据加载内存锁定, 开启后可能导致内存oom
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