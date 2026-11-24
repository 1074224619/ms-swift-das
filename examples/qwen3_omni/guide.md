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

### modify multisource yaml

Modify `configs/multimodal/data/tulu_sharegpt4v_llavavideo.yaml`:

```yaml
sources:
- sharegpt4v_instruct_gpt4-vision_cap100k_coco.json
- tulu-3-sft-mixture/data
- video.json
names:
- sharegpt4v_captioner_sft
- tulu-3-sft-mixture
- LLaVA-Video-178K
schedule:
- schedule_type: const
  weights: [0.4, 0.2, 0.4]
```

## Start training on GPU/NPU

```shell
bash train.sh tasks/train_vlm.py configs/multimodal/qwen3_omni/qwen3_omni.yaml \
    --model.model_path Qwen3-Omni-30B-A3B-Instruct
```

> **Note.** VeOmni's runtime `CheckpointTensorConverter` folds per-expert HF
> safetensor keys into VeOmni's fused `gate_up_proj` / `down_proj` layout at
> load time, so the stock HF checkpoint can be passed directly to training —
> no offline merge step is required. See
> `docs/transformers_v5/transformers_v5_moe_weight_loading.md` for the full
> format matrix and how to convert a VeOmni-format training checkpoint back
> to per-expert HF keys for inference engines.
>
> `scripts/moe_ckpt_merge/moe_merge.py` is deprecated. It still works and may
> be useful as a one-time optimization for very large checkpoints (e.g.
> Qwen3-235B) where you want to amortize the per-load stacking cost across
> many runs, but it is no longer a prerequisite.

## 在 ms-swift (Megatron-SWIFT) 中训练

VeOmni 处理好的数据可以直接给 ms-swift 使用：`--dataset` 支持本地 json /
parquet 目录，媒体相对路径通过 `ROOT_IMAGE_DIR` 解析。唯一需要额外处理的是
LLaVA-Video 的标注：它用 `<image>` 作为视频占位符，而 ms-swift 按媒体类型分发
占位符（`videos` 字段只替换 `<video>`），所以要先转换出 `video_swift.json`：

```bash
cd examples/qwen3_omni
python prepare_data.py --data_dir /public/home/wangxj3/Downloads/model_datasets/datasets/VeOmni
```

三个数据集对应 VeOmni 的 `tulu_sharegpt4v_llavavideo.yaml`：

| 数据 | 文件 | 默认采样 |
|---|---|---|
| ShareGPT4V caption | `sharegpt4v_instruct_gpt4-vision_cap100k_coco.json` | 10000 |
| tulu-3-sft-mixture | `tulu-3-sft-mixture/data`（parquet 目录） | 5000 |
| LLaVA-Video | `video_swift.json` | 5753（全量） |

采样条数可通过 `SHAREGPT4V_N` / `TULU_N` / `VIDEO_N` 调整。要严格复现 VeOmni
的 0.4 / 0.2 / 0.4 配比，可设置 `SHAREGPT4V_N=5753 TULU_N=2877`（视频全量）。

单机 8 卡 LoRA（推荐）：

```bash
cd examples/qwen3_omni
DATA_DIR=/public/home/wangxj3/Downloads/model_datasets/datasets/VeOmni \
    bash train_qwen3_omni_lora.sh 1 127.0.0.1 \
    /public/home/wangxj3/Downloads/model_datasets/models/Qwen3-Omni-30B-A3B-Instruct
```

多机（`hostfile` 每行一个节点 IP/主机名）：

```bash
MODEL=/path/to/Qwen3-Omni-30B-A3B-Instruct \
DATA_DIR=/path/to/VeOmni \
    bash run_train.sh hosts.txt 2 lora
```

全参数 SFT 把 `lora` 换成 `sft`；30B 全参显存占用较高，显存不足时请使用 LoRA。

关键参数说明：

- `--expert_model_parallel_size 8`：专家并行，单机 8 卡时专家均分到每张卡；
  多机可增大到 16/32（需能整除 world_size）。LoRA 不支持 ETP>1（默认 ETP=1）。
- `--freeze_vit true --freeze_aligner true`：冻结视觉/音频塔与 aligner，只训练 LLM。
- `ENABLE_AUDIO_OUTPUT=0`：只训练 thinker，不加载 talker / code2wav。
- `IMAGE_MAX_TOKEN_NUM` / `VIDEO_MAX_TOKEN_NUM` / `FPS_MAX_FRAMES`：控制图像/视频
  的 token 数与抽帧数；视频较长或显存紧张时调小。
- `--packing true --max_length 4096`：变长打包，超过 `max_length` 的样本默认丢弃
  （`truncation_strategy=delete`）。
- LoRA 目标模块 `all-linear` 已覆盖 MoE 专家（grouped GEMM），默认不含 router；
  需要训练 router 时改成 `--target_modules all-router all-linear`。
- 训练完成后合并权重：`megatron export --merge_lora true ...`。