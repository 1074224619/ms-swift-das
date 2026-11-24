# ms-swift-das

在 HCU 上基于 ms-swift 做模型训练（Megatron 后端，通过 mcore-bridge 接入）。

## 可用示例

| 模型 | 规模 | pretrain | sft | lora | 文档 |
|---|---|---|---|---|---|
| qwen3 | 8B / 32B / 30B-A3B | √ | √ | √ | [examples/qwen3/guide.md](examples/qwen3/guide.md) |
| qwen3_5 | 9B / 27B / 35B-A3B | √ | √ | √ | [examples/qwen3_5/guide.md](examples/qwen3_5/guide.md) |
| qwen3_omni | 30B-A3B | √ | √ | √ | [examples/qwen3_omni/guide.md](examples/qwen3_omni/guide.md) |
| glm | GLM-5.3-Flash | √ | √ | √ | 见脚本头注释 |
| deepseek | DeepSeek-V4-Flash | √ | √ | √ | 见脚本头注释 |
