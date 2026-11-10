# Qwen3-TTS 0.6B 单说话人微调（Windows / 16GB 显卡）

把 wwTTS 的「椿」中文女声微调进 `Qwen3-TTS-12Hz-0.6B-Base`，产物是一个
`tts_model_type: custom_voice` 的模型，用 `generate_custom_voice(text=, speaker="chun")` 推理。

原始脚本改自 <https://github.com/QwenLM/Qwen3-TTS/tree/main/finetuning>（Apache-2.0）。
本文只记录本项目为 0.6B / Windows / 16GB 做的改动。

## 流程

```bash
cd finetune

# 1) 数据清单（可选，train_raw.jsonl 已生成）
python build_manifest.py

# 2) 提取 audio_codes（12Hz tokenizer）
python prepare_data.py \
  --tokenizer_model_path Qwen/Qwen3-TTS-Tokenizer-12Hz \
  --input_jsonl train_raw.jsonl --output_jsonl train_with_codes.jsonl

# 3) 训练
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python sft_12hz.py \
  --init_model_path <0.6B-Base 路径> \
  --output_model_path output \
  --train_jsonl train_with_codes.jsonl \
  --batch_size 8 --grad_accum 1 --lr 1e-5 --num_epochs 24 \
  --save_every 4 --speaker_name chun

# 4) 试听
python gen_test.py --model output/checkpoint-epoch-23 --speaker chun --text "今天天气不错。"
```

## 相对官方脚本的改动

### 0.6B 必须过 `text_projection`（`sft_12hz.py`）

0.6B 的 `text_hidden_size=2048` 而 `hidden_size=1024`，文本 embedding 必须先投影。
1.7B 两者相等、`text_projection` 是 `Identity`，所以官方脚本直接省掉了这一步；
照搬到 0.6B 会报 `size of tensor a (2048) must match tensor b (1024)`。

```python
input_text_embedding = model.talker.text_projection(
    model.talker.get_text_embeddings()(input_text_ids)
) * text_embedding_mask
```

### 显存（16GB 卡）

官方示例是 `batch_size 32`，在 16GB 上直接 OOM。实测配置：

| batch | grad_accum | 有效 batch | 峰值显存 | 1 epoch |
|---|---|---|---|---|
| 1 | 8 | 8 | 8.94 GB | ~240 s |
| **8** | **1** | **8** | **9.28 GB** | **~30 s** |

两者优化效果等价，但 `batch_size 8` 少 8 倍前向次数。默认开启 gradient
checkpointing（`--no_grad_checkpointing` 可关，显存换速度）。

### Windows 编码

`prepare_data.py` 的 `open()` 必须带 `encoding="utf-8"`，否则中文路径/文本下
`UnicodeDecodeError: 'gbk' codec`。

### 采样率

wwTTS 是 44.1 kHz，模型要求 24 kHz。`dataset.py::extract_mels` 里用 librosa 重采样；
官方脚本假设输入已经是 24 kHz。

## 数据

- 344 条中文女声，约 43 分钟（已剔除 1 条男声 `_M.wav`、6 条短于 1 秒）
- 参考音频固定为 `zh_vo_Huanglong_main_1_6_21_48.wav`（「我是椿。」3.56 s）
  —— 全量用同一条 `ref_audio` 能显著提升说话人一致性

## 训练时长与收敛

344 条 ÷ batch 8 = 43 次优化器更新/epoch。

- **1 epoch（43 步）严重欠拟合**：loss 从 13.2 只降到 3.0，模型学不会
  `codec_eos_token` 终止，生成会一直顶到 `max_new_tokens`（8192 tokens ÷ 12.5 Hz
  = 655 s 的连续音频）。
- **过拟合自检**：16 条样本跑 300 步，loss 降到 0.06，生成正常收尾（1.1 s / 3.7 s）。
  这证明了数据链路和 `text_projection` 补丁正确，问题只在训练量。
- 正式训练用 24 epoch ≈ 1030 步。

`--max_steps` 和 `--save_every` 是本项目新增的参数，用于快速验证和 disc 空间控制
（每个 checkpoint 约 2.4 GB）。
