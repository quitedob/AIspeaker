# Devlog：Qwen3-TTS 0.6B「椿」单说话人微调

> 目的：不搬运 2.4 GB 的 checkpoint，而是把**从零复现**的全部信息记录在这里。
> 新机器按本文跑一遍即可得到等价权重（全程约 12–20 分钟）。
>
> 记录时间：2026-11-13　｜　最后验证：2026-11-09
> 项目：<https://github.com/quitedob/AIspeaker>

---

## 1. 结论先行

| 项目 | 结果 |
|---|---|
| 基座模型 | `Qwen/Qwen3-TTS-12Hz-0.6B-Base` |
| 目标声线 | wwTTS「椿」中文女声（344 条，约 43 分钟） |
| 产物 | `tts_model_type: custom_voice`，`speaker="chun"` |
| 最终 checkpoint | `finetune/output/checkpoint-epoch-23`（2.4 GB，**不进 git**） |
| 训练配置 | batch 8、grad_accum 1、lr 1e-5、24 epoch、1030 次优化器更新 |
| 最终 loss | 总计 0.82（main 0.16 / sub 2.22） |
| 显存峰值 | allocated 9.31 GB / reserved 11.83 GB（RTX 4080 16 GB） |

推理入口：

```bash
python gen_test.py --model output/checkpoint-epoch-23 --speaker chun --text "今天天气不错。"
```

---

## 2. 环境

本机实测环境（换机器时按这个对齐，版本差异是最常见的坑）：

| 项 | 值 |
|---|---|
| OS | Windows 10 Home 19045 |
| GPU | NVIDIA RTX 4080 16 GB（sm_89） |
| 驱动 | 581.15 |
| Python | 3.12.0（venv） |
| torch | 2.11.0+cu130（CUDA 13.0） |
| transformers | 4.57.3 |
| qwen-tts | 0.1.1 |
| accelerate | 1.12.0 |
| librosa / soundfile | 0.11.0 / 0.13.1 |
| 注意力后端 | `sdpa`（**未装 flash-attn**，Windows 上建议就用 sdpa） |

安装：

```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```

训练日志里会出现两条**无害**警告，不用管：

- `flash-attn is not installed` —— 用的是 sdpa，脚本默认就是。
- `SoX could not be found` —— 来自 faster-qwen3-tts 的导入，训练不走 SoX。

Windows 上 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 实际不生效
（日志会打印 `expandable_segments not supported on this platform`），设了也无妨，
但别指望它省显存。

---

## 3. 数据准备

### 3.1 下载 wwTTS 原始数据

数据源：<https://www.modelscope.cn/datasets/Genius-Society/wwTTS>（CC-BY-NC-ND，仅限非商业研究）。

```bash
# 项目根目录，需要能访问 modelscope.cn 和 pan.acgnai.top
.venv\Scripts\python example\download_wwtts.py --character 椿 --lang zh
```

数据落在 `example/.modelscope/downloads/extracted/<hash>/`，每个 `.wav` 配一个同名 `.lab`。
注意下载目录的 `<hash>` 每次可能不同，`build_manifest.py` 里用的是 glob `850a*/`，
**如果 hash 变了要改这个 glob**（或者直接把 `850a*` 改成 `*`）。

`torchcodec` 解码音频依赖 FFmpeg 动态库，重装 venv 后如果报
`Could not load libtorchcodec_core*.dll`，把 `tools/ffmpeg/.../bin/av*.dll`、`sw*.dll`
拷到 `.venv\Lib\site-packages\torchcodec\`（详见 `example/README.md`）。

### 3.2 生成训练清单

```bash
cd finetune
python build_manifest.py
```

产出 `train_raw.jsonl`，344 条。规则：

- 按 `zh_*.wav` 收集
- 剔除男声（`_M.wav`，1 条）
- 剔除短于 1 秒或空文本（6 条）
- `ref_audio` 固定为 `zh_vo_Huanglong_main_1_6_21_48.wav`（「我是椿。」约 3.6 s）

> **全量使用同一条 ref_audio 能显著提升说话人一致性**，这是官方 README 也强调的点。

### 3.3 提取 12Hz audio_codes

```bash
python prepare_data.py \
  --device cuda:0 \
  --tokenizer_model_path Qwen/Qwen3-TTS-Tokenizer-12Hz \
  --input_jsonl train_raw.jsonl \
  --output_jsonl train_with_codes.jsonl
```

产出 `train_with_codes.jsonl`（344 条，约 3 MB）。首次运行会自动从 HuggingFace
下载 `Qwen3-TTS-Tokenizer-12Hz`；国内网络可先设 `HF_ENDPOINT=https://hf-mirror.com`。

> `prepare_data.py` 的 `open()` 必须带 `encoding="utf-8"`，否则 Windows 中文环境下
> 会报 `UnicodeDecodeError: 'gbk' codec`。仓库里的版本已经修好。

`train_raw.jsonl` / `train_with_codes.jsonl` 都在 `.gitignore` 里，**不会随仓库走**，
新机器必须按上面两步重新生成。

---

## 4. 训练

```bash
cd finetune
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # 可选，Windows 上无效

python sft_12hz.py ^
  --init_model_path Qwen/Qwen3-TTS-12Hz-0.6B-Base ^
  --output_model_path output ^
  --train_jsonl train_with_codes.jsonl ^
  --batch_size 8 --grad_accum 1 ^
  --lr 1e-5 --num_epochs 24 ^
  --save_every 4 --speaker_name chun
```

### 4.1 显存与速度

官方示例是 `batch_size 32`，16 GB 卡直接 OOM。实测（`FINETUNE_NOTES.md`）：

| batch | grad_accum | 有效 batch | 峰值显存 | 1 epoch |
|---|---|---|---|---|
| 1 | 8 | 8 | 8.94 GB | ~240 s |
| **8** | **1** | **8** | **9.28 GB** | **~30 s** |

优化效果等价，但 `batch_size 8` 前向次数少 8 倍，所以用这个。
默认开 gradient checkpointing（`--no_grad_checkpointing` 可关，用显存换速度）。

344 条 ÷ batch 8 = **43 次更新/epoch**，24 epoch = **1030 步**。

### 4.2 预期日志

```
Epoch 0  | Step 0  | OptStep 1   | Loss: 11.0379 | main 9.3163  sub 5.7386 | peak 9.00GB
Epoch 0  | Step 40 | OptStep 41  | Loss: 3.1755  | main 1.7797  sub 4.6527 | peak 9.28GB
Epoch 5  | ...     |             | Loss: ~2.4
Epoch 15 | ...     |             | Loss: ~1.2
Epoch 23 | Step 40 | OptStep 1030| Loss: 0.8231  | main 0.1575  sub 2.2186 | peak 9.31GB reserved 11.83GB
```

loss 三段式 = `main_loss + 0.3 × sub_talker_loss`，所以打印里同时给出两项。

### 4.3 checkpoint

`--save_every 4` → 只在 epoch 3/7/11/15/19/23 存盘，每个约 2.4 GB：

```
output/
├── checkpoint-epoch-3
├── checkpoint-epoch-7
├── checkpoint-epoch-11
├── checkpoint-epoch-15
├── checkpoint-epoch-19
└── checkpoint-epoch-23   ← 用这个
```

每个目录都完整自包含（`model.safetensors` 1.7 G + config + speech_tokenizer + vocab），
可以直接 `from_pretrained`。

存盘时 `save_checkpoint()` 做了三件事：

1. 复制基座目录，改 `config.json`：`tts_model_type = "custom_voice"`，
   `spk_id = {"chun": 3000}`，`spk_is_dialect = {"chun": false}`
2. 丢弃 `speaker_encoder.*` 权重（推理不需要）
3. 把第一帧的说话人 embedding 写进 `codec_embedding.weight[3000]`

---

## 5. 相对官方脚本的改动（关键，照抄会失败）

改动全部集中在 `finetune/` 下三个文件：`sft_12hz.py`、`dataset.py`、`prepare_data.py`。
原始脚本来自 <https://github.com/QwenLM/Qwen3-TTS/tree/main/finetuning>（Apache-2.0）。

### 5.1 0.6B 必须过 `text_projection` ★

0.6B 的 `text_hidden_size = 2048`，而 `hidden_size = 1024`，文本 embedding 必须先投影。
1.7B 两者相等、`text_projection` 是 `Identity`，官方脚本就省掉了这一步；
照搬到 0.6B 会报 `size of tensor a (2048) must match tensor b (1024)`。

```python
input_text_embedding = model.talker.text_projection(
    model.talker.get_text_embeddings()(input_text_ids)
) * text_embedding_mask
```

### 5.2 子 talker 的 loss 被 shift 了两次 ★★（最隐蔽的坑）

官方 `forward_sub_talker_finetune` 内部走 `ForCausalLMLoss`，函数里**又做了一次
`shift_labels`**。但此时 logits 已经按 codebook 逐位置对齐、labels 也已切成
`codec_ids[:, 1:]`，再 shift 就错位一格。

在基座模型上实测：

| 对齐方式 | loss | argmax 命中率 |
|---|---|---|
| 双重 shift（官方） | 12.72 | 0.02 % |
| 正确对齐 | 5.68 | 17 % |

**症状**：错位的子 talker 在推理时产出乱码 codebook，又反馈进主 talker，
导致 `codec_eos_token` 永远预测不出来 —— 生成会一路顶到 `max_new_tokens`
（8192 tokens ÷ 12.5 Hz ≈ 655 秒的连续音频）。

修复是 `sft_12hz.py::sub_talker_loss_aligned()`，直接对已对齐的 logits 算交叉熵，
不再二次 shift：

```python
out = talker.code_predictor.forward_finetune(
    inputs_embeds=sub_embeds, labels=codec_ids[:, 1:]
)
loss = F.cross_entropy(
    out.logits.reshape(-1, out.logits.shape[-1]).float(),
    codec_ids[:, 1:].reshape(-1),
    ignore_index=-100,
)
```

> 历史对照：`finetune/output_old_badsubt/` 就是修复前的跑法，loss 卡在 10 上下
> 降不下去，可以删掉，别拿来推理。`finetune/output_smoke/` 是单步冒烟测试，
> `finetune/output_overfit/` 是过拟合自检，都只是实验产物。

### 5.3 采样率

wwTTS 原始是 44.1 kHz，模型要求 24 kHz。`dataset.py::extract_mels()` 里用
librosa 重采样；官方脚本假设输入已经是 24 kHz。

### 5.4 Windows 编码

`prepare_data.py` 的 `open()` 必须带 `encoding="utf-8"`（见 3.3）。

### 5.5 新增训练参数

- `--max_steps`：跑够 N 次优化器更新就停，用于快速验证
- `--save_every`：每 N 个 epoch 存一次盘（每个 2.4 GB，控制磁盘占用）
- `--grad_accum` / `--no_grad_checkpointing` / `--attn`（sdpa / flash_attention_2）

---

## 6. 为什么必须训满 24 epoch

| 配置 | 结果 |
|---|---|
| 1 epoch（43 步） | **严重欠拟合**：loss 13.2 → 3.0，学不会 `codec_eos_token`，生成不终止 |
| overfit 自检（16 条 × 300 步） | loss 0.06，生成正常收尾（1.1 s / 3.7 s）→ 证明数据链路和补丁正确 |
| 正式 24 epoch（1030 步） | loss 0.82，EOS 正常触发 |

overfit 自检非常重要：它把「代码/数据 bug」和「训练量不够」两种失败区分开了。
以后换数据集，建议先跑一遍小样本过拟合，再上全量。

---

## 7. 验证与诊断脚本

训练完不用靠耳朵猜，`finetune/` 下这些脚本可以量化检查：

| 脚本 | 作用 |
|---|---|
| `gen_test.py` | 快速推理，看时长/是否正常收尾 |
| `verify_ckpt.py` | 多条文本生成，报告时长、EOS 是否触发、音频能量 |
| `sweep_epochs.py` | 扫各个 checkpoint 的生成长度与重复度（len < max_new_tokens 即 EOS 正常） |
| `asr_eval.py` | 用 SenseVoice 反听写，判断「说了预期的话」 |
| `test_indist.py` | 训练分布内/外测试，区分泛化不足 vs 推理路径 bug |
| `check_codes.py` | 验证 audio_codes 能否还原出可识别语音（闭环） |
| `diag_eos.py` | 前向训练样本，统计 EOS 位置的 argmax 命中率 |
| `diag_gen.py` | 捕获实际生成的 codec 序列，量化子 talker 精度 |
| `probe_*` / `ev_sub.py` | 反推子 talker 的 hidden/codec 配对关系（5.2 的依据） |

快速验收：

```bash
python verify_ckpt.py --model output/checkpoint-epoch-23 --speaker chun
python asr_eval.py --files output.wav
```

---

## 8. 推理

```python
import torch, soundfile as sf
from qwen_tts import Qwen3TTSModel

tts = Qwen3TTSModel.from_pretrained(
    "finetune/output/checkpoint-epoch-23",
    device_map="cuda:0",
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
)
wavs, sr = tts.generate_custom_voice(text="今天天气不错。", speaker="chun")
sf.write("out.wav", wavs[0], sr)
```

---

## 9. 迁移到新机器要准备什么

仓库里只有 41 个代码/文档文件（244 KB），**大文件全部在 `.gitignore` 里**：

| 路径 | 大小 | 处理方式 |
|---|---|---|
| `finetune/output/` | 14 GB | 不用拷，按本文重训约 12–20 分钟；只留 `epoch-23` 也可 |
| `finetune/*.jsonl` | ~3 MB | 按第 3 节重新生成（含本机绝对路径，必须重生成） |
| `example/.modelscope/` | 1.8 GB | 重新下载（`download_wwtts.py`） |
| `tools/ffmpeg/` | 245 MB | 需要 torchcodec 解码时再下；或只拷 DLL |
| `.venv/` | 4.1 GB | 重建（见第 2 节） |
| `~/.cache/huggingface/` | 2.4 GB | 首次运行自动下载基座 |
| `config.yaml` | 2 KB | **手工重建**，含 API key 和本机绝对路径 |

注意两个**路径耦合**：

1. `train_raw.jsonl` / `train_with_codes.jsonl` 里存的是 `D:/python/qwen3tts/...` 绝对路径，
   换机器必须重新生成，不能直接拷。
2. `config.yaml` 的 `tts.ref_audio` 指向一个可能不存在的本机路径。只要 `speaker.pt`
   存在就不会用到它；`speaker.pt` 已在 git 里（3.6 KB），所以默认没问题。

另外 `config.yaml` 里的 LLM key 是明文且不进 git。新机器建议直接用环境变量：

```bash
set MINIMAX_API_KEY=sk-xxxx      # 或 LLM_API_KEY
```

`voice_app/config.py` 会优先读环境变量覆盖 `config.yaml`。

---

## 10. 已知问题 / 待办

- [ ] 网页应用仍用基座模型 + `speaker.pt` 克隆，**没有接入微调的 epoch-23**。
      要接的话需改 `voice_app/tts_engine.py`（`generate_custom_voice` 路径）和 `config.yaml`。
- [ ] `.gitignore` 里 `finetune/train.log` 只匹配这一个文件名，
      `train_official.log` 被提交进仓库了（3 KB，无害，但想统一可改成 `finetune/*.log`）。
- [ ] `train.log` / `train_official.log` 是 GBK 编码（Windows 控制台重定向），
      Linux/macOS 下用 `iconv -f GBK -t UTF-8` 查看。
- [ ] `build_manifest.py` 的 `EXTRACT_GLOB = "850a*/"` 依赖下载 hash，换环境可能失效。
- [ ] 没有固定随机种子，重训结果不会逐比特一致，但听感应等价。
