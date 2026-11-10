"""用基座模型确定子 talker 的正确配对。

talker 在位置 t 的 hidden 预测 t+1 的 codec_0；推理时这个 hidden 与
刚生成的 codec_0 一起喂给子 talker 预测同一帧的 codebook 1..15。
所以 hidden[P] 应该配 codec[P+1]，而不是 codec[P]。
"""
import argparse
import json

import torch
from qwen_tts import Qwen3TTSModel
from transformers import AutoConfig

from dataset import TTSDataset


def sub_stats(model, hidden, codec, tag):
    _, loss = model.talker.forward_sub_talker_finetune(codec, hidden)
    logits, _ = model.talker.forward_sub_talker_finetune(codec, hidden)
    tgt = codec[:, 1:]
    acc = (logits.argmax(-1) == tgt).float().mean().item()
    print(f"{tag:34s} N={hidden.shape[0]:5d}  loss {loss.item():7.3f}  acc {acc:.4f}")
    return loss.item(), acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--train_jsonl", default="train_with_codes.jsonl")
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()

    qwen = Qwen3TTSModel.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda:0",
        attn_implementation="sdpa",
    )
    model = qwen.model.eval()
    cfg = AutoConfig.from_pretrained(args.model)

    rows = [json.loads(l) for l in open(args.train_jsonl, encoding="utf-8").readlines()[: args.n]]
    ds = TTSDataset(rows, qwen.processor, cfg)
    batch = ds.collate_fn([ds[i] for i in range(len(rows))])
    dev, dt = model.device, model.dtype

    input_ids = batch["input_ids"].to(dev)
    codec_ids = batch["codec_ids"].to(dev)
    tm = batch["text_embedding_mask"].to(dev).to(dt)
    ce_m = batch["codec_embedding_mask"].to(dev).to(dt)
    attn = batch["attention_mask"].to(dev)
    labels = batch["codec_0_labels"].to(dev)
    cm = batch["codec_mask"].to(dev)
    ref_mels = batch["ref_mels"].to(dev).to(dt)

    with torch.no_grad():
        spk = model.speaker_encoder(ref_mels)
        te = model.talker.text_projection(model.talker.get_text_embeddings()(input_ids[:, :, 0])) * tm
        ce = model.talker.model.codec_embedding(input_ids[:, :, 1]) * ce_m
        ce[:, 6, :] = spk
        emb = te + ce
        for i in range(1, 16):
            e = model.talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i])
            emb = emb + e * cm.unsqueeze(-1).to(dt)
        out = model.talker(
            inputs_embeds=emb[:, :-1, :], attention_mask=attn[:, :-1],
            labels=labels[:, 1:], output_hidden_states=True,
        )
    hs = out.hidden_states[0][-1]          # (B, L-1, H)，索引 t 对应位置 t
    print(f"main talker loss: {out.loss.item():.4f}\n")
    print(f"random CE over 2048 = {torch.log(torch.tensor(2048.)).item():.3f}\n")

    # A. 官方写法：hidden[t] 配 codec[t]
    sub_stats(model, hs[cm[:, :-1]], codec_ids[cm], "A hidden[P]   <=> codec[P]  (官方)")

    # B. hidden[t] 配 codec[t+1]
    sub_stats(model, hs[cm[:, :-1]], codec_ids[:, 1:][cm[:, :-1]], "B hidden[P]   <=> codec[P+1]")

    # C. hidden[t-1] 配 codec[t]
    sub_stats(model, hs[cm[:, 1:]], codec_ids[cm[:, 1:]], "C hidden[P-1] <=> codec[P]")


if __name__ == "__main__":
    main()
