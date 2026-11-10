"""用基座模型反推子 talker 的正确输入配对。

基座模型是预训练的，正确的 (hidden, codec) 配对应当给出远低于随机
(log 2048 = 7.625) 的 loss。逐一枚举候选。
"""
import argparse
import json

import torch
from qwen_tts import Qwen3TTSModel
from transformers import AutoConfig

from dataset import TTSDataset


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
    model = qwen.model
    model.eval()
    cfg = AutoConfig.from_pretrained(args.model)

    rows = [json.loads(l) for l in open(args.train_jsonl, encoding="utf-8").readlines()[: args.n]]
    ds = TTSDataset(rows, qwen.processor, cfg)
    batch = ds.collate_fn([ds[i] for i in range(len(rows))])
    dev, dt = model.device, model.dtype

    input_ids = batch["input_ids"].to(dev)
    codec_ids = batch["codec_ids"].to(dev)
    text_mask = batch["text_embedding_mask"].to(dev).to(dt)
    codec_emask = batch["codec_embedding_mask"].to(dev).to(dt)
    attn = batch["attention_mask"].to(dev)
    labels = batch["codec_0_labels"].to(dev)
    codec_mask = batch["codec_mask"].to(dev)
    ref_mels = batch["ref_mels"].to(dev).to(dt)

    with torch.no_grad():
        spk = model.speaker_encoder(ref_mels)
        te = model.talker.text_projection(model.talker.get_text_embeddings()(input_ids[:, :, 0])) * text_mask
        ce = model.talker.model.codec_embedding(input_ids[:, :, 1]) * codec_emask
        ce[:, 6, :] = spk
        emb = te + ce
        for i in range(1, 16):
            e = model.talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i])
            emb = emb + e * codec_mask.unsqueeze(-1).to(dt)

        out = model.talker(
            inputs_embeds=emb[:, :-1, :], attention_mask=attn[:, :-1],
            labels=labels[:, 1:], output_hidden_states=True,
        )

        print("hidden_states type:", type(out.hidden_states))
        print("len(hidden_states):", len(out.hidden_states))
        h0 = out.hidden_states[0]
        print("hidden_states[0] type:", type(h0))
        if isinstance(h0, (tuple, list)):
            print("  len(hidden_states[0]):", len(h0), "shapes:", [tuple(t.shape) for t in h0[:3]], "...")
            print("  hidden_states[0][-1] shape:", tuple(h0[-1].shape))
            print("  hidden_states[0][0] shape:", tuple(h0[0].shape))
        else:
            print("  shape:", tuple(h0.shape))

        hs_first = out.hidden_states[0][-1]   # 官方脚本用的
        hs_last = out.hidden_states[-1][-1] if isinstance(out.hidden_states[-1], (tuple, list)) else out.hidden_states[-1]

        # 枚举：hidden 来源 x 配对方式
        cm_prev = codec_mask[:, :-1]
        tgt_all = codec_ids[codec_mask]          # (N, 16) 位于 codec 位置 P
        print(f"\ncodec positions: {tgt_all.shape[0]}")

        # 方式 A: H[P] 配 codec[P]（官方脚本）
        # 方式 B: H[P-1] 配 codec[P]
        cm_shift = torch.zeros_like(codec_mask)
        cm_shift[:, 1:] = codec_mask[:, :-1]     # 位置 P 处标记 P-1 是 codec
        # 只保留 P 本身是 codec 的位置
        cm_shift = cm_shift & codec_mask

        for name, hs in (("layer[-1]", hs_first), ("layer[0]", out.hidden_states[0][0])):
            for pair, hmask in (("A H[P]<=>codec[P]", cm_prev), ("B H[P-1]<=>codec[P]", cm_shift)):
                hsel = hs[hmask]
                csel = codec_ids[codec_mask][: hsel.shape[0]] if hmask is not cm_prev else codec_ids[codec_mask]
                if hsel.shape[0] != csel.shape[0]:
                    print(f"{name:10s} {pair}: count mismatch {hsel.shape[0]} vs {csel.shape[0]}")
                    continue
                _, l = model.talker.forward_sub_talker_finetune(csel, hsel)
                lg, _ = model.talker.forward_sub_talker_finetune(csel, hsel)
                t = csel[:, 1:]
                acc = (lg.argmax(-1) == t).float().mean().item()
                print(f"{name:10s} {pair}: loss {l.item():.3f}  acc {acc:.4f}  N={hsel.shape[0]}")

        # 参考：主 talker loss
        print(f"\nmain talker loss: {out.loss.item():.4f}  (random over {cfg.talker_config.vocab_size})")


if __name__ == "__main__":
    main()
