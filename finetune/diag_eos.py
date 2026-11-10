"""诊断：在训练样本上做前向，检查 EOS 位置的预测概率。

区分主 talker loss 与 sub_talker loss，并统计 EOS 位置 argmax 命中率。
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
    ap.add_argument("--base_model", default=None,
                    help="custom_voice checkpoint 不含 speaker_encoder，需从 base 载入")
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

    speaker_encoder = model.speaker_encoder
    if speaker_encoder is None:
        base = Qwen3TTSModel.from_pretrained(
            args.base_model, torch_dtype=torch.bfloat16, device_map="cuda:0",
            attn_implementation="sdpa",
        )
        speaker_encoder = base.model.speaker_encoder
        speaker_encoder.eval()

    rows = [json.loads(l) for l in open(args.train_jsonl, encoding="utf-8").readlines()[: args.n]]
    ds = TTSDataset(rows, qwen.processor, cfg)
    batch = ds.collate_fn([ds[i] for i in range(len(rows))])

    dev = model.device
    dt = model.dtype
    input_ids = batch["input_ids"].to(dev)
    codec_ids = batch["codec_ids"].to(dev)
    text_mask = batch["text_embedding_mask"].to(dev).to(dt)
    codec_mask_e = batch["codec_embedding_mask"].to(dev).to(dt)
    attn = batch["attention_mask"].to(dev)
    labels = batch["codec_0_labels"].to(dev)
    codec_mask = batch["codec_mask"].to(dev)
    ref_mels = batch["ref_mels"].to(dev).to(dt)

    with torch.no_grad():
        spk = speaker_encoder(ref_mels)
        te = model.talker.text_projection(model.talker.get_text_embeddings()(input_ids[:, :, 0])) * text_mask
        ce = model.talker.model.codec_embedding(input_ids[:, :, 1]) * codec_mask_e
        ce[:, 6, :] = spk
        emb = te + ce
        for i in range(1, 16):
            e = model.talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i])
            emb = emb + e * batch["codec_mask"].to(dev).unsqueeze(-1).to(dt)

        out = model.talker(
            inputs_embeds=emb[:, :-1, :], attention_mask=attn[:, :-1],
            labels=labels[:, 1:], output_hidden_states=True,
        )
        hs = out.hidden_states[0][-1]
        sub_logits, sub_loss = model.talker.forward_sub_talker_finetune(
            codec_ids[codec_mask], hs[codec_mask[:, :-1]]
        )

    eos = cfg.talker_config.codec_eos_token_id
    pred = out.logits.argmax(-1)
    print(f"main talker loss : {out.loss.item():.4f}")
    print(f"sub talker  loss : {sub_loss.item():.4f}")
    print(f"combined         : {(out.loss + 0.3 * sub_loss).item():.4f}")

    # 每个样本的最后一个 label 就是 EOS
    labels_v = labels[:, 1:]
    hits = 0
    for b in range(labels_v.shape[0]):
        pos = (labels_v[b] != -100).nonzero(as_tuple=True)[0]
        if len(pos) == 0:
            continue
        last = pos[-1].item()
        tgt = labels_v[b, last].item()
        got = pred[b, last].item()
        prob = torch.softmax(out.logits[b, last].float(), -1)[tgt].item()
        ok = got == tgt
        hits += ok
        rank = (out.logits[b, last] > out.logits[b, last, tgt]).sum().item()
        print(f"  sample {b}: target={tgt} argmax={got} p={prob:.4f} rank={rank} {'OK' if ok else 'MISS'}")
    print(f"EOS argmax 命中率: {hits}/{labels_v.shape[0]}")


if __name__ == "__main__":
    main()
