"""诊断推理过程：捕获实际生成的 codec 序列，并对子 talker 精度做量化。"""
import argparse

import torch
from qwen_tts import Qwen3TTSModel
from transformers import AutoConfig

from dataset import TTSDataset


def sub_talker_accuracy(qwen, cfg, jsonl, n=8):
    import json

    model = qwen.model
    rows = [json.loads(l) for l in open(jsonl, encoding="utf-8").readlines()[:n]]
    ds = TTSDataset(rows, qwen.processor, cfg)
    batch = ds.collate_fn([ds[i] for i in range(len(rows))])
    dev, dt = model.device, model.dtype

    input_ids = batch["input_ids"].to(dev)
    codec_ids = batch["codec_ids"].to(dev)
    masks = {
        "text_embedding_mask": batch["text_embedding_mask"].to(dev).to(dt),
        "codec_embedding_mask": batch["codec_embedding_mask"].to(dev).to(dt),
        "attention_mask": batch["attention_mask"].to(dev),
        "codec_mask": batch["codec_mask"].to(dev),
        "codec_0_labels": batch["codec_0_labels"].to(dev),
        "ref_mels": batch["ref_mels"].to(dev).to(dt),
    }
    with torch.no_grad():
        spk = model.speaker_encoder(masks["ref_mels"])
        te = model.talker.text_projection(model.talker.get_text_embeddings()(input_ids[:, :, 0])) * masks["text_embedding_mask"]
        ce = model.talker.model.codec_embedding(input_ids[:, :, 1]) * masks["codec_embedding_mask"]
        ce[:, 6, :] = spk
        emb = te + ce
        for i in range(1, 16):
            e = model.talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i])
            emb = emb + e * masks["codec_mask"].unsqueeze(-1).to(dt)
        out = model.talker(
            inputs_embeds=emb[:, :-1, :], attention_mask=masks["attention_mask"][:, :-1],
            labels=masks["codec_0_labels"][:, 1:], output_hidden_states=True,
        )
        hs = out.hidden_states[0][-1]
        cm = masks["codec_mask"]
        logits, loss = model.talker.forward_sub_talker_finetune(
            codec_ids[cm], hs[cm[:, :-1]]
        )
    tgt = codec_ids[cm][:, 1:]
    pred = logits.argmax(-1)
    print(f"sub-talker loss {loss.item():.3f}  (random = {torch.log(torch.tensor(2048.)).item():.3f})")
    accs = []
    for j in range(15):
        a = (pred[:, j] == tgt[:, j]).float().mean().item()
        accs.append(a)
    print("per-codebook acc:", " ".join(f"{a:.2f}" for a in accs))
    print(f"mean acc: {sum(accs)/len(accs):.3f}  (random = {1/2048:.5f})")


def inspect_generation(qwen, text, speaker, max_new_tokens=300):
    texts = [text]
    langs = ["Auto"]
    spks = [speaker]
    input_ids = qwen._tokenize_texts([qwen._build_assistant_text(t) for t in texts])
    gen_kwargs = qwen._merge_generate_kwargs(max_new_tokens=max_new_tokens)
    with torch.no_grad():
        codes_list, _ = qwen.model.generate(
            input_ids=input_ids, instruct_ids=[None], languages=langs,
            speakers=spks, non_streaming_mode=True, **gen_kwargs,
        )
    c = codes_list[0]
    eos = qwen.model.config.talker_config.codec_eos_token_id
    first = c[:, 0].tolist()
    print(f"\n[gen] {len(first)} tokens, EOS({eos}) present: {eos in first}")
    print(f"first 40 tokens: {first[:40]}")
    print(f"unique first-codebook values: {len(set(first))}")
    if eos in first:
        print(f"EOS at index {first.index(eos)}")
    else:
        print(f"last 10: {first[-10:]}")
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base_model", default=None)
    ap.add_argument("--train_jsonl", default="train_with_codes.jsonl")
    ap.add_argument("--text", default="今天天气不错。")
    ap.add_argument("--speaker", default="chun")
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()

    qwen = Qwen3TTSModel.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda:0",
        attn_implementation="sdpa",
    )
    qwen.model.eval()
    cfg = AutoConfig.from_pretrained(args.model)

    if qwen.model.speaker_encoder is None and args.base_model:
        base = Qwen3TTSModel.from_pretrained(
            args.base_model, torch_dtype=torch.bfloat16, device_map="cuda:0",
            attn_implementation="sdpa",
        )
        qwen.model.speaker_encoder = base.model.speaker_encoder

    sub_talker_accuracy(qwen, cfg, args.train_jsonl, args.n)
    inspect_generation(qwen, args.text, args.speaker)


if __name__ == "__main__":
    main()
