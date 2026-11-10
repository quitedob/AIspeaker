"""用干净的交叉熵（无二次 shift）比较子 talker 的三种 hidden/codec 配对。

hidden[t] 预测 codec_0[t+1]；推理时子 talker 收到的正是「预测 codec_0 时」的 hidden，
配合同一帧的 codec_0。所以正确配对应当是 hidden[t] <=> codec[t+1]。
用基座模型验证：正确配对的 loss 必须显著低于其它。
"""
import argparse
import json

import torch
import torch.nn.functional as F
from qwen_tts import Qwen3TTSModel
from transformers import AutoConfig

from dataset import TTSDataset


def clean_ce(talker, codec, hidden):
    sub_embeds = [hidden.unsqueeze(1)]
    for i in range(talker.config.num_code_groups - 1):
        layer = (talker.get_input_embeddings() if i == 0
                 else talker.code_predictor.get_input_embeddings()[i - 1])
        sub_embeds.append(layer(codec[:, i:i + 1]))
    out = talker.code_predictor.forward_finetune(
        inputs_embeds=torch.cat(sub_embeds, dim=1), labels=None
    )
    lg = out.logits
    tgt = codec[:, 1:]
    loss = F.cross_entropy(lg.reshape(-1, lg.shape[-1]).float(), tgt.reshape(-1))
    acc = (lg.argmax(-1) == tgt).float().mean().item()
    return loss.item(), acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--train_jsonl", default="train_with_codes.jsonl")
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()

    tts = Qwen3TTSModel.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda:0",
        attn_implementation="sdpa")
    model = tts.model.eval()
    cfg = AutoConfig.from_pretrained(args.model)

    rows = [json.loads(l) for l in open(args.train_jsonl, encoding="utf-8").readlines()[: args.n]]
    ds = TTSDataset(rows, tts.processor, cfg)
    b = ds.collate_fn([ds[i] for i in range(len(rows))])
    dev, dt = model.device, model.dtype

    ids = b["input_ids"].to(dev); cid = b["codec_ids"].to(dev)
    tm = b["text_embedding_mask"].to(dev).to(dt); cme = b["codec_embedding_mask"].to(dev).to(dt)
    at = b["attention_mask"].to(dev); lb = b["codec_0_labels"].to(dev)
    cm = b["codec_mask"].to(dev); rm = b["ref_mels"].to(dev).to(dt)

    with torch.no_grad():
        spk = model.speaker_encoder(rm)
        te = model.talker.text_projection(model.talker.get_text_embeddings()(ids[:, :, 0])) * tm
        ce = model.talker.model.codec_embedding(ids[:, :, 1]) * cme
        ce[:, 6, :] = spk
        emb = te + ce
        for i in range(1, 16):
            emb = emb + model.talker.code_predictor.get_input_embeddings()[i - 1](cid[:, :, i]) * cm.unsqueeze(-1).to(dt)
        o = model.talker(inputs_embeds=emb[:, :-1, :], attention_mask=at[:, :-1],
                         labels=lb[:, 1:], output_hidden_states=True)
        hs = o.hidden_states[0][-1]        # 位置 t 预测 codec_0[t+1]

        # A: hidden[t] <=> codec[t]   官方
        print("A hidden[t] <=> codec[t]   ", *clean_ce(model.talker, cid[cm], hs[cm[:, :-1]]))
        # B: hidden[t] <=> codec[t+1]
        m_b = cm[:, :-1]
        print("B hidden[t] <=> codec[t+1] ", *clean_ce(model.talker, cid[:, 1:][m_b], hs[m_b]))
        # C: hidden[t-1] <=> codec[t]
        print("C hidden[t-1] <=> codec[t] ", *clean_ce(model.talker, cid[cm], hs[cm[:, 1:]]))
    print(f"\nmain talker loss {o.loss.item():.4f} | random CE {torch.log(torch.tensor(2048.)).item():.3f}")


if __name__ == "__main__":
    main()
