"""在训练集分布内/外测试生成，判断是泛化不足还是推理路径 bug。"""
import argparse
import json

import numpy as np
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel


def analyze(w, sr):
    seg = w[: min(len(w), 5 * sr)]
    hop = int(0.05 * sr)
    fr = np.array([np.sqrt(np.mean(seg[i:i + hop] ** 2))
                   for i in range(0, max(len(seg) - hop, 1), hop)])
    zcr = float(np.mean(np.abs(np.diff(np.sign(seg))) / 2)) if len(seg) > 1 else 0.0
    return dict(dur=len(w) / sr, rms=float(np.sqrt(np.mean(w ** 2))),
                zcr=zcr, fr_std=float(fr.std()), clip=float(np.mean(np.abs(w) > 0.999)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--speaker", default="chun")
    ap.add_argument("--max_new", type=int, default=900)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--tag", default="t")
    args = ap.parse_args()

    tts = Qwen3TTSModel.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda:0",
        attn_implementation="sdpa",
    )

    train = [json.loads(l) for l in open("train_raw.jsonl", encoding="utf-8").readlines()]
    seen = [train[0]["text"], train[100]["text"], train[200]["text"]]
    novel = ["今天天气不错。", "我们出去走走吧。"]

    for kind, texts in (("IN-DIST", seen), ("NOVEL", novel)):
        for i, text in enumerate(texts):
            for seed in args.seeds:
                torch.manual_seed(seed)
                ids = tts._tokenize_texts([tts._build_assistant_text(text)])
                kw = tts._merge_generate_kwargs(max_new_tokens=args.max_new)
                with torch.no_grad():
                    codes, _ = tts.model.generate(
                        input_ids=ids, instruct_ids=[None], languages=["Auto"],
                        speakers=[args.speaker], non_streaming_mode=True, **kw,
                    )
                n = codes[0].shape[0]
                capped = n >= args.max_new - 1
                wavs, sr = tts.model.speech_tokenizer.decode([{"audio_codes": codes[0]}])
                a = analyze(wavs[0], sr)
                print(f"[{kind}] {text[:14]:16s} seed{seed} n={n:4d} capped={capped!s:5s} "
                      f"dur={a['dur']:7.2f}s zcr={a['zcr']:.3f} rms={a['rms']:.3f} "
                      f"fr_std={a['fr_std']:.3f} clip={a['clip']:.3f}")
                del codes, wavs
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
