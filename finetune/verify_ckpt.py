"""验证指定 checkpoint：多条文本生成，报告时长、EOS 是否触发、音频能量。"""
import argparse

import numpy as np
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

TEXTS = [
    "今天天气不错。",
    "我是椿，很高兴认识你。",
    "我们出去走走吧。",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--speaker", default="chun")
    ap.add_argument("--max_new", type=int, default=1500)
    ap.add_argument("--tag", default="v")
    args = ap.parse_args()

    tts = Qwen3TTSModel.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda:0",
        attn_implementation="sdpa",
    )
    print(f"== {args.model} ==\n")
    for i, text in enumerate(TEXTS):
        torch.manual_seed(i)
        input_ids = tts._tokenize_texts([tts._build_assistant_text(text)])
        kw = tts._merge_generate_kwargs(max_new_tokens=args.max_new)
        with torch.no_grad():
            codes, _ = tts.model.generate(
                input_ids=input_ids, instruct_ids=[None], languages=["Auto"],
                speakers=[args.speaker], non_streaming_mode=True, **kw,
            )
        n = codes[0].shape[0]
        hit_cap = n >= args.max_new - 1
        wavs, sr = tts.model.speech_tokenizer.decode([{"audio_codes": codes[0]}])
        w = wavs[0]
        dur = len(w) / sr
        rms = float(np.sqrt(np.mean(w.astype(np.float64) ** 2)))
        peak = float(np.abs(w).max())
        out = f"{args.tag}_{i}.wav"
        sf.write(out, w, sr)
        print(f"[{i}] n={n:5d} cap={hit_cap!s:5s} dur={dur:7.2f}s rms={rms:.4f} peak={peak:.3f} "
              f"-> {out}  | {text}")
        del codes, wavs
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
