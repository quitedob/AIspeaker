"""快速推理测试：检查生成的音频时长和内容是否正常。"""
import argparse

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--speaker", default="chun")
    ap.add_argument("--text", default="今天天气不错。")
    ap.add_argument("--out", default="gen_out.wav")
    ap.add_argument("--max_new_tokens", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    tts = Qwen3TTSModel.from_pretrained(
        args.model, device_map="cuda:0", dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    wavs, sr = tts.generate_custom_voice(
        text=args.text, speaker=args.speaker, max_new_tokens=args.max_new_tokens,
    )
    sf.write(args.out, wavs[0], sr)
    dur = len(wavs[0]) / sr
    print(f"[gen] {dur:.2f}s @ {sr}Hz -> {args.out}  (text: {args.text!r})")


if __name__ == "__main__":
    main()
