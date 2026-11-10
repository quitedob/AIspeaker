"""验证 audio_codes 是否能还原出可识别的语音 —— 用 ASR 做闭环检测。"""
import argparse
import json
import sys

sys.path.insert(0, "..")

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default="train_with_codes.jsonl")
    ap.add_argument("--idx", nargs="+", type=int, default=[0, 50, 150])
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-TTS-Tokenizer-12Hz")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.jsonl, encoding="utf-8").readlines()]
    tok = Qwen3TTSTokenizer.from_pretrained(args.tokenizer, device_map="cuda:0")

    from voice_app.asr_engine import ASREngine
    asr = ASREngine(device="cuda:0")

    for i in args.idx:
        r = rows[i]
        codes = torch.tensor([r["audio_codes"]], dtype=torch.long)
        wavs, sr = tok.decode([{"audio_codes": codes[0]}])
        out = f"roundtrip_{i}.wav"
        sf.write(out, wavs[0], sr)
        text = asr.transcribe(out)
        print(f"\n[{i}] 原始文本: {r['text']}")
        print(f"     roundtrip {len(wavs[0])/sr:.2f}s -> {out}")
        print(f"     ASR 转写: {text}")
        print(f"     codes shape: {codes[0].shape}")


if __name__ == "__main__":
    main()
