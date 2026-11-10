"""用 SenseVoice ASR 客观评估生成音频：能否转写出预期文本。"""
import argparse
import sys

sys.path.insert(0, "..")

import soundfile as sf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--expect", default="")
    ap.add_argument("--max_sec", type=float, default=30.0,
                    help="只转写前 N 秒，避免 600 秒音频跑太久")
    args = ap.parse_args()

    from voice_app.asr_engine import ASREngine
    asr = ASREngine(device="cuda:0")

    for f in args.files:
        w, sr = sf.read(f)
        if len(w) / sr > args.max_sec:
            w = w[: int(args.max_sec * sr)]
            tmp = f.rsplit(".", 1)[0] + "_clip.wav"
            sf.write(tmp, w, sr)
            path = tmp
        else:
            path = f
        text = asr.transcribe(path)
        hit = "HIT" if args.expect and args.expect in text else ""
        print(f"[{f}] {hit}\n    ASR: {text!r}")


if __name__ == "__main__":
    main()
