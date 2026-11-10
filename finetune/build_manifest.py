"""从 wwTTS 椿 中文数据生成训练清单 train_raw.jsonl。"""
import glob
import json
import os

import soundfile as sf

EXTRACT_GLOB = "example/.modelscope/downloads/extracted/850a*/"
REF_NAME = "zh_vo_Huanglong_main_1_6_21_48.wav"  # 「我是椿。」4 秒，干净
OUT = "finetune/train_raw.jsonl"
MIN_DUR = 1.0


def win(p: str) -> str:
    return os.path.abspath(p).replace(os.sep, "/")


def main() -> None:
    d = glob.glob(EXTRACT_GLOB)[0]
    ref = os.path.join(d, REF_NAME)
    ref_text = open(ref[:-4] + ".lab", encoding="utf-8").read().strip()
    print(f"参考音频: {REF_NAME} {sf.info(ref).duration:.2f}s | {ref_text}")

    rows = []
    skipped_male = skipped_short = 0
    for f in sorted(glob.glob(d + "zh_*.wav")):
        text = open(f[:-4] + ".lab", encoding="utf-8").read().strip()
        dur = sf.info(f).duration
        if f.endswith("_M.wav"):
            skipped_male += 1
            continue
        if dur < MIN_DUR or not text:
            skipped_short += 1
            continue
        rows.append({"audio": win(f), "text": text, "ref_audio": win(ref)})

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    total_min = sum(sf.info(r["audio"]).duration for r in rows) / 60
    print(f"写入 {len(rows)} 条 -> {OUT}")
    print(f"跳过: 男声 {skipped_male}, 过短/空文本 {skipped_short}")
    print(f"总时长 {total_min:.1f} 分钟")


if __name__ == "__main__":
    main()
