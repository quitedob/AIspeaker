"""对比不同 checkpoint 的生成行为：贪心 vs 采样，捕获第一个 codebook 序列。"""
import argparse

import torch
from qwen_tts import Qwen3TTSModel


def probe(model_path, text, speaker, max_new=240):
    tts = Qwen3TTSModel.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="cuda:0",
        attn_implementation="sdpa",
    )
    eos = tts.model.config.talker_config.codec_eos_token_id
    input_ids = tts._tokenize_texts([tts._build_assistant_text(text)])
    gen_kwargs = tts._merge_generate_kwargs(max_new_tokens=max_new)

    for mode, kw in (
        ("greedy", dict(do_sample=False, subtalker_dosample=False)),
        ("sample", {}),
    ):
        k = dict(gen_kwargs)
        k.update(kw)
        with torch.no_grad():
            codes, _ = tts.model.generate(
                input_ids=input_ids, instruct_ids=[None], languages=["Auto"],
                speakers=[speaker], non_streaming_mode=True, **k,
            )
        first = codes[0][:, 0].tolist()
        has = eos in first
        print(f"  [{mode:6s}] n={len(first):3d} eos={has}" +
              (f" at {first.index(eos)}" if has else f" last={first[-6:]}"))
        if not has:
            print(f"            head={first[:12]}")
        del codes
    del tts
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--text", default="今天天气不错。")
    ap.add_argument("--speaker", default="chun")
    args = ap.parse_args()
    for m in args.models:
        print(f"\n== {m} ==")
        probe(m, args.text, args.speaker)


if __name__ == "__main__":
    main()
