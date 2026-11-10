"""扫描各 epoch checkpoint 的生成长度与重复度。

注意：talker_codes_list 用 `[:stop_index]` 截断，EOS 本身不在返回序列里。
所以 len < max_new_tokens 就代表正常触发了 EOS。
"""
import argparse
from collections import Counter

import torch
from qwen_tts import Qwen3TTSModel


def probe(path, text, speaker, max_new):
    tts = Qwen3TTSModel.from_pretrained(
        path, torch_dtype=torch.bfloat16, device_map="cuda:0",
        attn_implementation="sdpa",
    )
    input_ids = tts._tokenize_texts([tts._build_assistant_text(text)])
    kw = tts._merge_generate_kwargs(max_new_tokens=max_new)
    with torch.no_grad():
        codes, _ = tts.model.generate(
            input_ids=input_ids, instruct_ids=[None], languages=["Auto"],
            speakers=[speaker], non_streaming_mode=True, **kw,
        )
    first = codes[0][:, 0].tolist()
    stopped = len(first) < max_new
    top = Counter(first).most_common(1)[0]
    ratio = top[1] / max(len(first), 1)
    print(f"  n={len(first):4d}  stopped={stopped}  "
          f"dur={len(first)/12.5:6.2f}s  top_token={top[0]}x{top[1]} ({ratio:.0%})")
    del tts, codes
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="output")
    ap.add_argument("--epochs", nargs="+", type=int, default=[3, 7, 11, 15, 19, 23])
    ap.add_argument("--text", default="今天天气不错。")
    ap.add_argument("--speaker", default="chun")
    ap.add_argument("--max_new", type=int, default=300)
    args = ap.parse_args()
    for e in args.epochs:
        print(f"\n== epoch-{e} ==")
        probe(f"{args.dir}/checkpoint-epoch-{e}", args.text, args.speaker, args.max_new)


if __name__ == "__main__":
    main()
