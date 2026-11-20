#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ppl_hf_ref.py —— 用 HuggingFace transformers 复算同一份语料的 ppl（对账用）

为什么需要它
------------
quality_eval.py 的 ppl 是靠"逐 token teacher forcing + 引擎自己的 decode 路径"
算出来的。这条路只要有一个 off-by-one（位置、context_len、slot_mapping），ppl
就会明显偏 —— 但"明显偏"是多少，光看数字看不出来。

所以另起一个进程（nano-vllm 会把 90% 显存吃成 KV 池，同进程里装不下第二份权重），
用 transformers 的官方前向把同样的 64 段 * 512 token 再算一遍。两边应该落在同一个
量级（差异主要来自 attention kernel 与 bf16 累加顺序，通常 <1%）。

不追求逐位相同：这里要证的是"没有算错"，不是"两边实现一致"。

用法:
    python ppl_hf_ref.py --corpus assets/corpus --model models/Qwen3-0.6B \
        --out results/ppl_hf_ref.json
"""
import argparse
import json
import math
import os
import sys

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM


def run(model_dir, segs, tag):
    tok_len = len(segs[0])
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    per_seg = []
    with torch.inference_mode():
        for i, seg in enumerate(segs):
            ids = torch.tensor([seg], device="cuda")
            logits = model(ids).logits.float()
            nll = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)),
                                  ids[:, 1:].reshape(-1), reduction="none")
            per_seg.append(float(nll.sum()))
            if (i + 1) % 16 == 0:
                print(f"  [{tag}] {i+1}/{len(segs)}", flush=True)
    total = sum(per_seg)
    n = len(segs) * (tok_len - 1)
    return {"ppl": math.exp(total / n), "nll_per_token": total / n,
            "n_tokens": n, "per_seg_nll_sum": per_seg}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen3-0.6B")
    ap.add_argument("--corpus", default="assets/corpus")
    ap.add_argument("--length", type=int, default=512)
    ap.add_argument("--out", default="results/ppl_hf_ref.json")
    args = ap.parse_args()

    out = {"model": args.model, "impl": "transformers + sdpa, bf16", "langs": {}}
    for lang in ("en", "zh"):
        with open(os.path.join(args.corpus, f"{lang}_{args.length}.ids.json"), encoding="utf-8") as f:
            segs = json.load(f)
        print(f"[{lang}] {len(segs)} 段 × {args.length} ...", flush=True)
        out["langs"][lang] = run(args.model, segs, lang)
        r = out["langs"][lang]
        print(f"[{lang}] ppl={r['ppl']:.3f} nll/token={r['nll_per_token']:.4f} n={r['n_tokens']}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"-> {args.out}")


if __name__ == "__main__":
    sys.exit(main())
