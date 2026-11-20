#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""quality_eval.py —— 量化质量评估口径（步 3）

为什么要有这一步
----------------
后面每一步「更快」都要配一个「掉了多少」，而"我觉得还行"不是证据。本脚本把
三件互相独立的事固化成可复现的数字，之后 int8 / int4 / fp8 都跑同一套：

  1. ppl        —— 固定语料上的 next-token NLL（逐段记录，支持配对比较）
  2. logits     —— 8 条固定 prompt 的 top-100 logits + 全词表 logsumexp
                   -> 与基线比 KL，能捕捉采样分布的整体平移/收缩
  3. generation —— 固定 seed + 固定 temperature 的 20 条输出
                   -> 文本、token id、重复率、以及与基线的逐 token 一致率

设计上的三个要点（都踩过或差点踩）
----------------------------------
* **ppl 走引擎自己的 teacher forcing**，不是另写一份模型前向：逐 token 把"真实
  的下一个 token"喂回去，读引擎算出来的 logits。这样一个 off-by-one 或 KV 写错
  位置都会立刻被 ppl 拉爆（正常 ~15，写错位置会到几十上百）。
  实现上绕开 Scheduler，直接用 ModelRunner 的 prepare_* + run_model，避免采样
  带来的随机性；KV 块自己从 BlockManager 拿（num_cached_blocks=0，绝不走前缀
  缓存，否则不同段可能命中彼此的块）。

* **评估时关掉 prefix caching**：同一批 prompt 跑第二遍会命中第一遍留下的块，
  prefill 的输入长度随之变化，numerics 也就会微动 —— 那样"两次生成结果逐 token
  相同"这条断言会被误判成不成立。质量评估不需要缓存。

* **NLL 用引擎自身的 bf16 logits**（与线上同口径），只把 logsumexp 和取对数
  放到 fp32。不为了"更精确"把 lm_head 换成 fp32 GEMM —— 那测的就不是线上那个
  模型了。32768 个 token 聚合之后，bf16 的舍入噪声在均值上远小于 1e-4。

用法
----
    # 建基线（此时还没量化，只是把口径固定下来）
    python quality_eval.py --model models/Qwen3-0.6B --tag baseline \
        --out results/quality_baseline.json

    # 之后每一步量化，都拿基线做差
    python quality_eval.py --model models/Qwen3-0.6B --tag int8 \
        --baseline results/quality_baseline.json --out results/quality_int8.json

    # 只验脚本能跑
    python quality_eval.py --quick
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time

import torch

# 主口径段长（步 3 起的全部 ppl 基线都是 512 段）。3584 段是步 11 的长上下文口径，
# 由 --length 切换；两套语料的 token 流同源（3584 = 7×512），所以能逐位对上。
CORPUS_LEN = 512


# ---------------------------------------------------------------- 语料
def load_corpus(root: str, lang: str, length: int):
    path = os.path.join(root, f"{lang}_{length}.ids.json")
    with open(path, encoding="utf-8") as f:
        segs = json.load(f)
    for i, s in enumerate(segs):
        assert len(s) == length, f"{path} 第 {i} 段长度 {len(s)} != {length}"
    return segs


def sha16(obj) -> str:
    return hashlib.sha256(json.dumps(obj, separators=(",", ":")).encode()).hexdigest()[:16]


# ---------------------------------------------------------------- 绕过 Scheduler 的序列
class ScoreSeq:
    """只实现 prepare_prefill / prepare_decode 需要的那几个属性。

    不复用 Sequence 是因为它的 num_tokens / num_blocks 是联动的 property：
    teacher forcing 需要"token_ids 是完整的 512 个，但当前只消费到第 i 个"这种
    状态，直接改 Sequence 会和它的 property 打架。这里显式维护两套：
      * token_ids  全程不变（答案）
      * num_tokens 当前消费位置 + 1（= 位置 i）
    """

    def __init__(self, token_ids: list[int], block_size: int):
        self.token_ids = token_ids
        self.block_size = block_size
        self.num_tokens = 1
        self.last_token = token_ids[0]
        self.num_cached_tokens = 0
        self.num_scheduled_tokens = 1
        self.temperature = 1.0
        self.ignore_eos = True
        self.blocks: list[int] = []
        self.block_table: list[int] = []

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, k):
        return self.token_ids[k]

    def alloc(self, bm):
        self.num_tokens = len(self.token_ids)
        bm.allocate(self, 0)              # 0 个缓存块 -> 一定拿新块，绝不复用
        self.blocks = list(self.block_table)
        self.set_pos(0)

    def set_pos(self, i: int):
        """把序列摆到「已经写了 0..i，现在要为 token i 算 hidden」的状态。"""
        self.num_tokens = i + 1
        self.last_token = self.token_ids[i]
        self.block_table = self.blocks[:self.num_blocks]

    def set_prefill(self, n: int):
        """prefill 前 n 个 token（一次算完，不给下一步留答案）。"""
        self.num_tokens = n
        self.num_scheduled_tokens = n
        self.block_table = self.blocks[:self.num_blocks]

    def free(self, bm):
        self.block_table = self.blocks   # deallocate 要按完整块表退引用
        bm.deallocate(self)


def _nll(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """logits(bs,vocab) -> 每行的 -log p(target)，fp32 求和。"""
    lg = logits.float()
    lse = torch.logsumexp(lg, dim=-1)
    picked = lg.gather(1, targets.view(-1, 1)).squeeze(1)
    return lse - picked


# ---------------------------------------------------------------- 1. ppl
def eval_ppl(mr, bm, segs: list[list[int]], batch: int = 32, verbose: bool = True,
             pos_block: int = 512):
    """teacher forcing 求 next-token NLL。

    第一个 token 用 prefill（1 个 token）算，之后 L-2 步全部走 decode —— decode 是
    线上真正产出 token 的那条路径，用它算 ppl 才有意义；顺带它还覆盖了 CUDA graph。

    `pos_block` 把序列切成等长的位置桶，另外记一份**分位置**的 NLL。理由是长上下文
    的量化误差是逐位置变化的量：只报一个整段均值，就无法回答「误差是随上下文累积
    的，还是恒定的」。fp8 KV 的第一步验收（步 11）正是这个问题。
    """
    L = len(segs[0])
    nb_pos = (L + pos_block - 1) // pos_block
    per_seg_sum = [0.0] * len(segs)
    per_seg_n = [0] * len(segs)
    per_seg_pos = [[0.0] * nb_pos for _ in segs]
    per_seg_pos_n = [[0] * nb_pos for _ in segs]

    def acc(j, i, v):
        """把位置 i 的 NLL 记到段 j 的总额与第 i//pos_block 个位置桶。"""
        per_seg_sum[j] += v
        per_seg_n[j] += 1
        b = i // pos_block
        per_seg_pos[j][b] += v
        per_seg_pos_n[j][b] += 1

    from nanovllm.utils.context import reset_context
    t0 = time.perf_counter()
    for b0 in range(0, len(segs), batch):
        chunk = segs[b0: b0 + batch]
        shims = []
        for ids in chunk:
            sh = ScoreSeq(ids, mr.block_size)
            sh.alloc(bm)
            shims.append(sh)
        bs = len(shims)
        dev = "cuda"

        # --- 位置 0：prefill 1 个 token，预测 token[1]
        input_ids, positions = mr.prepare_prefill(shims)
        logits = mr.run_model(input_ids, positions, True)
        reset_context()
        tgt = torch.tensor([sh.token_ids[1] for sh in shims], dtype=torch.long, device=dev)
        nll = _nll(logits, tgt)
        for j, v in enumerate(nll.tolist()):
            acc(b0 + j, 0, v)

        # --- 位置 1..L-2：decode，用金标准 token 喂回去
        for i in range(1, L - 1):
            for sh in shims:
                sh.set_pos(i)
            input_ids, positions = mr.prepare_decode(shims)
            logits = mr.run_model(input_ids, positions, False)
            reset_context()
            tgt = torch.tensor([sh.token_ids[i + 1] for sh in shims], dtype=torch.long, device=dev)
            nll = _nll(logits, tgt)
            for j, v in enumerate(nll.tolist()):
                acc(b0 + j, i, v)

        for sh in shims:
            sh.free(bm)
        if verbose:
            print(f"  [ppl] {b0 + bs}/{len(segs)} 段  "
                  f"{time.perf_counter() - t0:.1f}s", flush=True)
    total = sum(per_seg_sum)
    n = sum(per_seg_n)
    pos_sum = [sum(s[b] for s in per_seg_pos) for b in range(nb_pos)]
    pos_n = [sum(s[b] for s in per_seg_pos_n) for b in range(nb_pos)]
    return {
        "nll_sum": total,
        "n_tokens": n,
        "nll_per_token": total / n,
        "ppl": math.exp(total / n),
        "per_seg_nll_mean": [s / k for s, k in zip(per_seg_sum, per_seg_n)],
        "per_seg_nll_sum": per_seg_sum,
        "per_seg_n": per_seg_n,
        "pos_block": pos_block,
        "pos_nll_mean": [s / k if k else float("nan") for s, k in zip(pos_sum, pos_n)],
        "pos_n": pos_n,
        "per_seg_pos_nll_mean": [[s / k if k else float("nan") for s, k in zip(sr, kn)]
                                 for sr, kn in zip(per_seg_pos, per_seg_pos_n)],
        "secs": round(time.perf_counter() - t0, 2),
    }


# ---------------------------------------------------------------- 2. logits
TOPK = 500


def eval_logits(mr, bm, prompts: list[list[int]]):
    """8 条固定 prompt 各取最后一个位置的 logits，存 top-100 + 全词表 lse。

    存 lse 是为了让 KL 能算全：top-100 之外的那部分质量按 1 - sum(top100) 处理，
    两边用同一个口径，残差项本身也进 KL。不存整个 152k 向量（8×152k 太大，而且
    逐位比较没有意义 —— 我们要的是分布差多少）。
    """
    shims = []
    for ids in prompts:
        sh = ScoreSeq(ids, mr.block_size)
        sh.alloc(bm)
        sh.set_prefill(len(ids))
        shims.append(sh)
    input_ids, positions = mr.prepare_prefill(shims)
    logits = mr.run_model(input_ids, positions, True).float()
    from nanovllm.utils.context import reset_context
    reset_context()
    out = []
    for row in logits:
        lse = torch.logsumexp(row, dim=-1)
        top = torch.topk(row, TOPK)
        p = torch.softmax(row, dim=-1)
        ent = float(-(p * torch.log(p + 1e-12)).sum())
        covered = float(torch.exp(top.values - lse).sum())
        out.append({
            "lse": round(float(lse), 4),
            "topk_ids": top.indices.tolist(),
            "topk_logits": [round(float(v), 4) for v in top.values],
            "entropy": round(ent, 4),
            "topk_mass": round(covered, 6),
            "argmax": int(top.indices[0]),
        })
    for sh in shims:
        sh.free(bm)
    return out


def kl_topk(a: dict, b: dict, vocab_mass=1.0) -> float:
    """KL(P_a || P_b)，只在「两边 top-K 的交集 ∪ {rest}」上求。

    为什么取交集而不是并集：top-K 的边界项每次运行都可能换一两个（logits 差
    1 个 bf16 ulp 就足以让相近的两个 token 换位），若对"只在一侧出现"的 token
    直接给 q=0，KL 会变成 inf —— 自己和自己比都会 inf，指标就废了。取交集、
    把剩余质量合成一个 rest 项，两个分布都仍是标准归一化的，KL ≥ 0 且自比为 0。
    """
    def dist(x):
        return {tid: math.exp(lg - x["lse"])
                for tid, lg in zip(x["topk_ids"], x["topk_logits"])}
    pa, pb = dist(a), dist(b)
    keys = [k for k in pa if k in pb]
    p_rest = max(vocab_mass - sum(pa[k] for k in keys), 0.0)
    q_rest = max(vocab_mass - sum(pb[k] for k in keys), 0.0)
    kl = sum(pa[k] * math.log(pa[k] / pb[k]) for k in keys if pa[k] > 0)
    if p_rest > 0:
        if q_rest <= 0:
            return float("inf")
        kl += p_rest * math.log(p_rest / q_rest)
    return kl


# ---------------------------------------------------------------- 3. generation
def ngram_stats(ids: list[int], n: int):
    if len(ids) < n:
        return 0, 0
    grams = [tuple(ids[i: i + n]) for i in range(len(ids) - n + 1)]
    return len(set(grams)), len(grams)


def gen_metrics(samples: list[dict]):
    """重复率。token 级，不用字符级 —— 中文一个字一个 token 的比例很高，字符级
    会把"合理重复的虚词"记成重复。

    max_repeat_n 是「这一段里最长的、出现过两次的 n-gram 长度」：模型退化成长
    循环时这个值会直接顶到上限 16。它不能取平均（平均 7.55 条没有对应实体），
    所以同时报均值与最差值。
    """
    per = []
    for s in samples:
        ids = s["token_ids"]
        u1, t1 = len(set(ids)), len(ids)
        u2, t2 = ngram_stats(ids, 2)
        u4, t4 = ngram_stats(ids, 4)
        best = 0
        for n in range(2, 17):
            u, t = ngram_stats(ids, n)
            if u < t:
                best = n
        per.append({
            "distinct1": u1 / t1 if t1 else 0.0,
            "distinct2": u2 / t2 if t2 else 0.0,
            "dup4": 1 - (u4 / t4 if t4 else 1.0),
            "max_repeat_n": best,
        })
    n = len(per) or 1
    macro = {k: sum(p[k] for p in per) / n for k in ("distinct1", "distinct2", "dup4")}
    macro["max_repeat_n_mean"] = sum(p["max_repeat_n"] for p in per) / n
    return {"macro": macro, "max_repeat_n_worst": max(p["max_repeat_n"] for p in per),
            "per_sample": per}


def eval_generation(llm, prompts: list[list[int]], seed: int, temperature: float,
                    max_tokens: int, repeats: int = 2):
    from nanovllm.sampling_params import SamplingParams
    runs = []
    for _ in range(repeats):
        torch.manual_seed(seed)
        sp = SamplingParams(temperature=temperature, max_tokens=max_tokens, ignore_eos=True)
        out = llm.generate(prompts, sp, use_tqdm=False)
        runs.append([o["token_ids"] for o in out])
    samples = [{"token_ids": ids, "text": llm.tokenizer.decode(ids)} for ids in runs[0]]
    # 可复现性：同一 seed 两次跑，逐 token 比
    mismatch = 0
    for a, b in zip(runs[0], runs[1]):
        if a != b:
            mismatch += sum(1 for x, y in zip(a, b) if x != y) + abs(len(a) - len(b))
    return samples, {"identical": mismatch == 0, "mismatch_tokens": mismatch}


# ---------------------------------------------------------------- 对比
def compare(base: dict, cur: dict) -> dict:
    res = {}
    # ppl：配对比较（同一段、同一 token 数），差值比绝对值可靠得多
    for lang in ("en", "zh"):
        b, c = base["ppl"].get(lang), cur["ppl"].get(lang)
        if not b or not c:
            continue
        # 段数/段长不同就不能配对（512 基线 vs 3584 结果）。静默 zip 会把 9 段
        # 对 64 段截成 9 对，算出一个看起来正常的假差值。
        if len(b["per_seg_nll_mean"]) != len(c["per_seg_nll_mean"]):
            res.setdefault("_skipped", []).append(
                f"{lang}: 段数不同 {len(b['per_seg_nll_mean'])} vs "
                f"{len(c['per_seg_nll_mean'])}，无法配对")
            continue
        paired = [x - y for x, y in zip(c["per_seg_nll_mean"], b["per_seg_nll_mean"])]
        m = sum(paired) / len(paired)
        var = sum((x - m) ** 2 for x in paired) / max(len(paired) - 1, 1)
        se = math.sqrt(var / len(paired))
        res[lang] = {
            "ppl_base": b["ppl"], "ppl_cur": c["ppl"],
            "ppl_rel": c["ppl"] / b["ppl"] - 1,
            "nll_delta_per_token": (c["nll_sum"] - b["nll_sum"]) / b["n_tokens"],
            "paired_delta": m, "paired_se": se,
            "paired_z": m / se if se > 0 else 0.0,
            "segs_worse": sum(1 for x in paired if x > 0),
            "segs_total": len(paired),
        }
        # 分位置桶的配对差：回答「退化是随上下文累积，还是整段恒定」
        pb = b.get("pos_block") or c.get("pos_block")
        if b.get("per_seg_pos_nll_mean") and c.get("per_seg_pos_nll_mean"):
            nb = min(len(b["per_seg_pos_nll_mean"][0]), len(c["per_seg_pos_nll_mean"][0]))
            buckets = []
            for k in range(nb):
                d = [cc[k] - bb[k]
                     for bb, cc in zip(b["per_seg_pos_nll_mean"], c["per_seg_pos_nll_mean"])]
                m2 = sum(d) / len(d)
                v2 = sum((x - m2) ** 2 for x in d) / max(len(d) - 1, 1)
                se2 = math.sqrt(v2 / len(d))
                buckets.append({
                    "start": k * pb,
                    "nll_base": b["pos_nll_mean"][k], "nll_cur": c["pos_nll_mean"][k],
                    "delta": m2, "se": se2, "z": m2 / se2 if se2 > 0 else 0.0,
                })
            res[lang]["pos_buckets"] = buckets
    # logits：KL 与 argmax 一致率
    # 注意 `--skip logits` 时这两个字段是空 dict，`"logits" in cur` 仍然成立，
    # 所以必须判内部键，否则 skip 模式下做对比会 KeyError。
    if base.get("logits", {}).get("prompts") and cur.get("logits", {}).get("prompts"):
        kls, same = [], 0
        for a, b in zip(base["logits"]["prompts"], cur["logits"]["prompts"]):
            kls.append(kl_topk(a, b))
            same += int(a["argmax"] == b["argmax"])
        res["logits"] = {
            "kl_mean": sum(kls) / len(kls), "kl_max": max(kls),
            "kl_per_prompt": kls,
            "argmax_agree": same / len(kls),
        }
    # generation：逐 token 一致率
    if base.get("gen", {}).get("samples") and cur.get("gen", {}).get("samples"):
        tot = hit = 0
        first_div = []
        for a, b in zip(base["gen"]["samples"], cur["gen"]["samples"]):
            n = min(len(a["token_ids"]), len(b["token_ids"]))
            hit += sum(1 for x, y in zip(a["token_ids"][:n], b["token_ids"][:n]) if x == y)
            tot += max(len(a["token_ids"]), len(b["token_ids"]))
            # 首次分歧位置：比"总一致率"稳得多。自回归会放大分歧，第 5 个 token
            # 岔了和第 60 个 token 岔了，在总一致率上看起来差不多。
            d = next((i for i, (x, y) in enumerate(zip(a["token_ids"], b["token_ids"])) if x != y),
                     n)
            first_div.append(d)
        res["gen"] = {
            "token_agree": hit / tot if tot else 0.0,
            "identical_samples": sum(
                1 for a, b in zip(base["gen"]["samples"], cur["gen"]["samples"])
                if a["token_ids"] == b["token_ids"]),
            "samples_total": len(base["gen"]["samples"]),
            "first_div_mean": sum(first_div) / len(first_div),
            "first_div_min": min(first_div),
            "distinct2_base": base["gen"]["metrics"]["macro"]["distinct2"],
            "distinct2_cur": cur["gen"]["metrics"]["macro"]["distinct2"],
            "max_repeat_n_base": base["gen"]["metrics"]["macro"]["max_repeat_n_mean"],
            "max_repeat_n_cur": cur["gen"]["metrics"]["macro"]["max_repeat_n_mean"],
            "max_repeat_n_worst_base": base["gen"]["metrics"].get("max_repeat_n_worst"),
            "max_repeat_n_worst_cur": cur["gen"]["metrics"].get("max_repeat_n_worst"),
        }
    return res


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen3-0.6B")
    ap.add_argument("--corpus", default="assets/corpus")
    ap.add_argument("--out", default=None)
    ap.add_argument("--baseline", default=None, help="与这份结果做差")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--cfg", default='{"prefill_compile": "model"}')
    # 用独立参数而不是塞进 --cfg：JSON 走 wsl.exe 的两层引号会被吃掉
    ap.add_argument("--quant", default=None, choices=["none", "lm_head", "all", "int4_head", "int4_all", "int4_mlp",
                             "fp8_fake"])
    ap.add_argument("--keep-bf16", type=int, default=None, help="1/0：prefill 是否保留 bf16 权重")
    ap.add_argument("--int4-group", type=int, default=None, help="int4 的 group size，0=per-channel")
    ap.add_argument("--kv-cache-dtype", default=None, choices=["bf16", "fp8"],
                    help="步 11：真实 fp8 KV cache（自动带 attn_backend=triton）")
    ap.add_argument("--fp8-kv", default=None, choices=["off", "fake", "fake_k", "fake_v"],
                    help="步 11 预演：fake = K/V 按 e4m3 往返再写入 cache（测 ppl 代价）")
    ap.add_argument("--fp8-act-head", type=int, default=None,
                    help="1：连 lm_head 的激活也量化（默认 0，跳过）")
    ap.add_argument("--prefill-compile", default=None, choices=["off", "layer", "model"],
                    help="覆盖 cfg 里的 prefill 编译档（假量化用的是 torch 算子，建议 off）")
    ap.add_argument("--fp8-act", default=None, choices=["off", "tensor", "token", "int8"],
                    help="步 12 预演：激活按 e4m3 往返（只改数值），测 W8A8 的 ppl 代价")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--batch", type=int, default=32, help="ppl 的批大小（受 KV 块数限制）")
    ap.add_argument("--length", type=int, default=512,
                    help="ppl 语料段长。512 是主口径（步 3 起的基线）；3584 用于测长上下文")
    ap.add_argument("--pos-block", type=int, default=512,
                    help="分位置统计的桶宽（默认 512，与短语料同粒度）")
    ap.add_argument("--skip", default="", help="逗号分隔: ppl,logits,gen")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    if args.quick:
        skip |= {"logits", "gen"}

    from nanovllm import LLM
    cfg = json.loads(args.cfg)
    if args.quant:
        cfg["quant"] = args.quant
    if args.keep_bf16 is not None:
        cfg["int8_keep_bf16"] = bool(args.keep_bf16)
    if args.int4_group is not None:
        cfg["int4_group"] = int(args.int4_group)
    if args.fp8_kv is not None:
        cfg["fp8_kv"] = args.fp8_kv
    if args.fp8_act is not None:
        cfg["fp8_act"] = args.fp8_act
    if args.fp8_act_head is not None:
        cfg["fp8_act_skip_head"] = not bool(args.fp8_act_head)
    if args.prefill_compile is not None:
        cfg["prefill_compile"] = args.prefill_compile
    if args.kv_cache_dtype is not None:
        cfg["kv_cache_dtype"] = args.kv_cache_dtype
        if args.kv_cache_dtype == "fp8":
            cfg["attn_backend"] = "triton"
    cfg.setdefault("enable_prefix_caching", False)   # 见文件头：评估口径统一关缓存
    print(f"===== quality_eval [{args.tag}] cfg={cfg} =====", flush=True)
    t_load = time.perf_counter()
    llm = LLM(args.model, **cfg)
    mr, bm = llm.model_runner, llm.scheduler.block_manager
    print(f"加载 {time.perf_counter() - t_load:.1f}s, KV blocks={len(bm.blocks)}", flush=True)

    # 批大小必须塞得进 KV 池，否则 BlockManager 会静默抢占（表现为步时变长、
    # ppl 却看不出异常 —— 那是最难查的一类假数据）。按段长算上界再夹一次。
    per_seq = -(-args.length // mr.block_size)
    max_batch = max(1, len(bm.blocks) // per_seq)
    if args.batch > max_batch:
        print(f"[!] batch={args.batch} 超过 KV 池上限（{per_seq} 块/段 × {args.batch} "
              f"> {len(bm.blocks)}）→ 夹到 {max_batch}", flush=True)
        args.batch = max_batch
    print(f"[cfg] length={args.length} pos_block={args.pos_block} "
          f"batch={args.batch} ({per_seq} 块/段)", flush=True)

    en = load_corpus(args.corpus, "en", args.length)
    zh = load_corpus(args.corpus, "zh", args.length)
    n_seg = 8 if args.quick else len(en)
    result = {
        "meta": {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "tag": args.tag,
            "model": args.model,
            "cfg": cfg,
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "seed": args.seed,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "corpus": {"en": sha16(en), "zh": sha16(zh),
                       "n_segs": [len(en), len(zh)], "len": args.length,
                       "pos_block": args.pos_block},
            "ppl_path": "decode teacher-forcing (引擎自身 logits, bf16)",
            "topk": TOPK,
        },
        "ppl": {}, "logits": {}, "gen": {},
    }

    # ---- ppl
    if "ppl" not in skip:
        for lang, segs in (("en", en), ("zh", zh)):
            print(f"[ppl] {lang} ...", flush=True)
            result["ppl"][lang] = eval_ppl(mr, bm, segs[:n_seg], batch=args.batch,
                                            pos_block=args.pos_block)
            r = result["ppl"][lang]
            print(f"[ppl] {lang}: ppl={r['ppl']:.3f} "
                  f"nll/token={r['nll_per_token']:.4f} n={r['n_tokens']} "
                  f"({r['secs']}s)", flush=True)
            if len(r.get("pos_nll_mean", [])) > 1:
                seg = "  ".join(f"{k*args.pos_block}:{v:.3f}"
                                for k, v in enumerate(r["pos_nll_mean"]))
                print(f"[ppl] {lang} 分位置 NLL/token（桶起点:值）: {seg}", flush=True)

    # ---- logits（4 EN + 4 ZH，各取前 128 token；bs=8 一次前向）
    if "logits" not in skip:
        print("[logits] 8 条 prompt 一次 bs=8 前向 ...", flush=True)
        prompts = [en[i][:128] for i in range(4)] + [zh[i][:128] for i in range(4)]
        first = eval_logits(mr, bm, prompts)
        # 同一个进程里连着算两遍：把"指标自身的抖动"标出来。
        # 不标的话，将来 int8 报出一个 KL=1e-3，没人知道其中多少是噪声。
        second = eval_logits(mr, bm, prompts)
        self_kl = [kl_topk(a, b) for a, b in zip(first, second)]
        result["logits"] = {
            "prompts": first,
            "prompt_ids": [sha16(p) for p in prompts],
            "prompt_len": 128,
            "self_kl_mean": sum(self_kl) / len(self_kl),
            "self_kl_max": max(self_kl),
            "self_argmax_agree": sum(int(a["argmax"] == b["argmax"])
                                     for a, b in zip(first, second)) / len(first),
        }
        for i, p in enumerate(first):
            print(f"  p{i}: argmax={p['argmax']} entropy={p['entropy']:.3f} "
                  f"top{TOPK}mass={p['topk_mass']:.4f}")
        print(f"  同进程重复 KL={result['logits']['self_kl_mean']:.2e} "
              f"max={result['logits']['self_kl_max']:.2e}")

    # ---- generation
    if "gen" not in skip:
        print("[gen] 20 条 prompt × %d token，seed=%d T=%.2f ..."
              % (args.max_tokens, args.seed, args.temperature), flush=True)
        prompts = [en[i][:32] for i in range(10)] + [zh[i][:32] for i in range(10)]
        if args.quick:
            prompts = prompts[:4]
        samples, det = eval_generation(llm, prompts, args.seed, args.temperature,
                                       args.max_tokens, repeats=2)
        result["gen"] = {
            "seed": args.seed, "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "samples": samples,
            "metrics": gen_metrics(samples),
            "determinism": det,
        }
        m = result["gen"]["metrics"]["macro"]
        print(f"[gen] distinct2={m['distinct2']:.4f} dup4={m['dup4']:.4f} "
              f"max_repeat_n mean={m['max_repeat_n_mean']:.2f} "
              f"worst={result['gen']['metrics']['max_repeat_n_worst']} "
              f"可复现={det['identical']}", flush=True)

    # ---- 对比
    if args.baseline and os.path.exists(args.baseline):
        with open(args.baseline, encoding="utf-8") as f:
            base = json.load(f)
        result["compare_base"] = os.path.basename(args.baseline)
        result["compare"] = compare(base, result)
        c = result["compare"]
        if "en" in c:
            for lang in ("en", "zh"):
                if lang in c:
                    x = c[lang]
                    print(f"[cmp] {lang} ppl {x['ppl_base']:.3f} -> {x['ppl_cur']:.3f} "
                          f"({x['ppl_rel']*100:+.2f}%) 配对Δ={x['paired_delta']*1000:+.3f}me"
                          f"±{x['paired_se']*1000:.3f} z={x['paired_z']:+.1f}")
                    if len(x.get("pos_buckets") or []) > 1:
                        print(f"[cmp] {lang} 分位置配对Δ（me，括号内为 z）:")
                        print("        " + "  ".join(
                            f"{bk['start']}:{bk['delta']*1000:+.2f}({bk['z']:+.1f})"
                            for bk in x["pos_buckets"]))
        if "logits" in c:
            print(f"[cmp] KL mean={c['logits']['kl_mean']:.2e} "
                  f"max={c['logits']['kl_max']:.2e} argmax一致={c['logits']['argmax_agree']:.3f}")
        if "gen" in c:
            print(f"[cmp] 逐 token 一致率={c['gen']['token_agree']:.4f} "
                  f"完全相同样本={c['gen']['identical_samples']}/{c['gen']['samples_total']} "
                  f"首次分歧位置 均值={c['gen']['first_div_mean']:.1f} "
                  f"最小={c['gen']['first_div_min']}")

    result["meta"]["elapsed_s"] = round(time.perf_counter() - t_load, 1)
    out = args.out or f"results/quality_{args.tag}.json"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
