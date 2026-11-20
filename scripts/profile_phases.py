#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
步 1：验证 prefill 的 launch 开销归因（v2）
============================================

要回答四个问题：

  Q1  prefill 的 CPU 提交占多少？是否与 token 数无关（=固定开销）？
      -> 在 run_model 外层插桩，分别量 submit / tail / wall。
         判据：submit 在 seq_len=16..2048 上是否平坦。

  Q2  这堆固定开销是否随层数线性增长？（决定"整层 compile"能省多少）
      -> 临时把 model.layers 截成 28/14/4/1 层，量 submit 的边际成本。

  Q3  钱花在哪类 op 上？
      -> torch.profiler 的 key_averages(self_cpu_time_total)。

  Q4  decode 稳态（走 CUDA graph）的 CPU 占比？决定步 13 的位置。

不改任何源码：只包 run_model + 临时截断 layers（结束时恢复）。

用法（WSL，~/nano-vllm 下）：
    .venv/bin/python scripts/profile_phases.py --model models/Qwen3-0.6B
输出：
    results/profile_phases.txt    可读报告
    results/profile_phases.csv    样本级明细
    results/trace_prefill.json    chrome trace
"""

import argparse
import atexit
import csv
import json
import os
import statistics
import sys
from time import perf_counter

import torch

from nanovllm import LLM, SamplingParams


RECORDS = []
PHASE = {"name": "init"}
CAP = {"len": None, "armed": False, "path": None, "table": None}
ALL_LAYERS = []


# ------------------------------------------------------------------ 插桩

def install_hooks(mr):
    orig = mr.run_model

    def wrapped(input_ids, positions, is_prefill):
        n = int(input_ids.numel())
        do_cap = CAP["armed"] and is_prefill and n == CAP["len"]
        if do_cap:
            CAP["armed"] = False

        torch.cuda.synchronize()          # 起点干净：GPU 空闲
        t0 = perf_counter()

        if do_cap:
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=False,
                with_stack=False,
                profile_memory=False,
            ) as prof:
                out = orig(input_ids, positions, is_prefill)
            t1 = perf_counter()
            torch.cuda.synchronize()
            prof.export_chrome_trace(CAP["path"])
            CAP["table"] = prof.key_averages().table(
                sort_by="self_cpu_time_total", row_limit=22
            )
            RECORDS.append(dict(
                phase=PHASE["name"], kind="prefill", profiled=1, n=n, layers=len(mr.model.model.layers),
                submit_ms=(t1 - t0) * 1e3, tail_ms=0.0, wall_ms=(t1 - t0) * 1e3,
            ))
            return out

        out = orig(input_ids, positions, is_prefill)
        t1 = perf_counter()
        torch.cuda.synchronize()
        t2 = perf_counter()
        RECORDS.append(dict(
            phase=PHASE["name"],
            kind="prefill" if is_prefill else "decode",
            profiled=0, n=n, layers=len(mr.model.model.layers),
            submit_ms=(t1 - t0) * 1e3,
            tail_ms=(t2 - t1) * 1e3,
            wall_ms=(t2 - t0) * 1e3,
        ))
        return out

    mr.run_model = wrapped


# ------------------------------------------------------------------ trace 解析

GPU_CATS = ("kernel", "gpu_memcpy", "gpu_memset", "Kernel", "Memcpy", "Memset")


def parse_trace(path):
    """只依赖 CPU/runtime 事件；若没有 GPU kernel 事件也要能出 api 统计。"""
    with open(path, "r") as f:
        tr = json.load(f)
    evs = tr.get("traceEvents", [])
    gpu, cpu_api = [], {}
    for e in evs:
        if e.get("ph") != "X" or e.get("dur") is None:
            continue
        cat = e.get("cat", "")
        args = e.get("args") or {}
        if cat in GPU_CATS or ("stream" in args and not cat):
            gpu.append(dict(ts=e["ts"], dur=e["dur"], name=e.get("name", "?"),
                            stream=args.get("stream", e.get("tid"))))
        elif cat in ("cuda_runtime", "cuda_driver"):
            slot = cpu_api.setdefault(e.get("name", "?"), [0, 0.0])
            slot[0] += 1
            slot[1] += e["dur"]

    out = dict(cpu_api=cpu_api, gpu=None, n_events=len(evs),
               n_cpu_op=sum(1 for e in evs if e.get("cat") == "cpu_op"))
    if not gpu:
        return out

    per_stream = {}
    for g in gpu:
        s = per_stream.setdefault(g["stream"], [0.0, 0])
        s[0] += g["dur"]
        s[1] += 1
    main = max(per_stream.items(), key=lambda kv: kv[1][0])[0]
    seq = sorted([g for g in gpu if g["stream"] == main], key=lambda g: g["ts"])
    busy = sum(g["dur"] for g in seq)
    wall = (seq[-1]["ts"] + seq[-1]["dur"]) - seq[0]["ts"]
    gaps = []
    for a, b in zip(seq, seq[1:]):
        gap = b["ts"] - (a["ts"] + a["dur"])
        if gap > 0:
            gaps.append((gap, a["name"]))
    buckets = [(0, 5), (5, 20), (20, 50), (50, 100), (100, 500), (500, 1e9)]
    hist = [(f"{lo:g}-{'inf' if hi > 1e8 else f'{hi:g}'}us",
             len([g for g, _ in gaps if lo <= g < hi]),
             sum(g for g, _ in gaps if lo <= g < hi)) for lo, hi in buckets]
    by_kernel = {}
    for g in seq:
        s = by_kernel.setdefault(g["name"], [0.0, 0])
        s[0] += g["dur"]
        s[1] += 1
    out.update(gpu=dict(n_kernels=len(seq), busy_us=busy, wall_us=wall,
                       gap_sum_us=sum(g for g, _ in gaps), hist=hist,
                       top_kernels=sorted(by_kernel.items(), key=lambda kv: -kv[1][0])[:12],
                       top_gaps=sorted(gaps, reverse=True)[:5]))
    return out


# ------------------------------------------------------------------ 统计

def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else float("nan")


def p90(xs):
    xs = sorted(x for x in xs if x is not None)
    return xs[min(len(xs) - 1, int(len(xs) * 0.9))] if xs else float("nan")


def agg(records, phase=None, kind=None):
    """按 n 聚合成 {n: (submit, tail, wall, count)}。"""
    g = {}
    for r in records:
        if r["profiled"]:
            continue
        if phase and r["phase"] != phase:
            continue
        if kind and r["kind"] != kind:
            continue
        g.setdefault(r["n"], []).append(r)
    return dict(sorted((n, (med([x["submit_ms"] for x in xs]),
                            med([x["tail_ms"] for x in xs]),
                            med([x["wall_ms"] for x in xs]), len(xs)))
                       for n, xs in g.items()))


# ------------------------------------------------------------------ 主流程

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen3-0.6B")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--lens", default="16,128,512,1024,2048,3584")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--profile-len", type=int, default=2048)
    ap.add_argument("--decode-bs", type=int, default=64)
    ap.add_argument("--skip-decode", action="store_true")
    ap.add_argument("--skip-profile", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    trace_path = os.path.join(args.out_dir, "trace_prefill.json")
    lens = [int(x) for x in args.lens.split(",") if x.strip()]

    def pmt(L, seed):
        g = torch.Generator().manual_seed(seed)
        return torch.randint(100, 10000, (L,), generator=g).tolist()

    def SP(mt):
        return SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=mt)

    print("[1/6] loading model ...", flush=True)
    t_load = perf_counter()
    # 关掉 prefix caching：本步只关心纯 prefill 的开销结构，
    # 不希望重复测量因前缀命中而变成短 prefill。
    llm = LLM(
        args.model,
        max_num_batched_tokens=16384,
        max_num_seqs=max(64, args.decode_bs),
        gpu_memory_utilization=0.9,
        enable_prefix_caching=False,
    )
    mr = llm.model_runner
    ALL_LAYERS.extend(list(mr.model.model.layers))
    print(f"      loaded in {perf_counter()-t_load:.1f}s, "
          f"kv_blocks={mr.config.num_kvcache_blocks}, layers={len(ALL_LAYERS)}", flush=True)

    print("[2/6] warmup & raw wall (no instrumentation) ...", flush=True)
    PHASE["name"] = "warmup"
    llm.generate([pmt(512, 0)], SP(2), use_tqdm=False)
    llm.generate([pmt(2048, 1)], SP(2), use_tqdm=False)   # 把 2048 的 shape 也预热

    # 不插桩的裸墙钟：插桩会在每次 prefill 前 synchronize，让 GPU 长时间空闲而掉频，
    # 使 tail（GPU 时间）被高估。这里用同一进程、同一 prompt 量级做对照。
    raw = {}
    for L in lens:
        for r in range(2):
            t0 = perf_counter()
            llm.generate([pmt(L, 3000 + r * 31 + L)], SP(1), use_tqdm=False)
            raw.setdefault(L, []).append((perf_counter() - t0) * 1e3)

    install_hooks(mr)

    print("[3/6] prefill sweep: submit vs seq_len ...", flush=True)
    PHASE["name"] = "sweep"
    order = list(lens)
    import random
    random.Random(0).shuffle(order)
    for rep in range(args.reps):
        for L in order:
            llm.generate([pmt(L, 1000 + rep * 97 + L)], SP(1), use_tqdm=False)

    print("[4/6] layer scan: submit vs depth ...", flush=True)
    for k in [28, 14, 4, 1]:
        mr.model.model.layers = torch.nn.ModuleList(ALL_LAYERS[:k])
        PHASE["name"] = f"layers{k}"
        llm.generate([pmt(2048, 700 + k)], SP(2), use_tqdm=False)   # 丢弃
        llm.generate([pmt(2048, 800 + k)], SP(2), use_tqdm=False)   # 记录（单样本，够用）
    mr.model.model.layers = torch.nn.ModuleList(ALL_LAYERS)

    prof_table = None
    if not args.skip_profile:
        print("[5/6] profiler capture ...", flush=True)
        PHASE["name"] = "profile"
        CAP.update(len=128, armed=True, path=trace_path + ".warm")
        llm.generate([pmt(128, 77)], SP(3), use_tqdm=False)          # 烧掉 kineto 首次开销
        CAP.update(len=args.profile_len, armed=True, path=trace_path)
        llm.generate([pmt(args.profile_len, 78)], SP(3), use_tqdm=False)
        prof_table = CAP["table"]

    if not args.skip_decode:
        print("[6/6] decode steady state ...", flush=True)
        PHASE["name"] = "dec1"
        llm.generate([pmt(2048, 91)], SP(200), use_tqdm=False)
        PHASE["name"] = "decN"
        llm.generate([pmt(256, 500 + i) for i in range(args.decode_bs)], SP(48),
                     use_tqdm=False)

    atexit.unregister(llm.exit)
    llm.exit()

    # ------------------------------------------------------------- 汇总
    sweep = agg(RECORDS, phase="sweep", kind="prefill")
    layers = {int(k[6:]): agg(RECORDS, phase=k) for k in [f"layers{k}" for k in [28, 14, 4, 1]]}
    layers = {k: v[2048] for k, v in layers.items() if 2048 in v}
    trace = parse_trace(trace_path) if os.path.exists(trace_path) else None

    lines = []
    def W(s=""):
        lines.append(s)
        print(s, flush=True)

    W("=" * 80)
    W("步 1 · prefill 开销归因（nano-vllm / Qwen3-0.6B / RTX 4070 Laptop 8G）")
    W("=" * 80)
    W("")
    W("口径：submit = run_model 返回时的 CPU 墙钟（调用前已 synchronize，GPU 空闲）")
    W("      tail   = 返回后等待 GPU 跑完的时间")
    W("      wall   = submit + tail（≈ 无插桩时的 TTFT 上界）")
    W("      submit 大 & 与 token 数无关 => CPU 提交是固定瓶颈（launch bound）")
    W("")

    W("-" * 80)
    W("A. prefill：submit / tail / wall vs 序列长度（bs=1，各 3 次取中位）")
    W("-" * 80)
    W(f"{'seq_len':>8} {'n':>3} {'raw_wall':>9} {'submit(ms)':>11} {'tail(ms)':>9} "
      f"{'wall(ms)':>9} {'注入开销':>9}")
    rows = [(n, s, t, w, c) for n, (s, t, w, c) in sweep.items()]
    for n, s, t, w, c in rows:
        rw = med(raw.get(n, [float('nan')]))
        W(f"{n:>8} {c:>3} {rw:>9.2f} {s:>11.2f} {t:>9.2f} "
          f"{w:>9.2f} {w-rw:>+9.2f}")

    verdict = []
    if len(rows) >= 3:
        fit = [(r[0], r[1]) for r in rows if r[0] <= 2048]
        nbar = sum(x for x, _ in fit) / len(fit)
        ybar = sum(y for _, y in fit) / len(fit)
        den = sum((x - nbar) ** 2 for x, _ in fit) or 1.0
        b = sum((x - nbar) * (y - ybar) for x, y in fit) / den     # ms/token
        a = ybar - b * nbar                                        # ms
        slope = b * 1000                                           # µs/token
        small = rows[0]
        ref = next((r for r in rows if r[0] == 2048), rows[-1])
        W("")
        W(f"小 prefill({small[0]} tok): submit={small[1]:.2f}ms, tail={small[2]:.2f}ms, "
          f"wall={small[3]:.2f}ms -> GPU 利用率 {small[2]/max(small[3],1e-9)*100:.1f}%")
        W(f"参考点({ref[0]} tok)   : submit={ref[1]:.2f}ms, tail={ref[2]:.2f}ms, "
          f"wall={ref[3]:.2f}ms")
        W(f"submit 的 token 斜率     : {slope:+.2f} µs/token "
          f"（{small[0]}->{ref[0]} tok 共变 {ref[1]-small[1]:+.2f}ms）")
        W(f"最小二乘拟合 (n<=2048)    : submit = {a:.1f} ms + {slope:+.2f} µs/token × n")
        W("")
        idle_proxy = small[3] - small[2]
        if a > 25 and abs(slope) < 5.0:
            verdict.append(
                f"✅ 成立且比预期更极端：submit 在 {small[0]}~{ref[0]} token 上几乎恒定"
                f"（{small[1]:.1f} vs {ref[1]:.1f} ms，斜率仅 {slope:+.2f} µs/token）。"
                f"\n     即 ~{a:.0f} ms 是与 token 数无关的固定提交开销（截距）。"
            )
        elif abs(slope) < 1.0:
            verdict.append(f"⚠️ 开销是固定的（{small[1]:.1f}ms）但低于 25ms 阈值，收益没那么大。")
        else:
            verdict.append(
                f"⚠️ 开销随 token 增长（{slope:+.1f} µs/token），不完全是固定 launch 开销。"
            )
        if small[2] / max(small[3], 1e-9) < 0.10:
            verdict.append(
                f"     最直接的证据：{small[0]} token 的 prefill，GPU 只花 {small[2]:.2f} ms，"
                f"CPU 却提交了 {small[1]:.1f} ms —— GPU 空闲 ≈ {idle_proxy:.1f} ms"
                f"（利用率 {small[2]/max(small[3],1e-9)*100:.1f}%）。"
            )

    W("")
    W("-" * 80)
    W("B. 开销的来源：按层数截断（seq_len=2048，各 1 次记录值）")
    W("-" * 80)
    if layers:
        W(f"{'层数':>6} {'submit(ms)':>11} {'tail(ms)':>9} {'wall(ms)':>9}")
        for k in sorted(layers, reverse=True):
            s, t, w, _ = layers[k]
            W(f"{k:>6} {s:>11.2f} {t:>9.2f} {w:>9.2f}")
        if 28 in layers and 4 in layers:
            per_layer = (layers[28][0] - layers[4][0]) / 24
            base = layers[4][0] - per_layer * 4
            W("")
            W(f"边际成本 ≈ {per_layer:.2f} ms/层；外推到 0 层的基础开销 ≈ {base:.1f} ms")
            W(f"（基础开销 = embedding + final norm + lm_head + sampler + 框架入口）")
            if per_layer > 0.8:
                verdict.append(
                    f"     开销是**逐层均匀铺开**的：每层 {per_layer:.2f} ms（28 层共 "
                    f"{per_layer*28:.1f} ms），0 层基础仅 {base:.1f} ms。"
                    f"\n     => 把 op 合并/图化能直接砍掉这 {per_layer*28:.0f} ms 的大部分。"
                )
            else:
                verdict.append(
                    f"     开销主要在层外：0 层基础 {base:.1f} ms，每层仅 {per_layer:.2f} ms。"
                    f"\n     => 优化重点应放在入口/lm_head/prepare，而不是层内 op 合并。"
                )
    else:
        W("（无数据）")

    if trace:
        W("")
        W("-" * 80)
        W(f"C. profiler：prefill@{args.profile_len} 的 CPU 侧结构")
        W("-" * 80)
        W(f"trace 事件数 {trace['n_events']}，其中 cpu_op {trace['n_cpu_op']} 条")
        g = trace.get("gpu")
        if g:
            W(f"GPU kernel 事件 {g['n_kernels']} 个；净忙时 {g['busy_us']/1e3:.2f} ms，"
              f"墙钟 {g['wall_us']/1e3:.2f} ms，间隙和 {g['gap_sum_us']/1e3:.2f} ms")
        else:
            W("⚠️ 本环境的 CUPTI 未产出 GPU kernel 事件（只有 cpu_op / cuda_runtime / cuda_driver），")
            W("   因此没有 gap 直方图；改用 A 节的 submit/tail 反推 GPU 空闲，结论不受影响。")
        api = sorted(trace["cpu_api"].items(), key=lambda kv: -kv[1][1])[:10]
        W("")
        W("CUDA runtime/driver API 调用（真实驱动侧开销）：")
        W(f"{'API':>40} {'次数':>6} {'总时长(ms)':>12} {'单次(us)':>10}")
        for nm, (cnt, tot) in api:
            W(f"{nm[:40]:>40} {cnt:>6} {tot/1e3:>12.2f} {tot/max(cnt,1):>10.1f}")
        tot_api = sum(v[1] for v in trace["cpu_api"].values())
        W(f"{'合计':>40} {sum(v[0] for v in trace['cpu_api'].values()):>6} "
          f"{tot_api/1e3:>12.2f}")

    if prof_table:
        W("")
        W("-" * 80)
        W("D. profiler：按 self CPU time 排序（含 profiler 放大，看结构不看绝对值）")
        W("-" * 80)
        for ln in prof_table.splitlines():
            W(ln)
        W("")
        W("注意：表里 cudaDeviceSynchronize 一大块是 profiler 收尾自己的同步，不属于模型开销。")

    W("")
    W("-" * 80)
    W("E. decode 稳态（走 CUDA graph）")
    W("-" * 80)
    dec_rows = []
    for ph, label in (("dec1", "bs=1"), ("decN", f"bs={args.decode_bs}")):
        xs = [r for r in RECORDS if r["kind"] == "decode" and not r["profiled"] and r["phase"] == ph]
        if len(xs) > 10:
            xs = xs[len(xs) // 5:]
        if not xs:
            continue
        s, t, w = med([r["submit_ms"] for r in xs]), med([r["tail_ms"] for r in xs]), med([r["wall_ms"] for r in xs])
        dec_rows.append((label, len(xs), s, t, w))
        W(f"{label:>8} steps={len(xs):>3}  submit={s:>6.2f}ms  tail={t:>6.2f}ms  "
          f"wall={w:>6.2f}ms  CPU占比={s/max(w,1e-9)*100:>4.0f}%  "
          f"wall_p90={p90([r['wall_ms'] for r in xs]):>6.2f}ms")
    W("")
    W("注：插桩在每步前 synchronize，wall 会比无插桩（表 2 的 6.6 ms）略大；")
    W("    这里看 submit 的绝对值和 CPU 占比量级。")

    W("")
    W("=" * 80)
    W("结论")
    W("=" * 80)
    for v in verdict:
        W(v)
    if dec_rows:
        d = dec_rows[0]
        W(f"     decode(bs=1) 的 CPU 占比 {d[2]/max(d[4],1e-9)*100:.0f}%（submit {d[2]:.2f}ms / wall {d[4]:.2f}ms），"
          f"{'< 20%，步 13（CPU 流水线）可后置' if d[2]/max(d[4],1e-9) < 0.2 else '> 20%，步 13 应提前'}。")
    W("")
    W("判定标准（TODO 步 1）：gap > 25ms => 「prefill 里约 31ms 是 launch 开销」成立。")

    txt = os.path.join(args.out_dir, "profile_phases.txt")
    with open(txt, "w") as f:
        f.write("\n".join(lines) + "\n")
    csv_path = os.path.join(args.out_dir, "profile_phases.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["phase", "kind", "profiled", "n", "layers",
                                          "submit_ms", "tail_ms", "wall_ms"])
        w.writeheader()
        for r in RECORDS:
            w.writerow(r)

    print(f"\nwrote {txt}\nwrote {csv_path}", flush=True)
    if os.path.exists(trace_path):
        print(f"wrote {trace_path}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
