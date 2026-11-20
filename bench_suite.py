#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nano-vllm 基准测试套件 —— 产出可写进简历/报告的数据
=====================================================
测量三类指标, 全部按"配置 -> 一行 CSV"落盘, 方便事后画图/做表:

  1. ttft       : 单请求首 token 延迟 (max_tokens=1 的墙钟时间近似)
  2. throughput : N 并发下的输出吞吐 (tok/s) 与单请求平均延迟
  3. shared     : 共享长前缀场景下, 第 1 轮(冷) vs 第 2 轮(热) 的 TTFT 差异
                  -> 这就是 prefix caching 的量化收益

用法 (在 WSL 的 ~/nano-vllm 下, 已 source .venv/bin/activate):
    python bench_suite.py --model models/Qwen3-0.6B          # 跑完整矩阵(约 3-6 分钟)
    python bench_suite.py --model models/Qwen3-0.6B --quick  # 只跑 baseline, 先验证能跑
    python bench_suite.py --model models/Qwen3-0.6B --single '{"enforce_eager": true}' --tag eager_on

结果追加写入 ./bench_results.csv (不存在则建表头)。
"""
import argparse
import atexit
import csv
import json
import os
import statistics
import subprocess
import sys
import time

CSV_PATH = "bench_results.csv"
FIELDS = [
    "ts", "tag", "mode", "cfg",
    "concurrency", "prompt_len", "max_tokens",
    "ttft_s", "latency_s", "out_tok_s", "prefill_tok_s",
    "total_out_tokens", "peak_mem_gb", "num_kv_blocks", "block_size",
    "hit_blocks", "lookup_blocks",
    "note",
]

# ---------------------------------------------------------------- 文本池
# 用来切 token 的语料: 多语言混合, 保证 token 分布不是极端重复
_PARA = (
    "大模型推理引擎的核心矛盾是显存带宽与并行度之间的权衡。PagedAttention 把 KV cache "
    "切成固定大小的 block，用 block table 做逻辑到物理的映射，从而消除显存碎片。"
    "调度器在每个 step 里决定这一轮跑哪些请求、跑 prefill 还是 decode，并在显存不足时"
    "触发抢占与重算。CUDA graph 把一整轮 decode 的 kernel 序列录制下来回放，省掉启动开销。"
    "Prefix caching 用链式 hash 复用相同前缀的 block，让多轮对话与 few-shot 场景的首 token "
    "延迟大幅下降。Tensor parallelism 把权重按列或按行切分到多卡，用 all-reduce 汇总部分和。"
    "The throughput of a batched decoder is bounded by memory bandwidth, not FLOPs, "
    "which is why quantization and KV cache management matter more than raw compute. "
)


def build_corpus(tokenizer, min_tokens=40000):
    ids = tokenizer.encode(_PARA * 40)
    while len(ids) < min_tokens:
        ids = ids + ids
    return ids


def unique_prompt(corpus, length, idx):
    """不同请求取不同起点 + 唯一首 token, 避免意外共享前缀污染测量

    这里有两个坑，都踩过：

    1. 起点只依赖 idx 时，idx=0 对任何 length 都得到 start=0，于是
       512/2048/3584 三条 prompt 都从 corpus[0] 开始，后者前缀是前者的超集。
    2. 但只把 length 掺进起点也不够：corpus 是 _PARA * 40，token 序列本身
       有周期 P≈1664；只要 start 之差是 P 的整数倍，两段内容就完全相同。

    所以额外在开头放一个随 (length, idx) 变化的 token，让第一块的链式 hash
    必然不同。首个 token 的取值不影响 prefill 的计算量，只影响内容。
    """
    span = len(corpus) - length - 1
    start = (idx * 977 + length * 13) % span
    salt = 100 + (idx * 17 + length // 256) % 9000
    return [salt] + corpus[start: start + length - 1]


def shared_prefix_prompt(corpus, prefix_len, idx, suffix_len=16):
    """相同长前缀 + 不同短后缀 -> prefix cache 命中场景"""
    tail_start = len(corpus) - (idx + 1) * (suffix_len + 7)
    return corpus[:prefix_len] + corpus[tail_start: tail_start + suffix_len]


# ---------------------------------------------------------------- 测量
def measure_ttft(llm, tok, sampling_mod, corpus, lengths, repeat=3):
    rows = []
    for L in lengths:
        # 关键: 先跑一次同长度的抛荒请求, 把首次调用开销(cuBLAS autotune /
        # allocator 扩容 / flash-attn kernel 选择)烧掉, 否则第一个测的长度会被污染。
        # 用不同的起点构造 prompt, 避免 prefix cache 命中。
        llm.generate([unique_prompt(corpus, L, 97)],
                     sampling_mod(temperature=0.6, max_tokens=1, ignore_eos=True),
                     use_tqdm=False)
        times = []
        for r in range(repeat):
            ids = unique_prompt(corpus, L, r)
            sp = sampling_mod(temperature=0.6, max_tokens=1, ignore_eos=True)
            t0 = time.perf_counter()
            llm.generate([ids], sp, use_tqdm=False)
            times.append(time.perf_counter() - t0)
        ttft = min(times)  # 取最小值, 排除偶发抖动
        rows.append({
            "mode": "ttft", "prompt_len": L, "concurrency": 1, "max_tokens": 1,
            "ttft_s": round(ttft, 4), "latency_s": round(ttft, 4),
            "prefill_tok_s": round(L / ttft, 1),
            "out_tok_s": "", "total_out_tokens": 1,
        })
    return rows


def measure_throughput(llm, tok, sampling_mod, corpus, concurrencies, prompt_len=256, max_tokens=128):
    rows = []
    for C in concurrencies:
        prompts = [unique_prompt(corpus, prompt_len, i) for i in range(C)]
        sp = sampling_mod(temperature=0.6, max_tokens=max_tokens, ignore_eos=True)
        t0 = time.perf_counter()
        out = llm.generate(prompts, sp, use_tqdm=False)
        dt = time.perf_counter() - t0
        total_out = sum(len(o["token_ids"]) for o in out)
        rows.append({
            "mode": "throughput", "prompt_len": prompt_len, "concurrency": C,
            "max_tokens": max_tokens,
            "ttft_s": "", "latency_s": round(dt, 4),
            "out_tok_s": round(total_out / dt, 1),
            "prefill_tok_s": round((C * prompt_len) / dt, 1),
            "total_out_tokens": total_out,
        })
    return rows


def measure_throughput_runs(llm, tok, sampling_mod, corpus, concurrencies,
                            prompt_len=256, mt_list=(128,), repeat=1, warmup=True):
    """同一并发下扫多个 max_tokens, 用来把一次 generate 的墙钟拆成
    「prefill 固定段 + 每步 decode」—— dt(n) = T_prefill + n × t_step, 两点就能解。

    为什么需要这个: measure_throughput 的计时窗口**包含 prefill**。
    C=64/prompt 256 时首批就是 64×256 = 16384 token 的 prefill, 它和 decode 混在
    一个数里, 于是「int8 吞吐 +7.7%」既可能是 decode 变快, 也可能是 prefill 变慢
    被 avg 掉。扫长度能分开, 而且长 n 那几点的噪声被摊薄。

    repeat>1 时同一 prompt 会重跑 —— 必须配 enable_prefix_caching=False,
    否则第 2 次起 prefill 全部命中, T_prefill 直接变 0, 拟合就废了。
    """
    rows = []
    for C in concurrencies:
        prompts = [unique_prompt(corpus, prompt_len, i) for i in range(C)]
        if warmup:
            # 必须先烧掉「本进程第一次 C 并发」的固定开销：cuBLAS autotune、
            # allocator 扩容、以及 bs=C 的 CUDA graph 现场捕获。不烧的话
            # 第一个 max_tokens 那点会被抬高 1 s 以上，两点拟合出来的斜率能翻倍。
            # 用不同起点构造 prompt，且 prefix cache 已关，不会污染后续测量。
            llm.generate([unique_prompt(corpus, prompt_len, 4096 + i) for i in range(C)],
                         sampling_mod(temperature=0.6, max_tokens=1, ignore_eos=True),
                         use_tqdm=False)
        for mt in mt_list:
            dts = []
            total_out = 0
            for _ in range(repeat):
                sp = sampling_mod(temperature=0.6, max_tokens=mt, ignore_eos=True)
                t0 = time.perf_counter()
                out = llm.generate(prompts, sp, use_tqdm=False)
                dts.append(time.perf_counter() - t0)
                total_out = sum(len(o["token_ids"]) for o in out)
            dt = statistics.median(dts)
            rows.append({
                "mode": "throughput", "prompt_len": prompt_len, "concurrency": C,
                "max_tokens": mt,
                "ttft_s": "", "latency_s": round(dt, 4),
                "out_tok_s": round(total_out / dt, 1),
                "prefill_tok_s": round((C * prompt_len) / dt, 1),
                "total_out_tokens": total_out,
                "note": (f"reps={repeat} dts={[round(x, 4) for x in dts]}"
                         if repeat > 1 else ""),
            })
    return rows


def measure_shared_prefix(llm, tok, sampling_mod, corpus, prefix_len, rounds=3, bm=None):
    """单请求、相同长前缀、不同短后缀:
       第 1 轮(冷) vs 第 2/3 轮(热) 的 TTFT 差异 = prefix caching 的直接收益。
       注意不能让多个请求挤在同一轮, 否则第 2 个起就已经命中, 冷热就分不开了。

       改造 A 之后, bm 上多了 cache_hit_blocks / cache_lookup_blocks, 每轮取差值
       就能拿到这一轮真实的命中块数 —— 这是"命中率"而非"冷热对比"的证据。"""
    rows = []
    # 预热: 必须烧掉进程内首次调用的固定开销(cuBLAS autotune / allocator 扩容 /
    # flash-attn kernel 选择), 否则"冷"那一轮会被抬到 1.2s, 与 cache 开关无关。
    # 用一段人造 id 序列, 保证与下面的共享前缀没有任何共同块, 不污染冷启动语义。
    warm_ids = [(i % 10000) + 100 for i in range(prefix_len)]
    llm.generate([warm_ids],
                 sampling_mod(temperature=0.6, max_tokens=1, ignore_eos=True), use_tqdm=False)
    for rnd in range(rounds):
        base_hit = getattr(bm, "cache_hit_blocks", 0)
        base_look = getattr(bm, "cache_lookup_blocks", 0)
        ids = shared_prefix_prompt(corpus, prefix_len, rnd)
        sp = sampling_mod(temperature=0.6, max_tokens=1, ignore_eos=True)
        t0 = time.perf_counter()
        llm.generate([ids], sp, use_tqdm=False)
        dt = time.perf_counter() - t0
        d_hit = getattr(bm, "cache_hit_blocks", 0) - base_hit
        d_look = getattr(bm, "cache_lookup_blocks", 0) - base_look
        n_blk = d_look if d_look else (prefix_len // 256)
        rows.append({
            "mode": "shared", "prompt_len": prefix_len, "concurrency": 1, "max_tokens": 1,
            "ttft_s": round(dt, 4), "latency_s": round(dt, 4),
            "prefill_tok_s": round(prefix_len / dt, 1),
            "out_tok_s": "", "total_out_tokens": 1,
            "hit_blocks": d_hit, "lookup_blocks": d_look,
            "note": f"cold (hit {d_hit}/{n_blk})" if rnd == 0
                    else f"warm{rnd} (hit {d_hit}/{n_blk})",
        })
    return rows


# ---------------------------------------------------------------- 单配置模式
def run_single(model, cfg, tag, quick, csv_path=CSV_PATH, ttft_only=False, shared_only=False,
               throughput_only=False, conc=(1, 8, 32, 64), prompt_len=256,
               mt_list=(128,), repeat=1):
    import torch
    from nanovllm import LLM
    from nanovllm.sampling_params import SamplingParams

    print(f"\n===== [{tag}] cfg={cfg} =====", flush=True)
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    t_load = time.perf_counter()
    llm = LLM(model, **cfg)
    print(f"加载耗时 {time.perf_counter() - t_load:.1f}s", flush=True)

    tok = llm.tokenizer
    corpus = build_corpus(tok)
    bm = llm.scheduler.block_manager
    num_kv_blocks = len(bm.blocks)
    block_size = bm.block_size
    print(f"KV blocks={num_kv_blocks} block_size={block_size} "
          f"容量≈{num_kv_blocks * block_size / 1024:.0f}K tokens", flush=True)

    rows = []
    try:
        if quick:
            rows += measure_ttft(llm, tok, SamplingParams, corpus, [1024], repeat=1)
            rows += measure_throughput(llm, tok, SamplingParams, corpus, [8], prompt_len=256, max_tokens=64)
        elif ttft_only:
            rows += measure_ttft(llm, tok, SamplingParams, corpus, [512, 2048, 3584])
        elif shared_only:
            # 改造 A 的对照实验: 只跑共享前缀场景, 一条命令 20 秒出数
            rows += measure_shared_prefix(llm, tok, SamplingParams, corpus,
                                          prefix_len=2048, rounds=5, bm=bm)
        elif throughput_only:
            # 步 2 归因实验: 只跑吞吐, 且同一并发扫多个 max_tokens 拆 prefill / decode
            rows += measure_throughput_runs(llm, tok, SamplingParams, corpus,
                                            list(conc), prompt_len=prompt_len,
                                            mt_list=mt_list, repeat=repeat)
        else:
            rows += measure_ttft(llm, tok, SamplingParams, corpus, [512, 2048, 3584])
            rows += measure_throughput(llm, tok, SamplingParams, corpus, [1, 8, 32, 64])
            rows += measure_shared_prefix(llm, tok, SamplingParams, corpus,
                                          prefix_len=2048, rounds=5, bm=bm)
    finally:
        try:
            llm.exit()
        except Exception:
            pass
        # LLMEngine.__init__ 里已经 atexit.register(self.exit) 过一次,
        # 手动 exit() 后必须注销, 否则进程退出时二次调用 ->
        # AttributeError: 'LLM' object has no attribute 'model_runner'
        try:
            atexit.unregister(llm.exit)
        except Exception:
            pass

    peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    out_rows = []
    for r in rows:
        row = {
            "ts": ts, "tag": tag, "cfg": json.dumps(cfg, ensure_ascii=False),
            "peak_mem_gb": round(peak, 2), "num_kv_blocks": num_kv_blocks,
            "block_size": block_size, "note": r.pop("note", ""),
        }
        row.update({k: r.get(k, "") for k in FIELDS if k not in row})
        out_rows.append(row)

    new_file = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        for row in out_rows:
            w.writerow({k: row.get(k, "") for k in FIELDS})

    print("\n结果:")
    print(f"{'mode':<11}{'conc':>5}{'plen':>6}{'ttft_s':>9}{'lat_s':>9}"
          f"{'out_tok/s':>11}{'prefill_tok/s':>14}{'cache hit':>12}")
    for r in out_rows:
        lb = r.get("lookup_blocks", "")
        hit = f"{r.get('hit_blocks')}/{lb}" if str(lb) not in ("", "0") else "-"
        print(f"{r['mode']:<11}{str(r['concurrency']):>5}{str(r['prompt_len']):>6}"
              f"{str(r['ttft_s']):>9}{str(r['latency_s']):>9}{str(r['out_tok_s']):>11}"
              f"{str(r['prefill_tok_s']):>14}{hit:>12}")

    # 改造 A 的观测点: prefix caching 是开是关, 命中率多少
    # 只统计 shared 行, 不要用 bm 的进程级累计值 —— 预热与其它场景的 lookup 会混进分母
    sh_rows = [r for r in out_rows if r["mode"] == "shared"]
    if sh_rows:
        th = sum(int(r["hit_blocks"]) for r in sh_rows)
        tl = sum(int(r["lookup_blocks"]) for r in sh_rows)
        if getattr(bm, "enable_prefix_caching", None) is False:
            print(f"prefix cache: 已关闭 (enable_prefix_caching=False) · "
                  f"shared 场景 {th}/{tl} 命中", flush=True)
        else:
            steady = [r for r in sh_rows if not str(r["note"]).startswith("cold")]
            sh_h = sum(int(r["hit_blocks"]) for r in steady)
            sh_l = sum(int(r["lookup_blocks"]) for r in steady)
            rate = sh_h / sh_l * 100 if sh_l else 0.0
            print(f"prefix cache: shared 场景命中 {th}/{tl} = {th/tl*100:.1f}% "
                  f"(含冷启动) | 稳态 {sh_h}/{sh_l} = {rate:.1f}%", flush=True)
    print(f"-> 已追加到 {os.path.abspath(csv_path)}")
    return out_rows


# ---------------------------------------------------------------- 矩阵模式
BASE = {"kvcache_block_size": 256, "max_num_seqs": 512,
        "gpu_memory_utilization": 0.85, "max_model_len": 4096}


def default_matrix():
    base = dict(BASE)
    return [
        ("baseline",        dict(base, enforce_eager=False)),
        ("eager_on",        dict(base, enforce_eager=True)),
        ("block_512",       dict(base, kvcache_block_size=512)),
        ("seqs_64",         dict(base, max_num_seqs=64)),
        ("seqs_16",         dict(base, max_num_seqs=16)),
        ("mem_0.75",        dict(base, gpu_memory_utilization=0.75)),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--single", help='单个配置的 JSON, 如 \'{"enforce_eager": true}\'')
    ap.add_argument("--tag", default="run")
    ap.add_argument("--quick", action="store_true", help="只跑最小集合, 用于验证脚本可用")
    ap.add_argument("--ttft-only", action="store_true", help="只测 TTFT, 不测吞吐(改完脚本后补数用)")
    ap.add_argument("--shared-only", action="store_true",
                    help="只测共享前缀场景(改造 A 的 cache 开关对照实验, 约 20 秒)")
    ap.add_argument("--csv", default=CSV_PATH)
    ap.add_argument("--quant", default=None, choices=["none", "lm_head", "all", "int4_head", "int4_all", "int4_mlp"],
                    help="步 4：int8 量化配置（不想用 --single 塞 JSON 时走这个）")
    ap.add_argument("--int4-group", type=int, default=None, help="int4 的 group size，0=per-channel")
    ap.add_argument("--keep-bf16", type=int, default=None,
                    help="1/0：prefill 是否保留 bf16 权重（仅 quant=all 有意义）")
    ap.add_argument("--matrix", help="自定义矩阵: JSON list of [tag, cfg]")
    ap.add_argument("--prefill-compile", default=None, choices=["off", "ref", "layer", "model"],
                    help="步 2：prefill 编译档位（不传则用 Config 默认 model）")
    ap.add_argument("--attn-backend", default=None, choices=["fa2", "triton"],
                    help="步 10：decode attention 后端（fa2 默认 / triton 自写 paged）")
    ap.add_argument("--kv-cache-dtype", default=None, choices=["bf16", "fp8"],
                    help="步 11：KV cache 存储格式（fp8 需 attn-backend=triton，会自动带）")
    ap.add_argument("--prefix-caching", type=int, default=None, choices=[0, 1],
                    help="enable_prefix_caching，扫长度时必须 0，否则第 2 次起 prefill 全命中")
    ap.add_argument("--throughput-only", action="store_true",
                    help="只跑吞吐（配合 --conc/--prompt-len/--max-tokens-list）")
    ap.add_argument("--conc", default="1,8,32,64", help="并发列表, 逗号分隔")
    ap.add_argument("--prompt-len", type=int, default=256)
    ap.add_argument("--max-tokens-list", default="128", help="输出长度列表, 逗号分隔（拆 prefill/decode 用）")
    ap.add_argument("--repeat", type=int, default=1, help="同一配置重复次数, 取中位（需配 --prefix-caching 0）")
    ap.add_argument("--_child", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    conc = tuple(int(x) for x in args.conc.split(",") if x.strip())
    mt_list = tuple(int(x) for x in str(args.max_tokens_list).split(",") if x.strip())

    def _merge(cfg):
        if args.quant:
            # 与 default_matrix 的 baseline 同口径（mem 0.85 而非默认 0.9），
            # 否则 throughput 比的不是量化，是 KV 池大小
            cfg = dict(BASE, **cfg)
            cfg["quant"] = args.quant
        if args.keep_bf16 is not None:
            cfg["int8_keep_bf16"] = bool(args.keep_bf16)
        if args.int4_group is not None:
            cfg["int4_group"] = int(args.int4_group)
        if args.prefill_compile is not None:
            cfg["prefill_compile"] = args.prefill_compile
        if args.prefix_caching is not None:
            cfg["enable_prefix_caching"] = bool(args.prefix_caching)
        if args.attn_backend is not None:
            cfg["attn_backend"] = args.attn_backend
        if args.kv_cache_dtype is not None:
            cfg["kv_cache_dtype"] = args.kv_cache_dtype
            if args.kv_cache_dtype == "fp8" and "attn_backend" not in cfg:
                cfg["attn_backend"] = "triton"   # FA2 读不了 fp8 cache
        return cfg

    # 子进程模式: 真正跑测量
    if args._child:
        cfg = _merge(json.loads(args.single))
        run_single(args.model, cfg, args.tag, args.quick, args.csv, args.ttft_only, args.shared_only,
                   args.throughput_only, conc, args.prompt_len, mt_list, args.repeat)
        return

    # 父进程模式: 逐个配置起独立进程
    # 必须独立进程 —— CUDA graph / NCCL 状态在同一进程里反复初始化容易踩坑
    if args.single or args.quant:
        cfg = _merge(json.loads(args.single) if args.single else {})
        run_single(args.model, cfg, args.tag, args.quick, args.csv,
                   args.ttft_only, args.shared_only,
                   args.throughput_only, conc, args.prompt_len, mt_list, args.repeat)
        return

    matrix = json.loads(args.matrix) if args.matrix else default_matrix()
    if args.quick:
        matrix = matrix[:1]
    print(f"将依次测试 {len(matrix)} 个配置, 结果写入 {args.csv}")
    for tag, cfg in matrix:
        cmd = [sys.executable, os.path.abspath(__file__),
               "--model", args.model, "--single", json.dumps(cfg),
               "--tag", tag, "--csv", args.csv, "--_child"]
        if args.quick:
            cmd.append("--quick")
        if args.ttft_only:
            cmd.append("--ttft-only")
        if args.shared_only:
            cmd.append("--shared-only")
        if args.throughput_only:
            cmd.append("--throughput-only")
            cmd += ["--conc", args.conc, "--prompt-len", str(args.prompt_len),
                    "--max-tokens-list", args.max_tokens_list, "--repeat", str(args.repeat)]
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print(f"[!] 配置 {tag} 失败 (returncode={r.returncode}), 已跳过, 继续下一个")
    print("\n全部完成, 打开 bench_results.csv 查看 / 直接发给我, 我帮你出图表和结论。")


if __name__ == "__main__":
    main()
