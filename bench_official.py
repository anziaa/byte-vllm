"""复刻 nano-vllm 官方 README 的 benchmark 协议, 用于同配置对标。

官方数字 (README, 2026-04-26 提交, RTX 4070 Laptop 8GB):
    vLLM       133,966 tok / 98.37 s  = 1361.84 tok/s
    nano-vllm  133,966 tok / 93.41 s  = 1434.13 tok/s

关键: 这不是"并发 256"。256 是请求总数, 一次性提交后由调度器按 KV 池容量分批。
      Qwen3-0.6B 在 8GB 卡上 KV 池约 37,376 tokens (146 blocks x 256),
      平均上下文 ~820 tokens -> 实际同时在跑的序列只有 ~46 条。

用法:
    python bench_official.py --model models/Qwen3-0.6B
"""
import argparse
import atexit
import os
import time
from random import randint, seed

from nanovllm import LLM, SamplingParams

OFFICIAL = {"nano-vllm": 1434.13, "vllm": 1361.84}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen3-0.6B")
    ap.add_argument("--num-seqs", type=int, default=256, help="官方协议为 256")
    ap.add_argument("--seed", type=int, default=0, help="官方为 0, 决定长度分布")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--enforce-eager", action="store_true")
    # 步 2/4 的开关：官方协议是"跑分"，必须能把每一步的优化单独关掉才叫对照
    ap.add_argument("--quant", default="none",
                    choices=["none", "lm_head", "all", "int4_head", "int4_all", "int4_mlp"])
    ap.add_argument("--keep-bf16", type=int, default=1)
    ap.add_argument("--int4-group", type=int, default=0)
    ap.add_argument("--prefix-caching", type=int, default=1)
    ap.add_argument("--prefill-compile", default="model", choices=["off", "layer", "model"])
    ap.add_argument("--mem-util", type=float, default=0.9, help="官方默认 0.9")
    ap.add_argument("--tag", default="")
    ap.add_argument("--repeat", type=int, default=1, help="同进程内重复次数（取最好/均值）")
    a = ap.parse_args()

    seed(a.seed)
    # 严格复刻 bench.py 的随机调用顺序(内层 range 参数先消耗随机数),
    # 否则长度分布会变, 总输出 token 对不上 133,966
    prompts = [[randint(0, 10000) for _ in range(randint(100, 1024))] for _ in range(a.num_seqs)]
    sps = [SamplingParams(temperature=0.6, max_tokens=randint(100, 1024), ignore_eos=True)
           for _ in range(a.num_seqs)]

    n_in = sum(len(p) for p in prompts)
    n_out = sum(s.max_tokens for s in sps)
    avg_ctx = n_in / a.num_seqs + n_out / a.num_seqs / 2

    print(f"===== 官方协议复现 =====")
    if a.tag:
        print(f"档位        {a.tag}  (quant={a.quant} keep_bf16={a.keep_bf16} prefix={a.prefix_caching} compile={a.prefill_compile})")
    print(f"模型        {a.model}")
    print(f"请求数      {a.num_seqs}")
    print(f"输入        平均 {n_in/a.num_seqs:.1f} / 总计 {n_in:,} tokens")
    print(f"输出        平均 {n_out/a.num_seqs:.1f} / 总计 {n_out:,} tokens")
    print(f"平均上下文  {avg_ctx:.0f} tokens (decode 期)\n", flush=True)

    path = os.path.expanduser(a.model)
    llm = LLM(path,
              enforce_eager=a.enforce_eager,
              max_model_len=a.max_model_len,
              gpu_memory_utilization=a.mem_util,
              enable_prefix_caching=bool(a.prefix_caching),
              prefill_compile=a.prefill_compile,
              quant=a.quant,
              int8_keep_bf16=bool(a.keep_bf16),
              int4_group=a.int4_group)
    try:
        atexit.unregister(llm.exit)
    except Exception:
        pass

    blocks, bsize = len(llm.scheduler.block_manager.blocks), llm.scheduler.block_manager.block_size
    kv_cap = blocks * bsize
    print(f"KV 池       {blocks} blocks x {bsize} = {kv_cap:,} tokens")
    print(f"同时可跑    ~{kv_cap / avg_ctx:.0f} 条 (256 条全并行需 {a.num_seqs*avg_ctx/1000:.0f}K, 是池子的 {a.num_seqs*avg_ctx/kv_cap:.1f} 倍)\n", flush=True)

    llm.generate(["Benchmark: "], SamplingParams())  # warmup, 官方同款
    # 重复时重建 SamplingParams：generate 内部会改 sp.max_tokens，复用会改变负载
    mt = [sp.max_tokens for sp in sps]
    dts = []
    for i in range(a.repeat):
        cur = [SamplingParams(temperature=0.6, max_tokens=m, ignore_eos=True) for m in mt]
        t0 = time.perf_counter()
        llm.generate(prompts, cur, use_tqdm=False)
        dt = time.perf_counter() - t0
        dts.append(dt)
        print(f"[run {i+1}] 耗时 {dt:.2f} s  吞吐 {n_out/dt:.2f} tok/s", flush=True)
    dt = min(dts)
    tput = n_out / dt

    print(f"耗时        {dt:.2f} s")
    print(f"吞吐        {tput:.2f} tok/s   (best of {a.repeat})")
    print()
    print("对照 (官方 README, 同硬件):")
    print(f"  nano-vllm {OFFICIAL['nano-vllm']:9.2f} tok/s   本机 {tput:9.2f}   差异 {(tput/OFFICIAL['nano-vllm']-1)*100:+.1f}%")
    print(f"  vLLM      {OFFICIAL['vllm']:9.2f} tok/s   (未安装, 仅作参照)")
    print()
    print(f"若与 baseline C=64 的 2994.6 tok/s 对比, 必须先归一化上下文:")
    print(f"  官方协议平均上下文 {avg_ctx:.0f} vs 本机 baseline 320 -> 每 token 边际成本 x{avg_ctx/320:.2f}")


if __name__ == "__main__":
    main()
