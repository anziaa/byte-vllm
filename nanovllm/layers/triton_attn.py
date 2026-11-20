"""自写 paged decode attention（步 10）。

为什么需要它：`flash_attn_with_kvcache`（2.8.3.post2）是编译好的 CUDA 黑盒，
签名里没有 k_scale / dtype / fp8 任何入口 —— KV 一旦存成 fp8 它就无法工作。
步 11 要 fp8 KV，必须先有一个能自己读 cache、自己反量化的 kernel。

判据已实测（RESULTS 8.14，C=64 / HKV=8 / HQ=16 / D=128 / block 256）：
  bf16：Tri/FA2 = 1.02 / 0.97 / 0.98（ctx=320/512/1024），两者都贴 196–220 GB/s
  fp8 ：同 kernel 换 dtype，耗时降 40–48%（字节减半近全额兑现）

所以本模块的目标不是"更快"，是**不变慢** —— 它是步 11 的入场券。
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _paged_decode_attn(
    Q, K, V, O, BT, SL, KS, VS,
    BTS: tl.constexpr, HKV: tl.constexpr, GQA: tl.constexpr, D: tl.constexpr,
    BS: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, SCALE: tl.constexpr,
    IS_FP8: tl.constexpr,
):
    """一个 program 负责 (seq, kv_head)：K/V 只读一遍，覆盖 GQA 个 q head。

    Q:   [B, HQ, D]        HQ = HKV * GQA
    K/V: [NB, BS, HKV, D]  BS = 引擎的 kvcache_block_size
    BT:  [B, BTS]          block_table（引擎里是乱序的）
    SL:  [B]               context_lens
    KS/VS: [NB, BS, HKV]   per-(token, head) 的 fp32 scale（bf16 cache 时为空张量，
                           但指针仍要传 —— Triton 不接受 None 参数）

    计算 tile 是 BN，一个存储块内切 BS // BN 个 tile。
    末块只循环有效 tile —— 否则 block padding 会凭空多读一倍字节
    （ctx=320 时实测假"慢 52%"，见 RESULTS 8.14）。
    """
    pid = tl.program_id(0)
    b = pid // HKV
    hkv = pid % HKV
    sl = tl.load(SL + b)

    offs_d = tl.arange(0, D)
    offs_m = tl.arange(0, BM)              # BM >= GQA，只有前 GQA 行有效
    q = tl.load(
        Q + b * (HKV * GQA * D) + (hkv * GQA + offs_m)[:, None] * D + offs_d[None, :],
        mask=offs_m[:, None] < GQA, other=0.0,
    )
    m_i = tl.full([BM], -float("inf"), tl.float32)
    l_i = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, D], tl.float32)

    n_blocks = tl.cdiv(sl, BS)
    for i in range(0, n_blocks):
        blk = tl.load(BT + b * BTS + i).to(tl.int64)
        n_tiles = tl.cdiv(tl.minimum(sl - i * BS, BS), BN)
        for j in range(0, n_tiles):
            offs_n = j * BN + tl.arange(0, BN)
            pos = i * BS + offs_n
            kbase = blk * (BS * HKV * D) + offs_n[:, None] * (HKV * D) + hkv * D + offs_d[None, :]
            if IS_FP8:
                # Triton 3.1：fp8 直接转 bf16 会崩，必须过 fp32（与步 4 的 w8a16 同一个坑）
                k = tl.load(K + kbase).to(tl.float32)
                sbase = blk * (BS * HKV) + offs_n * HKV + hkv
                k = (k * tl.load(KS + sbase)[:, None]).to(tl.bfloat16)
            else:
                k = tl.load(K + kbase)
            qk = tl.dot(q, tl.trans(k)) * SCALE
            qk = tl.where(pos[None, :] < sl, qk, -float("inf"))
            m_new = tl.maximum(m_i, tl.max(qk, 1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(qk - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, 1)
            if IS_FP8:
                v = tl.load(V + kbase).to(tl.float32)
                v = (v * tl.load(VS + sbase)[:, None]).to(tl.bfloat16)
            else:
                v = tl.load(V + kbase)
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
            m_i = m_new

    acc = acc / l_i[:, None]
    tl.store(
        O + b * (HKV * GQA * D) + (hkv * GQA + offs_m)[:, None] * D + offs_d[None, :],
        acc.to(O.dtype.element_ty),
        mask=offs_m[:, None] < GQA,
    )


@triton.jit
def _gather_dequant_kv(
    K, KS, V, VS, OK, OV, BT, CUK,
    BTS: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr, BS: tl.constexpr,
    TN: tl.constexpr, IS_FP8: tl.constexpr,
):
    """把 paged cache 按 block_table 反量化 gather 成 varlen 连续布局。

    prefill 的 prefix-hit 分支要把历史 KV 喂给 `flash_attn_varlen_func`，而它读不了
    fp8 paged cache。这里先摊平成 [total_k, HKV, D] 的 bf16 varlen 张量（block_table
    置 None），代价是每次命中多一次 gather 的读写，换来 prefill 路径零改动。

    一次搬 TN 个 token：块内布局是 [BS, HKV, D] 连续的，所以 TN 个连续 token 的源偏移
    在「token 优先」的扁平序里也是连续的，一个 [TN*HKV, D] 的 tile 就能一次load完。
    TN 必须整除 BS 且 p0 按 TN 对齐 —— 否则跨块。第一版一个 program 只搬 1 个 token，
    16K 个 program 全靠 launch 开销，实测多花 **14 ms**（占 163 ms prefill 的 8.7%）；
    改成 TN=8 后压到 2 ms 量级（RESULTS 8.18）。

    grid = (B, cdiv(max_seqlen_k, TN))，越界的组直接返回。
    """
    b = tl.program_id(0)
    p0 = tl.program_id(1) * TN
    kstart = tl.load(CUK + b)
    sk = tl.load(CUK + b + 1) - kstart
    if p0 >= sk:
        return
    blk = tl.load(BT + b * BTS + p0 // BS).to(tl.int64)
    base = blk * BS + (p0 % BS)                      # 组内第 0 个 token 的 slot

    offs = tl.arange(0, TN * HKV)
    tok = offs // HKV
    hh = offs % HKV
    d = tl.arange(0, D)
    slot = base + tok
    src = slot[:, None] * (HKV * D) + hh[:, None] * D + d[None, :]
    dst = (kstart + p0 + tok)[:, None] * (HKV * D) + hh[:, None] * D + d[None, :]
    m = (p0 + tok) < sk
    row_mask = m[:, None]

    if IS_FP8:
        sbase = slot * HKV + hh
        k = tl.load(K + src, mask=row_mask, other=0.0).to(tl.float32) * tl.load(KS + sbase, mask=m, other=0.0)[:, None]
        v = tl.load(V + src, mask=row_mask, other=0.0).to(tl.float32) * tl.load(VS + sbase, mask=m, other=0.0)[:, None]
        tl.store(OK + dst, k.to(tl.bfloat16), mask=row_mask)
        tl.store(OV + dst, v.to(tl.bfloat16), mask=row_mask)
    else:
        tl.store(OK + dst, tl.load(K + src, mask=row_mask), mask=row_mask)
        tl.store(OV + dst, tl.load(V + src, mask=row_mask), mask=row_mask)


_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)


def paged_decode_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
    bn: int = 32,
    num_warps: int = 4,
    num_stages: int = 1,
) -> torch.Tensor:
    """q: [B, HQ, D] → o: [B, HQ, D]。语义等价 flash_attn_with_kvcache(causal=True)。

    k_scale / v_scale：fp8 cache 时必给（[NB, BS, HKV] fp32）；bf16 cache 时忽略。
    """
    B, HQ, D = q.shape
    BS, HKV = k_cache.shape[1], k_cache.shape[2]
    assert HQ % HKV == 0, f"HQ={HQ} 不是 HKV={HKV} 的整数倍"
    gqa = HQ // HKV
    # BM 必须 >= GQA 且为 2 的幂；小 batch 下 BM=16 是 tl.dot 的下限，
    # GQA=2 只用到 2 行（算力浪费 8 倍，但 decode 是带宽 bound，实测无感）
    bm = max(16, triton.next_power_of_2(gqa))

    is_fp8 = k_cache.dtype in _FP8_DTYPES
    if k_scale is None:
        # Triton 不接受 None 参数；bf16 路径下这些指针不会被解引用
        k_scale = v_scale = k_cache
    o = torch.empty_like(q)
    _paged_decode_attn[(B * HKV,)](
        q, k_cache, v_cache, o, block_tables, context_lens, k_scale, v_scale,
        BTS=block_tables.shape[1], HKV=HKV, GQA=gqa, D=D,
        BS=BS, BM=bm, BN=bn, SCALE=scale,
        IS_FP8=is_fp8,
        num_warps=num_warps, num_stages=num_stages,
    )
    return o


def gather_kvcache(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    block_tables: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    batch: int,
    max_seqlen_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """fp8 paged cache → [total_k, HKV, D] bf16 varlen（给 flash_attn_varlen_func 用）。

    输出按 `batch × max_seqlen_k` 上界分配，**不读 `cu_seqlens_k[-1]`**：那个 `.item()`
    会一次 D2H 同步，而本函数是**每层调用一次** —— 28 次同步实测让 C=1 的命中 prefill
    从 15.9 涨到 26.4 ms（RESULTS 8.18g）。varlen 只读 `cu_seqlens` 覆盖的前 total_k 行，
    多分配的部分不会被读到，所以用上界是安全的。
    """
    _, HKV, D = k_cache.shape[-3:]
    cap = batch * max_seqlen_k
    ok = torch.empty(cap, HKV, D, dtype=torch.bfloat16, device=k_cache.device)
    ov = torch.empty(cap, HKV, D, dtype=torch.bfloat16, device=v_cache.device)
    bs = k_cache.shape[1]
    # TN 必须整除 BS，否则组内会跨块（块内才是连续的）
    tn = 8 if bs % 8 == 0 else 1
    _gather_dequant_kv[(batch, (max_seqlen_k + tn - 1) // tn)](
        k_cache, k_scale, v_cache, v_scale, ok, ov, block_tables, cu_seqlens_k,
        BTS=block_tables.shape[1], HKV=HKV, D=D, BS=bs, TN=tn,
        IS_FP8=k_cache.dtype in _FP8_DTYPES,
        num_warps=4, num_stages=1,
    )
    return ok, ov
