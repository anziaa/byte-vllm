"""步 4：weight-only int8（w8a16）的 kernel 与量化工具。

为什么不用 `aten::_weight_int8pack_mm`：torch 2.5.1 只给它注册了 CPU kernel，
CUDA 上直接 NotImplementedError（`_scaled_mm` 在 sm89 上也不吃 int8，报
"Expected mat1 to be Float8 matrix got Char"）。所以自己写 Triton kernel：
从显存读 int8 权重，在寄存器里反量化成 bf16，再走 mma。省的是**权重字节**，
激活保持 bf16 —— 这是 weight-only 的语义，精度损失远小于 W8A8。

微基准（`scripts/micro_int8_gemv.py`，L2 已 flush，vs bf16 F.linear）：

    shape      M=1..64（decode）   M>=256（prefill）
    qkv        1.73-1.80x          0.74x
    gate_up    1.81-1.92x          0.72x
    down       1.14-1.19x          0.45x
    lm_head    1.79-1.85x          0.73x

→ decode 稳赚、prefill 亏，所以 prefill 是否走 int8 必须由调用方决定，
  见 `Int8Mixin.use_int8_now()`。
"""
import os

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# ---------------------------------------------------------------- kernel

@triton.jit
def _w8a16_kernel(
    x_ptr, w_ptr, s_ptr, y_ptr,
    M, N, K,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    """y[M,N] = x[M,K] @ dequant(w[K,N])，w 是 int8 + per-N 的 scale。

    w 按 (K, N) 存：tile 是 (BK, BN)，沿 N 连续，coalescing 正常。

    坑：反量化不能写成 `w.to(tl.bfloat16) * s[None, :]` 直接喂 dot ——
    Triton 3.1 会在 LLVM 侧断言崩溃（SmallVector idx < size()）。中间过一层
    fp32 就正常，而且少一次 layout 转换。
    """
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    m_mask = rm < M

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    s = tl.load(s_ptr + rn).to(tl.float32)
    for kk in range(0, tl.cdiv(K, BK)):
        x = tl.load(x_ptr + rm[:, None] * K + (kk * BK + rk)[None, :],
                    mask=m_mask[:, None], other=0.0)
        w = tl.load(w_ptr + (kk * BK + rk)[:, None] * N + rn[None, :])
        wf = (w.to(tl.float32) * s[None, :]).to(tl.bfloat16)
        acc = tl.dot(x, wf, acc)
    tl.store(y_ptr + rm[:, None] * N + rn[None, :], acc.to(tl.bfloat16),
             mask=m_mask[:, None])


@triton.jit
def _w4a16_kernel(
    x_ptr, w_ptr, s_ptr, y_ptr,
    M, N, K,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    G: tl.constexpr = 0,
):
    """y[M,N] = x[M,K] @ dequant(w[K,N])，w 是 int4 打包成 uint8 的 [K/2, N]。

    打包约定：一个 byte 存沿 K 相邻的两个 nibble，低 4 位 = k，高 4 位 = k+1，
    值域 0..15 对应有符号 -8..7（加 8 偏移）。nibble 沿 K 打包而不是沿 N，
    是因为 scale 是 per-N 的 —— 沿 N 打包会让同一个 byte 里的两个值共享 scale，
    反而要拆开算，得不偿失。

    解包：`tl.join(lo, hi)` 得到 (BK/2, BN, 2)，**必须先 permute 成 (BK/2, 2, BN)
    再 reshape**。直接 reshape 到 (BK, BN) 是错的 —— join 加在最后一维，row-major
    展平后 nibble 会交错进 N 维（实测 rel_L2 141%，一眼能看出来）。
    permute 之后 row = 2i + t，正好对应 k = 2i（lo）和 2i+1（hi）。
    """
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    rk2 = tl.arange(0, BK // 2)
    m_mask = rm < M

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    if G == 0:                                   # per-output-channel：整行一个 scale
        s = tl.load(s_ptr + rn).to(tl.float32)
    for kk in range(0, tl.cdiv(K, BK)):
        if G != 0:
            # group-wise：块内所有 k 必须同属一个 group，所以要求 BK <= G 且块对齐
            s = tl.load(s_ptr + ((kk * BK) // G) * N + rn).to(tl.float32)
        x = tl.load(x_ptr + rm[:, None] * K + (kk * BK + rk)[None, :],
                    mask=m_mask[:, None], other=0.0)
        wp = tl.load(w_ptr + (kk * (BK // 2) + rk2)[:, None] * N + rn[None, :])
        lo = (wp & 0xF)
        hi = ((wp >> 4) & 0xF)
        wq = tl.reshape(tl.permute(tl.join(lo, hi), (0, 2, 1)), (BK, BN)).to(tl.float32)
        wf = ((wq - 8.0) * s[None, :]).to(tl.bfloat16)
        acc = tl.dot(x, wf, acc)
    tl.store(y_ptr + rm[:, None] * N + rn[None, :], acc.to(tl.bfloat16),
             mask=m_mask[:, None])


def _pick_tiles(M: int, N: int):
    """tile 表来自 scripts/autotune_w8a16.py（28 份不同权重连续发，模拟真实 decode）。

    两条硬约束 + 一条实测偏好：
      - BM 必须覆盖 M：固定 BM=16 时 M=64 会有 4 个 pid_m，权重被重复读 4 遍
        （lm_head M=64 从 1.84x 掉到 0.52x）。
      - N 小的（down, N=1024）在 decode 必须把 BN 压到 32 换并行度：BN=128 只有 8 个
        program，填不满 36 个 SM（空 28 个），M=32 时 42.8 µs vs BN=32 的 19.9 µs（2.15x）。
      - N 大的（qkv/gate_up）BK 取 32 比 64 好：每次 load 连续 32 列正好两个
        DRAM sector，BM 的浪费不被放大。

    | shape   | 实测最优 (BN,BK,w,s) | M=1 µs | M=32 µs |
    | qkv     | 64,32,4,3            | 22.5   | 23.7    |
    | gate_up | 64,32,4,3            | 31.1   | 32.9    |
    | down    | 32,128,4,3           | 19.2   | 19.9    |
    | lm_head | 128,32,4,2           | 637    | 690     |

    **prefill 分支（N≤2048, M≥256）另有一套最优值**，和 decode 的诉求正好相反：
    decode 要 BN=32 换 32 个 pid_n 填满 36 个 SM；prefill 的 M 大、pid_m 已经够多，
    反而要 BM×BN 更大（program 数减半 → 每块固定开销摊薄）＋ BK 减半（smem/级
    48→24 KB，流水从 2 级回到 3 级，能盖住访存延迟）。M=16384 实测 down
    3947→2736 µs、o_proj 2711→1812 µs（RESULTS 12.9）。

    转折点实测在 **M=256**（scripts/scan_tile_threshold.py，M=128 时仍是 32,128
    更快：down 55.3 vs 77.8 µs；到 256 反转：79.9 vs 99.3）。M≤128 一律走老表，
    所以 decode 完全不受影响。
    """
    BM = 16 if M <= 16 else (32 if M <= 32 else (64 if M <= 64 else 128))
    if N <= 2048:
        if M >= 256:
            return BM, 64, 64, 4, 3
        return BM, 32, 128, 4, 3
    if N >= 65536:
        return BM, 128, 32, 4, 2
    return BM, 64, 32, 4, 3


def w8a16_mm(x: torch.Tensor, w_kn: torch.Tensor, scale_n: torch.Tensor) -> torch.Tensor:
    """x (M,K) bf16；w_kn (K,N) int8；scale_n (N,)。"""
    assert x.dim() == 2, f"w8a16_mm 只接受 2D 激活，拿到 {tuple(x.shape)}"
    if not x.is_contiguous():
        x = x.contiguous()
    M, K = x.shape
    N = w_kn.shape[1]
    y = torch.empty((M, N), device=x.device, dtype=x.dtype)
    BM, BN, BK, warps, stages = _pick_tiles(M, N)
    grid = (triton.cdiv(N, BN), triton.cdiv(M, BM))
    _w8a16_kernel[grid](x, w_kn, scale_n, y, M, N, K,
                        BM=BM, BN=BN, BK=BK, num_warps=warps, num_stages=stages)
    return y


def w4a16_mm(x: torch.Tensor, w_kn: torch.Tensor, scale_n: torch.Tensor,
             group_size: int = 0) -> torch.Tensor:
    """x (M,K) bf16；w_kn (K/2,N) uint8（nibble 打包）。

    scale_n：group_size=0 时是 (N,) 的 per-channel scale；否则是 (K/G, N)。
    """
    assert x.dim() == 2, f"w4a16_mm 只接受 2D 激活，拿到 {tuple(x.shape)}"
    if not x.is_contiguous():
        x = x.contiguous()
    M, K = x.shape
    N = w_kn.shape[1]
    assert w_kn.shape[0] * 2 == K, f"packed 行数 {w_kn.shape[0]} 与 K={K} 不匹配"
    y = torch.empty((M, N), device=x.device, dtype=x.dtype)
    BM, BN, BK, warps, stages = _pick_tiles(M, N)
    if group_size:
        # 块内必须同 group；BK 超过 G 就会跨组，直接钳到 G
        BK = min(BK, group_size)
    grid = (triton.cdiv(N, BN), triton.cdiv(M, BM))
    _w4a16_kernel[grid](x, w_kn, scale_n, y, M, N, K,
                        BM=BM, BN=BN, BK=BK, num_warps=warps, num_stages=stages,
                        G=group_size)
    return y


@torch.library.custom_op("nanovllm::int4_w4a16", mutates_args=())
def int4_w4a16(x: torch.Tensor, w_kn: torch.Tensor, scale_n: torch.Tensor,
               group_size: int = 0) -> torch.Tensor:
    return w4a16_mm(x, w_kn, scale_n, group_size)


@int4_w4a16.register_fake
def _(x: torch.Tensor, w_kn: torch.Tensor, scale_n: torch.Tensor,
      group_size: int = 0) -> torch.Tensor:
    return x.new_empty((x.shape[0], w_kn.shape[1]))


# prefill 要进 torch.compile 的图，裸 triton.jit 不能直接内联，包成 custom op
# （同步 2 的 prefill_attn 模式）。mutates_args 留空：输出是新张量。
@torch.library.custom_op("nanovllm::int8_w8a16", mutates_args=())
def int8_w8a16(x: torch.Tensor, w_kn: torch.Tensor, scale_n: torch.Tensor) -> torch.Tensor:
    return w8a16_mm(x, w_kn, scale_n)


@int8_w8a16.register_fake
def _(x: torch.Tensor, w_kn: torch.Tensor, scale_n: torch.Tensor) -> torch.Tensor:
    return x.new_empty((x.shape[0], w_kn.shape[1]))


# ---------------------------------------------------------------- 量化

def quantize_weight_rowwise(w: torch.Tensor, chunk: int = 1024):
    """(N,K) bf16 → (K,N) int8 + (N,) bf16 scale，per-output-channel 对称量化。

    转置存成 (K,N) 是为了 kernel 里 (BK,BN) tile 沿 N 连续。

    **按行分块**（RESULTS 13）：整块算的时候，`w.float()` 对 lm_head（151936×1024）
    会一次性物化 622 MB fp32，把 allocator 的 reserved 顶到 3.2 GB；释放后留下
    0.675 GB 碎片还不回去（`gc` + `empty_cache` 都无效），直接从 KV 池里扣掉 12 块。
    每行独立计算 amax / scale / round，所以分块与整块**逐位一致**，只是峰值降到
    2 MB 量级（chunk=1024 行 × 1024 列 fp32）。
    """
    N, K = w.shape
    q = torch.empty((K, N), dtype=torch.int8, device=w.device)
    scale = torch.empty(N, dtype=torch.bfloat16, device=w.device)
    for i in range(0, N, chunk):
        sub = w[i:i + chunk]                                    # view，零拷贝
        amax = sub.abs().amax(dim=1).clamp_min(1e-8)
        s = (amax / 127.0).to(torch.bfloat16)
        qi = torch.round(sub.float() / s.float()[:, None]).clamp_(-127, 127).to(torch.int8)
        q[:, i:i + chunk] = qi.t()
        scale[i:i + chunk] = s
    return q, scale


def quantize_weight_int4(w: torch.Tensor, group_size: int = 0):
    """(N,K) bf16 → (K/2,N) uint8 + scale。

    group_size=0 → per-output-channel，(N,) scale；int4 值域 -8..7，步长是 int8 的
    127/7 ≈ 18 倍，per-channel 下 rel_L2 约 15%（int8 是 0.85%）。
    group_size=G → 沿 K 每 G 个元素共享一个 scale，返回 (K/G, N)。组内动态范围小
    得多，这是把 int4 的 ppl 代价压回可用区间的常规手段。
    """
    assert w.shape[1] % 2 == 0, f"K={w.shape[1]} 必须是偶数才能沿 K 打包"
    N, K = w.shape
    if group_size:
        assert K % group_size == 0, f"K={K} 不能被 group_size={group_size} 整除"
        wg = w.float().view(N, K // group_size, group_size)
        amax = wg.abs().amax(dim=2).clamp_min(1e-8)
        scale = (amax / 7.0).to(torch.bfloat16)                    # (N, K/G)
        q = torch.round(wg / scale.float()[:, :, None]).clamp_(-8, 7)
        q = (q + 8).to(torch.uint8).view(N, K).t().contiguous()     # (K,N)
        scale = scale.t().contiguous()                              # (K/G, N)
    else:
        amax = w.abs().amax(dim=1).clamp_min(1e-8)
        scale = (amax / 7.0).to(torch.bfloat16)                     # (N,)
        q = torch.round(w.float() / scale.float()[:, None]).clamp_(-8, 7)
        q = (q + 8).to(torch.uint8).t().contiguous()
    lo, hi = q[0::2], q[1::2]
    packed = (lo.to(torch.int16) | (hi.to(torch.int16) << 4)).to(torch.uint8)
    return packed.contiguous(), scale


# ---------------------------------------------------------------- mixin

_CFG = {"keep_bf16": True, "int4_group": 0, "fp8_act": "off",
        "fp8_act_skip_head": True}
FP8_MAX = 448.0


def set_keep_bf16(v: bool):
    """全局开关：是否给 prefill 保留 bf16 权重副本。由 ModelRunner 按 config 设置。"""
    _CFG["keep_bf16"] = bool(v)


def set_int4_group(v: int):
    """int4 的 group size（0 = per-output-channel）。由 ModelRunner 按 config 设置。"""
    _CFG["int4_group"] = int(v)


def set_fp8_act(v: str, skip_head: bool = True):
    """步 12 预演：激活假量化档位（off / tensor / token / int8）。

    skip_head=True 时**跳过 lm_head**（`tie_target` 标记）：它是最敏感的一块，
    而在 prefill 里只有 1.2× —— 为 0.2 倍的收益去赌 logits 精度不划算。
    """
    _CFG["fp8_act"] = str(v)
    _CFG["fp8_act_skip_head"] = bool(skip_head)


def fake_fp8_act(x: torch.Tensor, mode: str = "token") -> torch.Tensor:
    """把激活做一次 e4m3 往返，**dtype 仍是 bf16**。

    与 `Config.fp8_kv="fake"` 同一套思路：真 kernel 还没写，先让数值带上 fp8 的
    精度损失，用 ppl 判断这条路值不值得做（RESULTS 8.18d 就是这么预演 KV 的）。

    tensor：整块一个 scale —— torch 2.5.1 的 `_scaled_mm` 唯一支持的粒度
    token ：每行一个 scale —— 真 kernel 打算用的粒度（激活有离群值时差别很大）
    """
    if mode == "int8":
        # 步 12 的候选：int8 有 127 级（e4m3 每档只有 3 位尾数），精度更好，
        # 而 Ada 的 INT8 张量核与 FP8 同为 bf16 的 2×。
        s = x.abs().amax(-1, keepdim=True).clamp_min(1e-8) / 127.0
        q = torch.round(x.float() / s).clamp_(-127, 127).to(torch.int8)
        return q.to(x.dtype) * s
    if mode == "tensor":
        s = x.abs().amax().clamp_min(1e-8) / FP8_MAX
    else:
        s = x.abs().amax(-1, keepdim=True).clamp_min(1e-8) / FP8_MAX
    q = (x / s).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q.to(x.dtype) * s


class Int8Mixin:
    """给 LinearBase / VocabParallelEmbedding 这类"权重矩阵 + F.linear"的模块
    挂上 int8 能力。加载阶段完全不变（weight 仍是 bf16，weight_loader 照旧按
    tp 分片），`quantize_()` 之后才有 int8 张量。

    keep_bf16=True：保留 bf16 权重只给 prefill 用。
      prefill 是 compute-bound 大 M（微基准 M>=256 掉到 0.7x），decode 是
      bandwidth-bound 小 M（稳拿 1.8x）。代价是多一份 bf16 显存，换来 prefill
      不退步 —— 是否划算由端到端测出来决定，不靠猜。
    """

    def _init_int8(self, keep_bf16: bool | None = None, bits: int = 8):
        self.bits = bits                 # 8 或 4，由 ModelRunner.quantize_model 按档位设
        self.group_size = _CFG["int4_group"]
        self.keep_bf16 = _CFG["keep_bf16"] if keep_bf16 is None else keep_bf16
        self.w_int8 = None               # 名字沿用；int4 时是 (K/2,N) 的打包 uint8
        self.w_scale = None
        self.quantized = False
        self.int8_active = True        # 由 Qwen3ForCausalLM.set_int8_active 按 prefill/decode 切

    def quantize_(self, bits: int | None = None):
        """就地压缩。返回省下的字节数（GB）。"""
        if self.quantized:
            return 0.0
        if bits is not None:
            self.bits = bits
        w = self.weight.data
        n_before = w.numel() * w.element_size()
        # tie_word_embeddings 时 lm_head.weight 与 embed_tokens.weight 是同一块内存。
        # 只有 keep_bf16（要把 bf16 副本留下来）才需要 clone 断共享；
        # 纯量化路径下面会把 self.weight 置 None、量化又是只读的，克隆一份 311 MB
        # 纯属给自己制造碎片（RESULTS 13：这一份 clone 值 3 块 KV）。
        shared = getattr(self, "tie_target", None) is not None and \
            w.data_ptr() == self.tie_target.data_ptr()
        if shared and self.keep_bf16:
            w = w.clone()
            self.weight.data = w
        if self.bits == 4:
            self.w_int8, self.w_scale = quantize_weight_int4(w, self.group_size)
        else:
            self.w_int8, self.w_scale = quantize_weight_rowwise(
                w, int(os.environ.get("NANOVLLM_QUANT_CHUNK", 1024)))
        self.quantized = True
        n_after = self.w_int8.numel() + self.w_scale.numel() * 2
        if not self.keep_bf16:
            self.weight = None                      # 真正释放 bf16 权重
        else:
            self.weight.data = w
        return (n_before - n_after) / 1e9

    def use_int8_now(self) -> bool:
        """prefill（M 大）默认走 bf16；没有 bf16 副本时只能走 int8。

        只读模块自己的 bool，不读全局 Context —— 后者进了编译图会在首个真实
        请求到来时让图作废（步 2 踩过）。
        """
        if not self.quantized:
            return False
        if self.weight is None:
            return True
        return self.int8_active

    def _linear(self, x: torch.Tensor, use_bias: bool = True) -> torch.Tensor:
        skip_head = _CFG["fp8_act_skip_head"] and getattr(self, "tie_target", None) is not None
        if _CFG["fp8_act"] != "off" and not skip_head and \
                x.dtype in (torch.bfloat16, torch.float16):
            # 步 12 预演：只改数值，不改存储。开着它测出来的 ppl 就是 W8A8 的代价。
            x = fake_fp8_act(x, _CFG["fp8_act"])
        if self.w_int8 is not None and self.use_int8_now():
            is4 = self.bits == 4
            if self.weight is None:
                # 纯量化：prefill 也走 custom op（能进 torch.compile 的图）
                return int4_w4a16(x, self.w_int8, self.w_scale, self.group_size) if is4 else \
                    int8_w8a16(x, self.w_int8, self.w_scale)
            return w4a16_mm(x, self.w_int8, self.w_scale, self.group_size) if is4 else \
                w8a16_mm(x, self.w_int8, self.w_scale)
        bias = getattr(self, "bias", None) if use_bias else None
        return F.linear(x, self.weight, bias)
