import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.layers.triton_attn import paged_decode_attention, gather_kvcache
from nanovllm.utils.context import get_context


# ---------------------------------------------------------------- 步 10：decode 的 attention 后端
#
#   fa2    —— flash_attn_with_kvcache（默认；KV 只能是 bf16/fp16）
#   triton —— 自写 paged decode（能读 fp8 cache，步 11 的前置）
#
# 模块级默认值 + 实例快照：Attention 构造时读一次存成实例属性，避免在
# CUDA graph 捕获之后再改（捕获后改模块变量不会生效，容易踩坑）。

_ATTN_BACKEND = "fa2"
_FP8_KV = "off"


def set_attn_backend(name: str):
    """必须在模型构造之前调用（Attention.__init__ 会读快照）。"""
    global _ATTN_BACKEND
    assert name in ("fa2", "triton"), name
    _ATTN_BACKEND = name


def set_fp8_kv(mode: str):
    """步 11 的数值预演：写入 cache 前把 K/V 做一次 e4m3 往返。

    **不是真的 fp8 存储**——cache 还是 bf16，只是里面装的值等价于「量化到 e4m3
    再反量化」的结果。这样不用改 dtype、不用写 kernel，就能用同一个 ppl 口径
    量出 fp8 KV 的代价，作为步 11 的验收锚点。

    mode 除 off / fake（K+V 都量化）外，另有 fake_k / fake_v 两个单边档：
    RESULTS 8.15 实测误差源以 K 为主（输出误差 K 6.8% / V 1.9%），单边档用来
    校验「模拟量到的效应是否也按这个比例分配」，否则就是模拟本身有问题。
    """
    global _FP8_KV
    assert mode in ("off", "fake", "fake_k", "fake_v"), mode
    _FP8_KV = mode


FP8_E4M3_MAX = 448.0


def _fp8_roundtrip(t: torch.Tensor) -> torch.Tensor:
    """per-(token, head) scale 的 e4m3 量化/反量化。

    scale 取每一行（一个 token 的一个 head，head_dim 个元素）的 absmax/448。
    这是 fp8 KV 能拿到的最细粒度（不需要校准），所以结果偏乐观；
    真实的 per-head（跨 token 共享一个 scale）会更差一点，见 RESULTS 8.15。
    """
    amax = t.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    scale = amax / FP8_E4M3_MAX
    q = (t.float() / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    return (q.float() * scale).to(t.dtype)


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


@triton.jit
def store_kvcache_fp8_kernel(
    key_ptr, key_stride, value_ptr, value_stride,
    k_cache_ptr, v_cache_ptr, k_scale_ptr, v_scale_ptr,
    slot_mapping_ptr,
    HKV: tl.constexpr, D: tl.constexpr, FP8_MAX: tl.constexpr,
):
    """fp8 版：一行 = (一个 token 的一个 KV head)，scale 取该行 absmax/448。

    粒度选 per-(token, head) 而不是 per-head：两者都免校准，但前者更细，
    且与步 11 的预演（`_fp8_roundtrip`）逐位同构 —— 真实的 fp8 ppl 必须能复现
    预演的数，这是本轮最强的自检（RESULTS 8.17c）。
    scale 存 fp32，每行多 4 字节：128 维的 K 行从 256 → 132 字节（51.6%）。
    """
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx).to(tl.int64)
    if slot == -1: return
    offs_h = tl.arange(0, HKV)
    offs_d = tl.arange(0, D)
    src_k = idx * key_stride + offs_h[:, None] * D + offs_d[None, :]
    # V 的行 stride 与 K 不一定相等：K 过 rotary 后往往是连续张量，V 还是 packed
    # qkv 的切片（行 stride = (HQ + 2·HKV)·D）。第一版这里复用了 key_stride，
    # 结果 K 全对、V 全错（rel≈2.0）—— 引擎级探针逐层看一次就定位了。
    src_v = idx * value_stride + offs_h[:, None] * D + offs_d[None, :]
    dst = slot * (HKV * D) + offs_h[:, None] * D + offs_d[None, :]

    k = tl.load(key_ptr + src_k).to(tl.float32)
    ks = tl.maximum(tl.max(tl.abs(k), 1), 1e-8) / FP8_MAX
    kq = tl.minimum(tl.maximum(k / ks[:, None], -FP8_MAX), FP8_MAX)
    tl.store(k_cache_ptr + dst, kq.to(tl.float8e4nv))
    tl.store(k_scale_ptr + slot * HKV + offs_h, ks)

    v = tl.load(value_ptr + src_v).to(tl.float32)
    vs = tl.maximum(tl.max(tl.abs(v), 1), 1e-8) / FP8_MAX
    vq = tl.minimum(tl.maximum(v / vs[:, None], -FP8_MAX), FP8_MAX)
    tl.store(v_cache_ptr + dst, vq.to(tl.float8e4nv))
    tl.store(v_scale_ptr + slot * HKV + offs_h, vs)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
                  slot_mapping: torch.Tensor,
                  k_scale: torch.Tensor | None = None, v_scale: torch.Tensor | None = None):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    if k_cache.dtype == torch.float8_e4m3fn:
        assert k_scale is not None and v_scale is not None
        store_kvcache_fp8_kernel[(N,)](
            key, key.stride(0), value, value.stride(0),
            k_cache, v_cache, k_scale, v_scale, slot_mapping,
            HKV=num_heads, D=head_dim, FP8_MAX=FP8_E4M3_MAX,
            num_warps=4, num_stages=1,
        )
        return
    if _FP8_KV != "off":
        # 只改「写进 cache 的值」，不改 dtype：读侧无需任何改动
        if _FP8_KV != "fake_v":
            key = _fp8_roundtrip(key)
        if _FP8_KV != "fake_k":
            value = _fp8_roundtrip(value)
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


# ---------------------------------------------------------------- 步 2：prefill 的 attention 段
#
# 为什么要把它包成 custom op（而不是让 Dynamo 直接 trace）：
#
#   1. 它读全局 Context（is_prefill / cu_seqlens / slot_mapping ...）。Context 每一
#      步都是新对象，Dynamo 会在 `_CONTEXT.is_prefill` 上挂 guard，然后每一步都失效
#      —— 结果是"编译一次、之后每步重编译一次"，比不编译还慢。
#   2. 里面有手写 Triton kernel 的动态 grid 和 flash-attn 的 python 分支，Dynamo
#      本来就 trace 不动，会退回 graph break。
#
# 包成 custom op 之后，对编译图来说它就是"一个不透明的算子"：前面 11 个算子共享
# 同一张图，只有它被当成一次调用。语义与 `Attention.forward` 的 prefill 分支逐字一致。

@torch.library.custom_op("nanovllm::prefill_attn", mutates_args=())
def prefill_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    slot_mapping: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    block_tables: torch.Tensor | None,
    scale: float,
) -> torch.Tensor:
    if k_cache.numel() and v_cache.numel():
        store_kvcache(k, v, k_cache, v_cache, slot_mapping, k_scale, v_scale)
    if block_tables is not None:
        if k_cache.dtype == torch.float8_e4m3fn:
            # fp8 paged cache 不能直接喂 FA2（它没有 dtype/scale 入口）→ 先反量化摊平。
            # 结果走的是非 paged 的 varlen 路径，所以 block_table 置 None。
            k, v = gather_kvcache(k_cache, v_cache, k_scale, v_scale, block_tables,
                                  cu_seqlens_k, block_tables.shape[0], max_seqlen_k)
            block_tables = None
        else:
            k, v = k_cache, v_cache
    return flash_attn_varlen_func(q, k, v,
                                  max_seqlen_q=max_seqlen_q, cu_seqlens_q=cu_seqlens_q,
                                  max_seqlen_k=max_seqlen_k, cu_seqlens_k=cu_seqlens_k,
                                  softmax_scale=scale, causal=True, block_table=block_tables)


@prefill_attn.register_fake
def _prefill_attn_fake(q, k, v, k_cache, v_cache, k_scale, v_scale, slot_mapping,
                       cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                       block_tables, scale):
    # 输出形状与 q 相同：[total_q, num_heads, head_dim]
    return torch.empty_like(q)



class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.attn_backend = _ATTN_BACKEND
        self.k_cache = self.v_cache = torch.tensor([])
        # fp8 cache 的 per-(token, head) scale；bf16 cache 下恒为空张量
        self.k_scale = self.v_scale = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        k_scale, v_scale = self.k_scale, self.v_scale
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping, k_scale, v_scale)
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                block_tables = context.block_tables
                if k_cache.dtype == torch.float8_e4m3fn:
                    k, v = gather_kvcache(k_cache, v_cache, k_scale, v_scale, block_tables,
                                          context.cu_seqlens_k, block_tables.shape[0],
                                          context.max_seqlen_k)
                    block_tables = None
                else:
                    k, v = k_cache, v_cache
            else:
                block_tables = None
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=block_tables)
        else:    # decode
            if self.attn_backend == "triton" or k_cache.dtype == torch.float8_e4m3fn:
                # 与 FA2 同语义（causal 对单 token query 无影响：sl 已含当前 token）
                o = paged_decode_attention(q, k_cache, v_cache,
                                           context.block_tables, context.context_lens,
                                           self.scale, k_scale, v_scale)
            else:
                o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                            cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                            softmax_scale=self.scale, causal=True)
        return o

    def forward_prefill(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
        slot_mapping: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        block_tables: torch.Tensor | None,
    ) -> torch.Tensor:
        """编译路径专用：所有输入都从参数走，不碰全局 Context。

        k_cache / v_cache 也必须由调用方传进来 —— 它们是 module 上的普通张量属性，
        而 `allocate_kv_cache()` 会在编译之后把它们换成新张量。让编译图直接读
        `self.k_cache`，Dynamo 会把它当常量拷进图里，KV 就永远写不进真正的 cache。
        fp8 cache 的 k_scale / v_scale 同理。
        """
        return prefill_attn(q, k, v, k_cache, v_cache, k_scale, v_scale, slot_mapping,
                            cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                            block_tables, self.scale)
