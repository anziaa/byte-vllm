import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    enable_prefix_caching: bool = True
    # 步 2（P-A）：prefill 整层编译。off | layer | model
    #   off   —— 老路径：每个算子各自 @torch.compile，一层 6 个编译入口
    #   layer —— 每个 decoder 层编译成一个 region（28 个）
    #   model —— 整个模型 forward 编译成一个 region（1 个）
    prefill_compile: str = "model"
    # 步 4/5：weight-only 量化档位。
    #   none       —— 全 bf16
    #   lm_head    —— 只量化输出头（int8，0.31 GB，M 恒为 batch，稳赚 ~1.84x）
    #   all        —— 全线 int8（1.19 → 0.60 GB），prefill 是否退步由 int8_keep_bf16 决定
    # 步 5 的 int4 分档（从「最敢压」的 lm_head 往上爬）：
    #   int4_head  —— 层 int8 + lm_head int4
    #   int4_all   —— 全线 int4（1.19 → 0.30 GB）
    #   int4_mlp   —— 只 MLP int4，attn / lm_head 保持 bf16
    #   fp8_fake   —— 步 12 预演：权重按 per-channel e4m3 往返（存储仍是 bf16）。
    #                 配合 fp8_act 一起用，模拟真 W8A8 两侧都是 e4m3 的精度损失
    quant: str = "none"
    # True：给 prefill 留一份 bf16 权重（多占 0.6 GB，换 prefill 不退步）
    int8_keep_bf16: bool = True
    # 步 5：int4 的 group size（0 = per-output-channel）。组内共享 scale，
    # 越小组内动态范围越小、精度越高，代价是 scale 显存（K/G × N）。
    int4_group: int = 0
    num_kvcache_blocks: int = -1
    # 调参/对齐用：把 KV 池夹到指定块数（0 = 不夹）。两种配置夹成同一块数后，
    # 同一块 id 装的是同一批 token，cache 才能逐元素比（否则块 id 不同，比出来是假的）
    kv_blocks_limit: int = 0
    # 步 10：decode 的 attention 后端。
    #   fa2    —— flash_attn_with_kvcache，KV 只能是 bf16/fp16
    #   triton —— 自写 paged decode，能读 fp8 cache（步 11 的前置；bf16 下与 FA2 打平）
    attn_backend: str = "fa2"
    # 步 11：KV cache 的真实存储格式。
    #   bf16 —— 原样
    #   fp8  —— e4m3 + per-(token, head) 的 fp32 scale（额外 4 字节/行，见
    #           `allocate_kv_cache` 的 block_bytes 口径：单块 28 → 14.4 MiB）
    # fp8 只被自写 kernel 支持：decode 走 `paged_decode_attention`，
    # prefill 的 prefix-hit 分支先把 cache 反量化 gather 成 bf16 varlen 再喂 FA2。
    # 所以 kv_cache_dtype="fp8" 时 attn_backend 必须是 triton（fa2 读不了 fp8）。
    kv_cache_dtype: str = "bf16"
    # 步 11 的**数值预演**：把写入 cache 的 K/V 先量化到 e4m3 再反量化回 bf16 存。
    #   off  —— 不碰（默认，bf16）
    #   fake —— per-(token, head) scale 的 e4m3 往返（只改数值，不改 dtype/显存）
    # 用途是**提前量出 fp8 KV 的 ppl 代价**，不需要真 fp8 存储和 kernel；
    # 力度上它是 fp8 的**乐观下界**（行级 scale 比 8.15 里的 per-head 更细）。
    fp8_kv: str = "off"
    # 步 12 预演：把线性层的**激活**也做 e4m3 往返（假量化：算出来还是 bf16，
    # 只让数值带上 fp8 的精度）。先量 ppl 代价，再决定要不要写真正的 W8A8 kernel。
    #   off    —— 不碰（默认）
    #   tensor —— per-tensor scale（torch 2.5.1 的 `_scaled_mm` 只支持这个粒度）
    #   token  —— per-token scale（真 kernel 打算用的粒度，理论上略优于 tensor）
    # 注意：假量化用 torch 的 `.to(float8_e4m3fn)`（RNE），与 Triton 的截断行为
    # 有 0.26% 码点差异（见 RESULTS 8.18c），所以它是**乐观下界**。
    fp8_act: str = "off"
    # 激活量化是否跳过 lm_head。它是最敏感的一块（驱动整条 logits），而 prefill 里
    # 只值 1.2×（该 GEMM 本身就带宽受限），所以默认跳过。
    fp8_act_skip_head: bool = True
    # 步 12：PLD（prompt lookup decoding）投机解码。
    #   γ = 0 关闭；γ>0 时每步用「context 末尾 n-gram 的最长历史匹配」draft γ 个
    #   token，再一次前向 verify γ+1 个位置。
    #   只支持 **greedy**：verify 用 argmax，接受 = 与 draft 逐位相同，
    #   被拒处用 verify 自己的 argmax 兜底 → 输出与基线 greedy 逐 token 完全一致。
    #   （采样要保证分布不变必须走 rejection sampling，本引擎暂不提供。）
    pld_gamma: int = 0
    # draft 回看的 n-gram 阶数：取 context 末尾 n 个 token 去历史里找最近一次出现，
    # n 从大到小退。3 是 vLLM / prompt-lookup 的常用值。
    pld_max_n: int = 3
    # 按并发分档给 γ：`"1:32,4:12,16:4"` = bs<=1 用 32、bs<=4 用 12、bs<=16 用 4，
    # 超过最后一档就不投机（退回纯 decode，那条路有 CUDA graph，本来就是最优）。
    #
    # 为什么必须分档（RESULTS 8.21）：投机每步**权重只读一遍**，但 **query 数翻 k=γ+1 倍**。
    # 低并发是权重/固定成本主导（C=1 每步 4.5 ms 与 token 数无关）→ 多算几个位置几乎
    # 免费，实测 C=1 γ=32 → 15.1×；高并发每步成本随 query 数涨，而每步只多产 acc·γ 个
    # token → C=64 不该投机（γ=3 时 0.81×，是亏的）。
    # 实测最优点（同进程 A/B，RESULTS 8.21e）：C=1 → γ=32、C=4 → γ=12、C=16 → 分档 1.42×。
    #
    # **默认关闭**：verify 恒用 argmax，接受规则只在 greedy 下与「一步一 token」等价；
    # 本引擎的 Sampler 是按 temperature 采样的（且禁止 greedy），所以默认值留空。
    # 想用就把下面这串填进 `pld_tiers`，并且把 sampler 换成 argmax：
    #   "1:32,4:12,16:4"
    # `pld_tiers` 为空时，所有并发统一用 `pld_gamma`。
    pld_tiers: str = ""

    @property
    def pld_schedule(self) -> list[tuple[int, int]]:
        """[(max_bs, gamma)] 按 max_bs 升序；空 = 不投机。"""
        out = []
        for item in (self.pld_tiers or "").split(","):
            if item.strip():
                mb, g = item.split(":")
                out.append((int(mb), int(g)))
        if not out and self.pld_gamma > 0:
            out = [(512, self.pld_gamma)]
        return sorted(out)

    def pld_gamma_for(self, bs: int) -> int:
        """按 batch size 取 γ；0 = 这一档不投机。"""
        for max_bs, g in self.pld_schedule:
            if bs <= max_bs:
                return g
        return 0

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        assert self.quant in ("none", "lm_head", "all", "int4_head", "int4_all", "int4_mlp",
                              "fp8_fake"), self.quant
        assert self.int4_group in (0, 32, 64, 128, 256), self.int4_group
        assert self.attn_backend in ("fa2", "triton"), self.attn_backend
        assert self.kv_cache_dtype in ("bf16", "fp8"), self.kv_cache_dtype
        assert self.fp8_kv in ("off", "fake", "fake_k", "fake_v"), self.fp8_kv
        assert self.fp8_act in ("off", "tensor", "token", "int8"), self.fp8_act
        assert self.pld_gamma >= 0, self.pld_gamma
        assert 1 <= self.pld_max_n <= 8, self.pld_max_n
        for item in self.pld_tiers.split(","):
            if item.strip():
                mb, g = item.split(":")
                assert int(mb) >= 1 and int(g) >= 1, item
        # fp8 conv 只能由自写 kernel 读（FA2 的 kvcache 接口没有 dtype 入口）
        assert not (self.kv_cache_dtype == "fp8" and self.attn_backend == "fa2"), \
            "kv_cache_dtype=fp8 需要 attn_backend=triton"
        assert not (self.kv_cache_dtype == "fp8" and self.fp8_kv != "off"), \
            "kv_cache_dtype=fp8 已是真的 fp8，不要再叠 fake"
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
