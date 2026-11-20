import gc
import os
import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler

# 步 12 的定位开关：由 scripts/step12_pld_bench.py --pld-dbg N 直接赋值。
# （这台机器上 export 的变量进不了 wsl.exe 的子进程，所以不用环境变量。）
PLD_DBG = 0
PLD_DBG_STEPS = 60
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model
from nanovllm.layers.int8_linear import set_fp8_act, set_int4_group, set_keep_bf16
from nanovllm.layers.attention import set_attn_backend, set_fp8_kv


class ModelRunner:

    @staticmethod
    def _mem(tag: str):
        """NANOVLLM_DEBUG_MEM=1 时打一行内存分解，用于定位 KV 池口径（RESULTS 13）。"""
        if not os.environ.get("NANOVLLM_DEBUG_MEM"):
            return
        free, total = torch.cuda.mem_get_info()
        st = torch.cuda.memory_stats()
        print(f"[mem] {tag:14s} used={(total-free)/2**30:.3f} "
              f"reserved={torch.cuda.memory_reserved()/2**30:.3f} "
              f"alloc={st['allocated_bytes.all.current']/2**30:.3f} "
              f"peak={st['allocated_bytes.all.peak']/2**30:.3f}", flush=True)

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")
        set_keep_bf16(config.int8_keep_bf16)
        set_int4_group(config.int4_group)
        set_fp8_act(config.fp8_act, config.fp8_act_skip_head)   # 步 12 预演：激活假量化
        set_attn_backend(config.attn_backend)     # 必须在 Qwen3ForCausalLM 之前
        set_fp8_kv(config.fp8_kv)
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self._mem("load完")
        self.quantize_model()
        self._mem("量化完")
        self.sampler = Sampler()
        self.warmup_model()
        self._mem("warmup完")
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)
        # 步 2：prefill 编译预热必须放在这里，不能放在 allocate_kv_cache 之后 ——
        # 上面两句 set_default_device / set_default_dtype 会改 dynamo 的 ambient guard，
        # 谁在它之前编译，谁的图在第一个真实请求到来时就全部作废重编（实测白付 20 s）。
        self.warmup_prefill_compile()

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    @staticmethod
    def _quant_bits(name: str, quant: str) -> int | None:
        """该层压到几位；None = 保持 bf16。步 5 的 int4 分档靠这张表分派。"""
        if quant == "none":
            return None
        is_head = name == "lm_head"
        is_mlp = name.endswith(("gate_proj", "up_proj", "down_proj"))
        if quant == "lm_head":
            return 8 if is_head else None
        if quant == "all":
            return 8
        if quant == "int4_head":
            return 4 if is_head else 8
        if quant == "int4_all":
            return 4
        if quant == "int4_mlp":
            return 4 if is_mlp else None
        raise ValueError(f"未知 quant 档位: {quant}")

    def quantize_model(self):
        """步 4/5：加载后就地压成 int8 / int4（不改 safetensors 格式）。

        放在 load 之后、warmup 之前：warmup 要跑的是量化后的真路径。
        """
        quant = self.config.quant
        if quant == "none":
            return
        if quant == "fp8_fake":
            # 步 12 预演：真 W8A8 里权重也得是 e4m3（3 位尾数，比 int8 的 7 位粗得多），
            # 这里只把数值按 per-output-channel 往返一次，存储仍是 bf16 —— 为了在
            # 写 kernel 之前就知道「两侧都是 fp8」的总代价。
            n = 0
            for _, m in self.model.named_modules():
                w = getattr(m, "weight", None)
                if w is None or w.dim() != 2 or w.dtype != torch.bfloat16:
                    continue
                wd = w.data
                s = wd.abs().amax(1, keepdim=True).clamp_min(1e-8) / 448.0
                q = (wd.float() / s).clamp(-448, 448).to(torch.float8_e4m3fn)
                wd.copy_((q.to(torch.float32) * s).to(torch.bfloat16))
                n += 1
            print(f"[quant] quant=fp8_fake：{n} 个 2D 权重按 per-channel e4m3 往返"
                  f"（存储仍是 bf16，测的是精度不是显存）", flush=True)
            return
        n_before = sum(p.numel() * p.element_size() for p in self.model.parameters())
        targets = []
        for name, m in self.model.named_modules():
            if not hasattr(m, "quantize_"):
                continue
            bits = self._quant_bits(name, quant)
            if bits is None:
                continue
            targets.append((name, m, bits))
        # 先把加载期的空闲块还回去，否则量化从 1.477 GB 的 reserved 起步，碎片更容易留下
        torch.cuda.empty_cache()
        saved = 0.0
        for name, m, bits in targets:
            saved += m.quantize_(bits)
        n_after = sum(p.numel() * p.element_size() for p in self.model.parameters() if p is not None)
        n_q = sum(m.w_int8.numel() + m.w_scale.numel() * 2
                  for _, m, _ in targets if m.w_int8 is not None)
        n4 = sum(1 for _, m, bits in targets if bits == 4)
        print(f"[quant] quant={quant} keep_bf16={self.config.int8_keep_bf16} "
              f"模块={len(targets)}（int4 {n4} 个 / int8 {len(targets) - n4} 个）"
              f" 参数 {n_before/2**30:.3f} → {n_after/2**30:.3f} GiB "
              f"（其中量化权重 {n_q/2**30:.3f} GiB，int4_group={self.config.int4_group}，"
              f"省 {saved:.3f} GB）", flush=True)
        self._mem("量化后")
        gc.collect()
        torch.cuda.empty_cache()
        self._mem("gc+empty")

    def warmup_model(self):
        torch.cuda.empty_cache()
        self._mem("warmup前")
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def warmup_prefill_compile(self):
        """步 2（P-A）：把 prefill 编译路径的首次编译放到启动阶段做完。

        必须排在 `allocate_kv_cache()` 之后 —— 编译图要看到最终的 k_cache/v_cache
        张量；提前编译会看着一堆空张量把图固化下来。

        触发用的序列给了真实 block_table，为的是让 store_kvcache 那段也走一遍
        （否则 slot_mapping 是空的，assert 会挂）。写入的是空闲块的垃圾数据，
        启动阶段哈希表还是空的，后续请求分配到时会被完整覆写。
        """
        mode = os.environ.get("NANOVLLM_PREFILL_COMPILE", self.config.prefill_compile)
        model = self.model.model
        if mode in (None, "", "off") or not hasattr(model, "enable_prefill_compile"):
            return
        caches = []
        for layer in model.layers:
            attn = layer.self_attn.attn
            # 每层 4 个：k, v, k_scale, v_scale（bf16 时后两个是空张量，
            # 但位置要占住，`_prefill_body` 按 4*i 取）
            caches += [attn.k_cache, attn.v_cache, attn.k_scale, attn.v_scale]
        model.set_prefill_caches(caches)
        model.enable_prefill_compile(mode)
        n = 512
        seq = Sequence([0] * n)
        seq.num_scheduled_tokens = n
        seq.block_table = list(range((n + self.block_size - 1) // self.block_size))
        self.run([seq], True)

        # 第二种图变体：block_tables 非 None（= prefix 命中后的增量 prefill）。
        # 不预热的话，第一次命中会现场编译 —— 实测 model 模式 34 s，layer 模式 9 s。
        # 代价是启动时多编译一次；换来的是"命中即快"，而不是"第一次命中先等半分钟"。
        n2 = 1024
        seq2 = Sequence([0] * n2)
        seq2.num_cached_tokens = 512          # 前 512 token 视作已缓存
        seq2.num_scheduled_tokens = 512       # 只算后 512 token
        seq2.block_table = list(range((n2 + self.block_size - 1) // self.block_size))
        self.run([seq2], True)

        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        # 步 11：fp8 KV。每个 (token, head) 行多一个 fp32 scale（2 份：K 与 V），
        # 所以单块 = 14 MiB 数据 + 0.44 MiB scale = 14.44 MiB（bf16 是 28 MiB）。
        # block_bytes 必须把 scale 算进去，否则池子会按纯 fp8 估大、最后 OOM。
        cache_dtype = torch.float8_e4m3fn if config.kv_cache_dtype == "fp8" else hf_config.dtype
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * cache_dtype.itemsize
        if config.kv_cache_dtype == "fp8":
            block_bytes += 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * 4
        budget = total * config.gpu_memory_utilization - used - peak + current
        config.num_kvcache_blocks = int(budget) // block_bytes
        # 调参/对齐用：把池子夹到指定块数，让两种配置拿到同一套 block id
        # （块 id 不同时，两份 cache 的同一块装的根本不是同一批 token，没法逐元素比）
        if config.kv_blocks_limit:
            config.num_kvcache_blocks = min(config.num_kvcache_blocks, config.kv_blocks_limit)
        assert config.num_kvcache_blocks > 0
        # 分配口径全打出来：量化后池子反而不涨的账就在这里（RESULTS 表 13）。
        # 非torch = used - reserved（CUDA context / cuBLAS / Triton cubin）；
        # 碎片 = reserved - current（allocator 撑开后还不回去的空洞）；
        # 临时峰 = peak - current（warmup 一度分配、现已释放的量，与权重大小无关）。
        print(f"[kv] total={total/2**30:.2f} used={used/2**30:.2f} "
              f"reserved={torch.cuda.memory_reserved()/2**30:.2f} "
              f"current={current/2**30:.3f} peak={peak/2**30:.3f} "
              f"| 非torch={(used - torch.cuda.memory_reserved())/2**30:.2f} "
              f"碎片={(torch.cuda.memory_reserved()-current)/2**30:.3f} "
              f"临时峰={(peak-current)/2**30:.3f} "
              f"| budget={budget/2**30:.2f} block={block_bytes/2**20:.2f}MB "
              f"→ {config.num_kvcache_blocks} 块", flush=True)
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim,
                                    dtype=cache_dtype)
        if config.kv_cache_dtype == "fp8":
            # scale 与 cache 分开存（[layer, block, slot, head]），读侧按 slot 索引
            self.kv_scale = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads,
                                        dtype=torch.float32)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                if config.kv_cache_dtype == "fp8":
                    module.k_scale = self.kv_scale[0, layer_id]
                    module.v_scale = self.kv_scale[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            seqlen_k = end
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        # 步 4：prefill 走 bf16（大 M 下 int8 kernel 反而慢），decode 走 int8。
        # 只在这里切换，别让 forward 自己读 context —— 编译图不能读全局状态。
        self.model.set_int8_active(not is_prefill)
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def _verify_slots(self, seq, start: int, n: int) -> list[int]:
        """verify 要写的 n 个连续位置 [start, start+n) 对应的 cache 槽位。"""
        out = []
        start_block = start // self.block_size
        end_block = (start + n + self.block_size - 1) // self.block_size
        for i in range(start_block, end_block):
            slot_start = seq.block_table[i] * self.block_size
            if i == start_block:
                slot_start += start % self.block_size
            if i != end_block - 1:
                slot_end = seq.block_table[i] * self.block_size + self.block_size
            else:
                slot_end = seq.block_table[i] * self.block_size + start + n - i * self.block_size
            out.extend(range(slot_start, slot_end))
        return out

    def prepare_verify(self, seqs: list[Sequence]):
        """步 12：verify 的 eager 输入（`enforce_eager` 或没捕获到图时用）。

        形状与 prefill 的 prefix-cache 分支一致（`cu_seqlens_k > cu_seqlens_q` +
        block_table）—— 要读的正是已经在 cache 里的整段前缀。
        `start = len(seq) - 1`：最后一个 token 还没进 cache（老约定：它的槽位由本次
        前向写入），所以它要和 draft 一起送进去。
        """
        input_ids, positions, slot_mapping = [], [], []
        cu_seqlens_q, cu_seqlens_k = [0], [0]
        max_seqlen_q = max_seqlen_k = 0
        for seq in seqs:
            start = len(seq) - 1
            toks = [seq.last_token] + list(seq.draft_token_ids)
            n = len(toks)
            input_ids.extend(toks)
            positions.extend(range(start, start + n))
            cu_seqlens_q.append(cu_seqlens_q[-1] + n)
            cu_seqlens_k.append(cu_seqlens_k[-1] + start + n)
            max_seqlen_q = max(max_seqlen_q, n)
            max_seqlen_k = max(max_seqlen_k, start + n)
            slot_mapping.extend(self._verify_slots(seq, start, n))
        block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                    slot_mapping, None, block_tables, all_positions=True, verify=True)
        return input_ids, positions

    @torch.inference_mode()
    def run_verify(self, seqs: list[Sequence]) -> list[int]:
        """步 12：draft 的 verify。greedy（argmax），返回扁平 token 列表（每序列 k 个）。

        这里**必须**保持 int8 权重激活 —— 基线 decode 就是 int8，换成 prefill 的 bf16
        会让数值路径不同，"与基线逐 token 相同" 这条硬约束就守不住了。
        """
        if self.enforce_eager or not getattr(self, "verify_graphs", None):
            return self._run_verify_eager(seqs)
        return self._run_verify_graph(seqs)

    def _run_verify_eager(self, seqs):
        input_ids, positions = self.prepare_verify(seqs)
        self.model.set_int8_active(True)
        logits = self.model.compute_logits(self.model(input_ids, positions))
        self._pld_fingerprint("eager", logits, len(seqs), input_ids.size(0) // max(1, len(seqs)))
        token_ids = logits.argmax(dim=-1).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    def _run_verify_graph(self, seqs):
        """走捕获好的 verify 图（按并发分档）。eager 版每步要发 ~300 次 kernel，
        其中 Triton 的 Python 侧 launch 每个几十微秒 —— 实测 23 ms/步，比 1-token 的
        decode（1.9 ms）还慢一个量级。图一回放就没有这些开销了。
        """
        bs = len(seqs)
        gbs = next((x for x in self.graph_bs if x >= bs), None)
        if gbs is None:
            # bs 超出捕获过的最大档 → 退回 eager（正确性优先）
            return self._run_verify_eager(seqs)
        rec = self.verify_graphs.get(gbs)
        k = 1 + len(seqs[0].draft_token_ids)
        if rec is None or rec["k"] != k:
            # 档位对不上（例如调用方手工改了 draft 长度）→ 退回 eager，保证正确性
            return self._run_verify_eager(seqs)
        n_tok = bs * k
        # 图里的权重是捕获时固定的（int8）；这里补的是**图外**的 lm_head
        self.model.set_int8_active(True)
        # 补位行的槽位必须是 -1（= 不写），否则会踩别人的 cache
        rec["slot_mapping"].fill_(-1)
        ids, pos, slots = [], [], []
        cu_k = [0]
        for seq in seqs:
            start = len(seq) - 1
            toks = [seq.last_token] + list(seq.draft_token_ids)
            assert len(toks) == k, f"verify 的 query 数必须固定为 {k}，实际 {len(toks)}"
            ids.extend(toks)
            pos.extend(range(start, start + k))
            slots.extend(self._verify_slots(seq, start, k))
            cu_k.append(cu_k[-1] + start + k)
        rec["input_ids"][:n_tok].copy_(torch.tensor(ids, dtype=torch.int64), non_blocking=True)
        rec["positions"][:n_tok].copy_(torch.tensor(pos, dtype=torch.int64), non_blocking=True)
        rec["slot_mapping"][:n_tok].copy_(torch.tensor(slots, dtype=torch.int32), non_blocking=True)
        # 补位段：给足 k 个 key，保证 cu_k 单调且不少于 cu_q（负的 q-k 偏移是未定义行为）
        while len(cu_k) < gbs + 1:
            cu_k.append(cu_k[-1] + k)
        rec["cu_seqlens_k"].copy_(torch.tensor(cu_k[:gbs + 1], dtype=torch.int32),
                                  non_blocking=True)
        bt = self.prepare_block_tables(seqs)
        # 图是按 bucket 大小捕获的，实际 bs 可能更小 → 只用前 bs 行
        rbt = rec["block_tables"][:bs]
        rbt.zero_()
        rbt[:, :bt.size(1)] = bt
        rec["graph"].replay()
        set_context(True, all_positions=True, verify=True)
        logits = self.model.compute_logits(rec["outputs"][:n_tok])
        self._pld_fingerprint("graph", logits, bs, k)
        token_ids = logits.argmax(dim=-1).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    def _pld_fingerprint(self, tag: str, logits: torch.Tensor, bs: int, k: int):
        """PLD_DBG=1 时把 verify 的 logits 指纹打到 stdout（定位非确定性用）。

        只记「top1 / top1-top2 差值 / fp64 求和」——差值接近 0 的位置就是近平局，
        这些位置最容易因为 1 ulp 的差别翻转 argmax。跨进程对比这串指纹就能区分
        「数值路径真的不一样」和「引擎逻辑不一样」。
        """
        if not PLD_DBG:
            return
        dbg = getattr(self, "_pld_dbg", None)
        if dbg is None:
            dbg = self._pld_dbg = {"n": 0}
        limit = PLD_DBG_STEPS
        if dbg["n"] >= limit:
            return
        top = torch.topk(logits.detach().float(), 2, dim=-1)
        idx = top.indices.tolist()
        gap = (top.values[:, 0] - top.values[:, 1]).tolist()
        checksum = float(logits.detach().double().sum())
        print(f"VERIFYFP {tag} step={dbg['n']} bs={bs} k={k} sum={checksum:.6f}", flush=True)
        for i, (a, g) in enumerate(zip(idx, gap)):
            print(f"VERIFYFP   {i} top1={a[0]} gap={g:.7f}", flush=True)
        dbg["n"] += 1

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        # 只有调度器真的发了 draft 才走 verify —— 并发门控关掉投机时，
        # 这里自动落回普通 decode（CUDA graph，1 token/步）。
        if not is_prefill and any(seq.draft_token_ids for seq in seqs):
            return self.run_verify(seqs)
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        # 对照用：普通 prefill / decode 的 logits 指纹。基线 decode 若也跨进程不同，
        # 说明差异不在 verify 这条路上（那就不是 PLD 的 bug）。
        self._pld_fingerprint("prefill" if is_prefill else "decode", logits,
                              len(seqs), input_ids.size(0) // max(1, len(seqs)))
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.model.set_int8_active(True)      # 捕获的是 decode 图
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )

        # ---------------- 步 12：PLD 的 verify 图（**按并发分档**，一档一张）
        # 形状必须固定，否则 dynamo 每步重编译（实测 eager 23 ms/步、torch.compile
        # 反而 511 ms/步 —— cu_seqlens 每步都变，编译一次只能服务一步）。
        #   · 每序列固定 k = γ+1 个 query（draft 不足就补位，见 Scheduler）
        #   · max_seqlen_k 取 max_model_len 上界：多出来的 CTA 会因 cu_seqlens 越界
        #     立刻退出，代价只是空转，换来一张图覆盖所有上下文长度
        #   · γ 按 batch size 分档，所以每档各有一张图（形状 = bs×k 不同）
        self.verify_graphs = {}
        if any(g > 0 for _, g in config.pld_schedule):
            for bs in reversed(self.graph_bs):
                k = config.pld_gamma_for(bs) + 1
                if k <= 1:
                    continue
                q = bs * k
                v_ids = torch.zeros(q, dtype=torch.int64)
                v_pos = torch.zeros(q, dtype=torch.int64)
                v_slot = torch.zeros(q, dtype=torch.int32)
                v_cu_q = torch.arange(0, (bs + 1) * k, k, dtype=torch.int32)
                v_cu_k = torch.zeros(bs + 1, dtype=torch.int32)
                v_out = torch.zeros(q, hf_config.hidden_size)
                v_bt = block_tables[:bs]
                set_context(True, v_cu_q, v_cu_k, k, config.max_model_len,
                            v_slot, None, v_bt, all_positions=True, verify=True)
                v_out[:] = self.model(v_ids, v_pos)              # warmup
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, self.graph_pool):
                    v_out[:] = self.model(v_ids, v_pos)
                self.verify_graphs[bs] = dict(
                    k=k, graph=graph, input_ids=v_ids, positions=v_pos, slot_mapping=v_slot,
                    cu_seqlens_q=v_cu_q, cu_seqlens_k=v_cu_k, outputs=v_out, block_tables=v_bt)
                torch.cuda.synchronize()
                reset_context()
            print(f"[pld] verify 图: " + ", ".join(
                f"bs<={b} k={r['k']}" for b, r in sorted(self.verify_graphs.items())), flush=True)

        # verify 图已经用 int8 权重捕获过了，收尾时把标志交还默认
        self.model.set_int8_active(False)
