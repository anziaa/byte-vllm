from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int, enable_prefix_caching: bool = True):
        self.block_size = block_size
        self.enable_prefix_caching = enable_prefix_caching
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()
        # 命中率统计: lookup = 所有可被复用查找的完整块数, hit = 其中真正命中的块数
        self.cache_lookup_blocks: int = 0
        self.cache_hit_blocks: int = 0

    @property
    def cache_hit_rate(self) -> float:
        if self.cache_lookup_blocks == 0:
            return 0.0
        return self.cache_hit_blocks / self.cache_lookup_blocks

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self) -> int:
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    def _deallocate_block(self, block_id: int):
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> int:
        h = -1
        num_cached_blocks = 0
        num_new_blocks = seq.num_blocks
        if self.enable_prefix_caching:
            # 最后一个 block 不参与命中: 它可能还没写满, 内容会继续变化
            for i in range(seq.num_blocks - 1):
                token_ids = seq.block(i)
                h = self.compute_hash(token_ids, h)
                block_id = self.hash_to_block_id.get(h, -1)
                if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                    break
                num_cached_blocks += 1
                if block_id in self.used_block_ids:
                    num_new_blocks -= 1
        if len(self.free_block_ids) < num_new_blocks:
            return -1
        # 只在真正可分配时记账, 否则显存不足重入会把 lookup 重复累加, 命中率被稀释
        self.cache_lookup_blocks += seq.num_blocks - 1
        self.cache_hit_blocks += num_cached_blocks
        return num_cached_blocks

    def allocate(self, seq: Sequence, num_cached_blocks: int):
        assert not seq.block_table
        h = -1
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id[h]
            block = self.blocks[block_id]
            if block_id in self.used_block_ids:
                block.ref_count += 1
            else:
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)
        for i in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self._allocate_block())
        seq.num_cached_tokens = num_cached_blocks * self.block_size

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block())

    # 步 12：一次 verify 要写 n 个位置（含最后一个还没进 cache 的 token），
    # 可能一口气跨过多个块边界 —— 所以按「最大下标」算需要几个块，
    # 而不是原来那个只看 `len % block_size == 1` 的判定。
    def _blocks_for(self, seq: Sequence, n: int) -> int:
        return (len(seq) + n - 2) // self.block_size + 1

    def can_append_n(self, seq: Sequence, n: int) -> bool:
        need = self._blocks_for(seq, n) - len(seq.block_table)
        return len(self.free_block_ids) >= max(0, need)

    def may_append_n(self, seq: Sequence, n: int):
        need = self._blocks_for(seq, n)
        while len(seq.block_table) < need:
            seq.block_table.append(self._allocate_block())

    def hash_blocks(self, seq: Sequence, token_lo: int, token_hi: int):
        """把 token 区间 [token_lo, token_hi) 里**已经填满**的块登记为 prefix hash。

        区间必须显式传：步 12 之后一次前进可能不止一个 token，用
        `num_cached_tokens + num_scheduled_tokens` 这种隐式推导在
        「恰好跨块」时会把**内容还没生成完**的块拿去算 hash ——
        两条只在第 251 个 token 之后才不同的序列会碰撞，错误共享块。
        """
        if not self.enable_prefix_caching:
            return
        start = token_lo // self.block_size
        # 只能 hash 内容已经存在的块（token_ids 里得有）
        end = min(token_hi, len(seq.token_ids)) // self.block_size
        if start >= end: return
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.block_id
