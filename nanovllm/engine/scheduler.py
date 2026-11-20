from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.pld_gamma_for = config.pld_gamma_for    # 步 12：按并发取 γ，0 = 关
        # 接受率统计（步 12 的验收指标之一）：draft 发出去多少、被接受多少
        self.pld_drafted = 0
        self.pld_accepted = 0
        self.pld_steps = 0
        self.block_manager = BlockManager(
            config.num_kvcache_blocks,
            config.kvcache_block_size,
            config.enable_prefix_caching,
        )
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        num_batched_tokens = 0

        # prefill
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        # 步 12：投机只在低并发兑现 —— 每步权重只读一遍，但 KV 要按 query 数重复读。
        # 高并发本来就是 KV 主导（C=64 时 KV 占 75%），再加 γ 个 query 直接变亏。
        # 所以 γ 按 batch size 分档；超出最后一档就是 0 = 退回纯 decode。
        gamma = self.pld_gamma_for(len(self.running))
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            # 步 12：先出 draft，再按 1+γ 个位置申请块。draft 查的是这条序列自己的
            # 历史（prompt + 已生成），命中就直接抄，查不到就退化成普通 decode。
            draft = seq.lookup_draft(gamma) if gamma else []
            seq.num_real_drafts = len(draft)
            if gamma and len(draft) < gamma:
                # verify 的 CUDA graph 要求形状固定（每序列 k = γ+1 个 query），
                # 命中不足就用 last_token 补位。补位 token 若恰好等于 argmax 就是
                # 白赚（accept 规则只看「与 argmax 相同」），不命中也不影响正确性。
                draft = draft + [seq.last_token] * (gamma - len(draft))
            seq.draft_token_ids = draft
            n = 1 + len(draft)
            while not self.block_manager.can_append_n(seq, n):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = n
                seq.is_prefill = False
                self.block_manager.may_append_n(seq, n)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        off = 0
        for seq in seqs:
            cached0 = seq.num_cached_tokens
            if is_prefill:
                seq.num_cached_tokens += seq.num_scheduled_tokens
                seq.num_scheduled_tokens = 0
                self.block_manager.hash_blocks(seq, cached0, seq.num_cached_tokens)
                seq.draft_token_ids = []
                tok = token_ids[off]
                off += 1
                if seq.num_cached_tokens < seq.num_tokens:
                    continue
                emit = [tok]
            else:
                n = seq.num_scheduled_tokens
                verify = token_ids[off:off + n]
                off += n
                draft = seq.draft_token_ids
                acc = 0
                for i, d in enumerate(draft):
                    if i + 1 >= n or verify[i] != d:
                        break
                    acc += 1
                # 接受率只统计**真题** draft 的命中：补位 token 即便命中也不算
                self.pld_drafted += seq.num_real_drafts
                self.pld_accepted += min(acc, seq.num_real_drafts)
                self.pld_steps += 1
                # 被接受的 draft + 一位 verify 兜底。干净的性质：accept 的 draft 与
                # verify 的 argmax 逐位相同，第一位不同处直接用 verify —— 所以输出
                # 与「不开投机的 greedy」逐 token 完全一致（不需要任何采样修正）。
                emit = list(draft[:acc]) + [verify[acc]]
                # KV 不需要回滚：槽位按位置寻址，被拒 draft 写进的位置会被后续 token
                # 覆写；只要不推进 num_cached_tokens 就没人读它们。
                seq.num_cached_tokens += acc + 1
                seq.num_scheduled_tokens = 0
                seq.draft_token_ids = []
            self._emit(seq, emit)
            if not is_prefill and seq.status == SequenceStatus.RUNNING:
                # emit 之后再 hash：此时 token_ids 才含本步新产出的 acc+1 个 token，
                # 区间 [cached0, cached0+acc+1) 里填满的块都能登记。
                self.block_manager.hash_blocks(seq, cached0, seq.num_cached_tokens)

    def _emit(self, seq: Sequence, toks: list[int]):
        for tok in toks:
            seq.append_token(tok)
            if (not seq.ignore_eos and tok == self.eos) or \
                    seq.num_completion_tokens >= seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
                return
