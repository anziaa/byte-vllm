from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    block_size = 256
    # 步 12：PLD 的 n-gram 阶数上限（由 LLMEngine 按 Config 设置）
    pld_max_n = 3
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.num_scheduled_tokens = 0
        self.is_prefill = True
        self.block_table = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos
        # 步 12：本步要 verify 的 draft（调度器生成；空 = 退化成普通 decode）
        self.draft_token_ids: list[int] = []
        # 其中真正由 n-gram 命中的个数（补位的不算），只用于统计接受率
        self.num_real_drafts = 0
        # PLD 的 n-gram 索引：{(n, ngram): 最近一次出现的起始下标}。
        # 增量维护，draft 时 O(max_n) 次字典查询 —— 不能在热路径里扫全序列。
        self._ngram: dict[tuple, int] = {}
        self._ngram_next = 0      # 已建索引的 token 尾部位置（不含）

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    # ---------------------------------------------------------- 步 12：PLD
    def _index_upto(self, end: int):
        """把结束位置落在 `[_ngram_next, end)` 的 n-gram 登记进索引（增量、只加不回退）。"""
        for i in range(self._ngram_next, end):
            for n in range(1, min(self.pld_max_n, i + 1) + 1):
                self._ngram[(n, tuple(self.token_ids[i - n + 1:i + 1]))] = i - n + 1
        if end > self._ngram_next:
            self._ngram_next = end

    def lookup_draft(self, gamma: int) -> list[int]:
        """prompt lookup：拿 context 末尾 n 个 token 去历史里找最近一次出现，抄后面 γ 个。

        索引只建到「结束位置 < L-1」，所以末尾那段（含最后一个 token）不会命中自己 ——
        否则就是把要预测的答案当成 draft，接受率会虚高到 100%。
        """
        if gamma <= 0:
            return []
        L = len(self.token_ids)
        self._index_upto(max(0, L - 1))
        t = self.token_ids
        for n in range(min(self.pld_max_n, L - 1), 0, -1):
            j = self._ngram.get((n, tuple(t[L - n:])))
            if j is None:
                continue
            s = j + n
            d = t[s:s + gamma]
            if d:
                return d
        return []

    def __getstate__(self):
        last_state = self.last_token if not self.is_prefill else self.token_ids
        # 步 12：draft 是调度器（rank 0）算的，worker 得拿到才能组 verify 的输入
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens,
                self.num_scheduled_tokens, self.block_table, last_state,
                self.draft_token_ids)

    def __setstate__(self, state):
        (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens,
         self.num_scheduled_tokens, self.block_table, last_state,
         self.draft_token_ids) = state
        # n-gram 索引只在 rank 0 用（draft 由它生成），worker 侧不需要重建
        self._ngram = {}
        self._ngram_next = 0
        if isinstance(last_state, list):
            self.token_ids = last_state
            self.last_token = self.token_ids[-1]
        else:
            self.token_ids = []
            self.last_token = last_state
