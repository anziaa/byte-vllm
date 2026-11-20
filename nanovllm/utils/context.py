from dataclasses import dataclass
import torch


@dataclass(slots=True)
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    # 步 12：verify 要 **每个位置** 的 logits（prefill 只要每段最后一个）
    all_positions: bool = False
    # 步 12：本次前向是 PLD 的 verify。必须绕开 prefill 编译图 ——
    # 那张图是按 bf16 权重 trace 的，而 verify 要用 int8（与基线 decode 同精度）；
    # dynamo 会 guard `int8_active` 这个模块 bool，两者交替 = 每步重编译。
    verify: bool = False

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None, all_positions=False, verify=False):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables, all_positions, verify)

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
