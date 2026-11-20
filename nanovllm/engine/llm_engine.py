import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        Sequence.block_size = config.kvcache_block_size
        Sequence.pld_max_n = config.pld_max_n     # 步 12：draft 的回看阶数
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        # 步 12 的**契约**：PLD 的 verify 恒用 argmax，接受规则（draft == verify）只在
        # greedy 下等价于「一步一个 token 的贪心解码」。本引擎的 Sampler 是按
        # temperature 采样的、并且**明确禁止 greedy**（SamplingParams 里有断言），
        # 所以采样路径上开 PLD 会静默退化成 greedy 输出 —— 因此默认关闭，
        # 打开时把契约喊出来。采样要等价必须走 rejection sampling（未实现）。
        if config.pld_schedule:
            print("[pld] 注意：PLD 只对 greedy（argmax）等价；采样路径需要 rejection "
                  f"sampling，本引擎未实现。当前分档 {config.pld_tiers}", flush=True)
        # 步 12：记账用的步数计数器（按**引擎步**，不是按序列·步 —— 两者混用会把
        # PLD 的加速比算错，第一批冒烟就是因为这个把 1.6 token/步读成了 23 ms/步）
        self.steps_prefill = 0
        self.steps_decode = 0
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        # 步 12：decode 一步可能吐出多个 token（PLD），吞吐必须按「真正产出的 token 数」
        # 记，不能用 -len(seqs) —— 否则投机解码的加速比会被系统性低估。
        tokens_before = sum(seq.num_tokens for seq in seqs)
        scheduled = sum(seq.num_scheduled_tokens for seq in seqs)
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        if is_prefill:
            self.steps_prefill += 1
        else:
            self.steps_decode += 1
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        if is_prefill:
            num_tokens = scheduled
        else:
            num_tokens = -(sum(seq.num_tokens for seq in seqs) - tokens_before)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
