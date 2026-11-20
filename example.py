"""最小可用示例：加载 Qwen3-0.6B 生成两段文本。

    python example.py --model models/Qwen3-0.6B
"""
import argparse
import os

from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=os.environ.get("NANOVLLM_MODEL", "models/Qwen3-0.6B"),
        help="本地模型目录（默认读 $NANOVLLM_MODEL 或 models/Qwen3-0.6B）",
    )
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    path = args.model
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=args.max_tokens)
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
