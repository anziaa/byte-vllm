"""最小冒烟测试：加载模型、生成一段中文、打印耗时。

    python scripts/smoke_test.py --model models/Qwen3-0.6B
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanovllm import LLM, SamplingParams  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=os.environ.get("NANOVLLM_MODEL", "models/Qwen3-0.6B"),
        help="本地模型目录（默认读 $NANOVLLM_MODEL 或 models/Qwen3-0.6B）",
    )
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    llm = LLM(
        args.model,
        enforce_eager=True,
        gpu_memory_utilization=0.85,
        max_model_len=2048,
    )
    sp = SamplingParams(temperature=0.6, max_tokens=args.max_tokens)

    t0 = time.time()
    out = llm.generate(["用三句话解释大模型推理引擎中的 PagedAttention。"], sp)
    print(out[0]["text"])
    print("\n耗时 %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
