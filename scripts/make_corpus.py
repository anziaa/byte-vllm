#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_corpus.py —— 生成「固定 PPL 评估语料」，一次生成、之后冻结进仓库

为什么要有这一步（步 3 的前置）
------------------------------
量化（步 4/5/11）唯一的风险是质量，而"我觉得还行"不是证据。要判断
int8/int4/fp8 到底掉了多少，必须有一份**永远不变**的语料：它一变，两次
测量之间的差值就分不清是量化造成的还是语料造成的。

所以语料只生成一次，结果作为 token id 落盘（`*.ids.json`）并提交进仓库；
之后的 quality_eval.py 只读这份 id，不再碰文本清洗逻辑。

两类语料
--------
* EN：wikitext-2 (raw 版本经 Moses 分词后的 train.txt)。英文语言模型的公共
  基准，维基百科正文，词分布自然。首次运行从 GitHub 代理下载并缓存到
  `assets/corpus/_src/`（该目录不进 git），之后完全离线。
* ZH：本仓库的中文技术文档（docs/*.md + results/RESULTS.md + README.md），
  去掉表格、代码块、HTML 之后剩下的正文。

  ⚠️ 中文这份**不是公开基准**，ppl 绝对值不能和论文数字比。它存在的意义是
  「同一份语料上横向对比」——bf16 与 int8/int4 用完全相同的 32768 个 token，
  配对比较，只看相对变化。选它的理由是：真实中文技术语体、作者的写作风格
  一致、零外部依赖、永远可复现。

用法
----
    python make_corpus.py --docs-root /mnt/d/WorkBuddy/nano-vllm --out assets/corpus
    python make_corpus.py --docs-root ... --segments 64 --length 512
"""
import argparse
import glob
import hashlib
import json
import os
import re
import sys
import urllib.request

# wikitext-2 (Moses-tokenized) —— pytorch/examples 里那份，所有人做 word-level LM 都用它
EN_URLS = [
    "https://ghfast.top/https://raw.githubusercontent.com/pytorch/examples/main/"
    "word_language_model/data/wikitext-2/train.txt",
    "https://gh-proxy.com/https://raw.githubusercontent.com/pytorch/examples/main/"
    "word_language_model/data/wikitext-2/train.txt",
    "https://raw.githubusercontent.com/pytorch/examples/main/"
    "word_language_model/data/wikitext-2/train.txt",
]
ZH_GLOBS = ["docs/*.md", "results/*.md", "README.md", ".workbuddy/memory/*.md"]


# ---------------------------------------------------------------- 取源文本
def fetch_en(cache_dir: str) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, "en_wikitext2_train.txt")
    if os.path.exists(path) and os.path.getsize(path) > 1 << 20:
        return path
    last = None
    for url in EN_URLS:
        try:
            print(f"[en] 下载 {url.split('/')[2]} ...", flush=True)
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as r, open(path, "wb") as f:
                while True:
                    buf = r.read(1 << 20)
                    if not buf:
                        break
                    f.write(buf)
            print(f"[en] 缓存 {path} ({os.path.getsize(path)/2**20:.1f} MB)", flush=True)
            return path
        except Exception as e:                                     # noqa: BLE001
            last = e
            print(f"[en] 失败: {e}")
    raise SystemExit(f"三个源都下不到 wikitext-2，最后一个错误: {last}")


_CODE_FENCE = re.compile(r"^```")
_TABLE = re.compile(r"^\s*\|")
_IMG = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_HTML = re.compile(r"<[^>]+>")
_MDHEAD = re.compile(r"^\s{0,3}#{1,6}\s*")
_MDBULLET = re.compile(r"^\s*(?:[-*+]|\d+\.)\s+")
_MDBOLD = re.compile(r"\*\*|__|`")


def clean_markdown(text: str) -> str:
    """把 markdown 洗成连续中文正文。

    去掉的东西都是**会让 ppl 失真**的：表格（几乎全是数字和竖线）、代码块
    （符号串）、HTML、链接目标、行内代码。保留标题与正文。
    """
    out, in_fence = [], False
    for line in text.replace("\r\n", "\n").split("\n"):
        if _CODE_FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence or _TABLE.match(line):
            continue
        line = _IMG.sub("", line)
        line = _LINK.sub(r"\1", line)
        line = _HTML.sub("", line)
        line = _MDHEAD.sub("", line)
        line = _MDBULLET.sub("", line)
        line = _MDBOLD.sub("", line).strip()
        if not line:
            continue
        # 一行里 ASCII 占比过高 = 代码/路径/英文标识符，不属中文语料
        if sum(ch.isascii() and not ch.isspace() for ch in line) > 0.5 * len(line):
            continue
        out.append(line)
    return "\n".join(out)


def load_zh(docs_root: str, snapshot: str | None = None, refresh: bool = False) -> str:
    """中文正文。生成一次后落 `_src/zh_source.txt` 快照，之后默认读快照 —— 否则
    文档一改，重跑脚本就会得到另一份语料，"同一份语料对比"这个前提就没了。"""
    if snapshot and os.path.exists(snapshot) and not refresh:
        with open(snapshot, encoding="utf-8") as f:
            t = f.read()
        print(f"[zh] 用快照 {snapshot} ({len(t)} 字符)")
        return t
    files = []
    for pat in ZH_GLOBS:
        files.extend(sorted(glob.glob(os.path.join(docs_root, pat))))
    parts = []
    for p in files:
        rel = os.path.relpath(p, docs_root)
        with open(p, encoding="utf-8") as f:
            t = clean_markdown(f.read())
        if len(t) < 200:
            continue
        parts.append(t)
        print(f"[zh] {rel}: 清洗后 {len(t)} 字符")
    text = "\n".join(parts)
    if snapshot:
        os.makedirs(os.path.dirname(snapshot), exist_ok=True)
        with open(snapshot, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"[zh] 快照已写入 {snapshot}")
    return text


# ---------------------------------------------------------------- 切片
def slice_ids(ids: list[int], length: int, n: int) -> list[list[int]]:
    segs = []
    for i in range(0, len(ids) - length + 1, length):
        segs.append(ids[i: i + length])
        if len(segs) == n:
            break
    return segs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs-root", required=True, help="仓库根（含 docs/ results/ README.md）")
    ap.add_argument("--out", default="assets/corpus")
    ap.add_argument("--model", default="models/Qwen3-0.6B", help="取 tokenizer 用")
    ap.add_argument("--segments", type=int, default=64)
    ap.add_argument("--length", type=int, default=512)
    ap.add_argument("--refresh-zh", action="store_true", help="忽略 zh 快照，按当前文档重新生成")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    os.makedirs(args.out, exist_ok=True)

    # ---- EN
    raw = fetch_en(os.path.join(args.out, "_src"))
    lines = []
    with open(raw, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("=") or "<unk>" in line:
                continue
            lines.append(line)
    en_text = "\n".join(lines)
    en_ids = tok.encode(en_text, add_special_tokens=False)
    en_segs = slice_ids(en_ids, args.length, args.segments)
    print(f"[en] 可用 {len(en_ids)} token -> {len(en_segs)} 段 × {args.length}")

    # ---- ZH
    zh_text = load_zh(args.docs_root,
                      os.path.join(args.out, "_src", "zh_source.txt"),
                      refresh=args.refresh_zh)
    zh_ids = tok.encode(zh_text, add_special_tokens=False)
    zh_segs = slice_ids(zh_ids, args.length, args.segments)
    print(f"[zh] 可用 {len(zh_ids)} token -> {len(zh_segs)} 段 × {args.length}")

    for name, segs in (("en", en_segs), ("zh", zh_segs)):
        if len(segs) < args.segments:
            print(f"[!] {name} 只凑到 {len(segs)}/{args.segments} 段，ppl 的 token 数会偏少")
        ids_path = os.path.join(args.out, f"{name}_{args.length}.ids.json")
        txt_path = os.path.join(args.out, f"{name}_{args.length}.txt")
        with open(ids_path, "w", encoding="utf-8") as f:
            json.dump(segs, f)
        with open(txt_path, "w", encoding="utf-8") as f:
            for s in segs:
                f.write(tok.decode(s).replace("\n", " ") + "\n")
        sha = hashlib.sha256(json.dumps(segs, separators=(",", ":")).encode()).hexdigest()[:16]
        print(f"[{name}] {ids_path} sha={sha}  ({len(segs)*args.length} token)")


if __name__ == "__main__":
    sys.exit(main())
