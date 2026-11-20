#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成 3 张「机制/归因」类 SVG（fig5/6/7），并重建 results/figs/index.html。

与 charts.py 的分工:
    charts.py   —— 数据图, 从 bench_results.csv 出一条曲线一张图（fig1~fig4）
    本脚本      —— 归因图, 画的是"为什么慢"而非"慢多少"（fig5~fig7）
                   fig7 的数据来自 results/profile_phases.csv（实测）
                   fig5/fig6 的常量来自 RESULTS.md 已归档的实测值, 逐处注明来源

为什么必须落成静态 SVG:
    show_widget 画的是内联 SVG, 只活在对话里; README / GitHub 需要独立文件。
    GitHub 原生渲染 .svg, 且矢量放大不糊, 所以不导 PNG。

用法:
    python scripts/figs_concept.py                       # -> results/figs/
    python scripts/figs_concept.py --out docs/figs       # 换输出目录
    python scripts/figs_concept.py --skip-index          # 只出图不重建索引

零第三方依赖, 纯标准库。
"""
import argparse
import csv
import html
import os
import statistics

# ---------------------------------------------------------------- 主题
INK = "#1e293b"
MUTED = "#64748b"
FAINT = "#94a3b8"
GRID = "#e2e8f0"
AXIS = "#cbd5e1"
BG = "#ffffff"

BLUE = "#2563eb"
BLUE_M = "#93c5fd"
BLUE_L = "#dbeafe"
AMBER = "#d97706"
AMBER_M = "#fcd34d"
AMBER_L = "#fef3c7"
GREEN = "#059669"
GREEN_M = "#6ee7b7"
GREEN_L = "#d1fae5"
PURPLE = "#7c3aed"
PURPLE_M = "#c4b5fd"
PURPLE_L = "#ede9fe"
RED = "#dc2626"
SLATE = "#475569"
SLATE_L = "#f1f5f9"

FONT = "'Microsoft YaHei','PingFang SC','Noto Sans CJK SC',system-ui,sans-serif"


# ---------------------------------------------------------------- 原语
def esc(s):
    return html.escape(str(s), quote=True)


def text(x, y, s, size=12, color=INK, anchor="start", weight="400", opacity=None, ls=None):
    o = f' opacity="{opacity}"' if opacity is not None else ""
    l = f' letter-spacing="{ls}"' if ls is not None else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{color}" '
            f'text-anchor="{anchor}" font-weight="{weight}" font-family="{FONT}"{o}{l}>{esc(s)}</text>')


def rect(x, y, w, h, fill, rx=0, stroke="none", sw=1, opacity=None, dash=None):
    o = f' opacity="{opacity}"' if opacity is not None else ""
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(w,0):.1f}" height="{h:.1f}" '
            f'fill="{fill}" rx="{rx}" stroke="{stroke}" stroke-width="{sw}"{o}{d}/>')


def line(x1, y1, x2, y2, color=AXIS, w=1, dash=None, opacity=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    o = f' opacity="{opacity}"' if opacity is not None else ""
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{color}" stroke-width="{w}"{d}{o}/>')


def circle(x, y, r=4, fill=BLUE, stroke="#fff", sw=2):
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'


def polyline(pts, color, w=2.2, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    p = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    return (f'<polyline points="{p}" fill="none" stroke="{color}" '
            f'stroke-width="{w}" stroke-linejoin="round" stroke-linecap="round"{d}/>')


def wrap_svg(h, body, label):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 920 {h}" '
            f'width="920" height="{h}" role="img" aria-label="{esc(label)}">'
            f'<title>{esc(label)}</title>'
            f'<rect width="920" height="{h}" fill="{BG}"/>'
            + "".join(body) + "</svg>")


def head(title, note, y=40):
    """统一的标题 + 副标题"""
    out = [text(30, y, title, size=20, weight="700", ls="-0.3")]
    if note:
        out.append(text(30, y + 24, note, size=12, color=MUTED))
    return out


# ---------------------------------------------------------------- 图 5
def fig_step_cost(out):
    """
    一张图说清 decode 一步的 6.69ms 里 94% 在等显存，以及 1.19GB 权重怎么切。

    常量来源（results/RESULTS.md）：
      每步权重 1.19 GB = MLP 264M(0.53GB) + Attn 176M(0.35GB) + lm_head 156M(0.31GB)
      有效带宽 188 GB/s → 1.19GB / 188GB/s = 6.33 ms
      bs=1 实测墙钟 6.69 ms（表 2 / profile_phases E 节 6.69ms）
    """
    W, H = 920, 552
    b = [rect(0, 0, W, H, BG)]
    b += head("decode 一步 6.69 ms，94% 的时间在等显存",
              "Qwen3-0.6B · RTX 4070 Laptop 8GB · 有效带宽 188 GB/s（峰值的 73%）· bs=1 · 走 CUDA graph")

    X0, BARW = 30, 860

    # ---- ① 一步墙钟的构成
    y = 108
    b.append(text(X0, y, "① 一步的墙钟怎么花掉", size=13, weight="600"))
    b.append(text(X0 + BARW, y, "总 6.69 ms", size=11.5, color=MUTED, anchor="end"))

    y += 14
    w_w = BARW * (6.33 / 6.69)
    w_o = BARW - w_w
    b.append(rect(X0, y, w_w, 34, BLUE, rx=4))
    b.append(rect(X0 + w_w + 1.5, y, w_o - 1.5, 34, AMBER, rx=4))
    b.append(text(X0 + 14, y + 22, "读 1.19 GB 权重  →  6.33 ms", size=12.5, color="#fff", weight="600"))
    b.append(text(X0 + w_w - 14, y + 22, "94.6%", size=12.5, color="#fff", weight="700", anchor="end"))

    # KV 段太窄放不下字，另起一行用小色块说明
    b.append(rect(X0, y + 44, 9, 9, AMBER, rx=2))
    b.append(text(X0 + 15, y + 52,
                  "KV 读取 + 框架杂项：0.36 ms（5.4%）—— 单流时可忽略，高并发时才变成主角",
                  size=10.5, color=AMBER, weight="600"))

    b.append(text(X0, y + 76, "有效带宽 188 GB/s 由 C=1 与 C=64 两组独立实测反推出同一个常数，"
                              "说明瓶颈不是算力、是显存搬运。", size=11.5, color=MUTED))

    # ---- ② 权重三段
    y += 110
    b.append(text(X0, y, "② 这 1.19 GB 分三段，决定了量化该先砍哪里", size=13, weight="600"))
    b.append(text(X0 + BARW, y, "每步都完整读一遍", size=11.5, color=MUTED, anchor="end"))

    y += 14
    segs = [
        ("MLP", 0.53, 264, 44, BLUE_M, "#1e40af"),
        ("Attention", 0.35, 176, 30, AMBER_M, "#92400e"),
        ("lm_head", 0.31, 156, 26, PURPLE_M, "#5b21b6"),
    ]
    tot = sum(s[1] for s in segs)
    cx = X0
    for name, gb, m, pct, fill, dark in segs:
        w = BARW * gb / tot
        b.append(rect(cx, y, w - 2, 44, fill, rx=4))
        b.append(text(cx + 14, y + 20, f"{name}  {gb:.2f} GB", size=12.5, color=dark, weight="700"))
        b.append(text(cx + 14, y + 36, f"{m}M 参数 · {pct}%", size=10.5, color=dark, opacity=0.85))
        cx += w
    b.append(rect(X0, y, BARW, 44, "none", rx=4, stroke=AXIS, sw=1))

    # 重点标注 lm_head
    b.append(text(X0 + BARW, y + 72, "只为算 bs 行 logits，却读满 311 MB —— 量化第一刀",
                  size=11, color=PURPLE, anchor="end", weight="600"))

    # ---- ③ 三个攻击点
    y += 100
    b.append(text(X0, y, "③ 三个互不重叠的攻击面", size=13, weight="600"))
    y += 14

    cards = [
        ("Q", "量化权重", BLUE_L, BLUE, "砍 ② 整条",
         ["int8 全线：1.19 → 0.60 GB", "单流 148 → ≥240 tok/s", "先过 ppl 涨幅 <1% 这关"],
         "风险：中（质量）"),
        ("K", "fp8 KV + 自写 kernel", AMBER_L, AMBER, "砍 KV + 扩容",
         ["KV 每 token 224 KiB → 112", "C=64 吞吐 ×1.4~1.5", "突破 block 256 的硬限制"],
         "风险：高（数值）"),
        ("P", "消 prefill 的提交开销", GREEN_L, GREEN, "另一个维度",
         ["prefill 提交 30 ms → ~1 ms", "TTFT@2048 80 → ~46 ms", "prefix 命中 40.8 → ~8 ms"],
         "风险：低–中"),
    ]
    cw, gap = 276, 16
    for i, (tag, name, bg, fg, target, bullets, risk) in enumerate(cards):
        cx = X0 + i * (cw + gap)
        b.append(rect(cx, y, cw, 138, bg, rx=8))
        b.append(rect(cx, y, 4, 138, fg, rx=2))
        b.append(text(cx + 16, y + 26, tag, size=15, weight="700", color=fg))
        b.append(text(cx + 34, y + 26, name, size=12.5, weight="600", color=INK))
        b.append(text(cx + cw - 14, y + 26, target, size=9.5, color=fg, anchor="end", opacity=0.85))
        for j, bl in enumerate(bullets):
            b.append(text(cx + 16, y + 55 + j * 21, "· " + bl, size=11, color=SLATE))
        b.append(text(cx + 16, y + 122, risk, size=10, color=MUTED))

    b.append(text(X0, H - 16, "数据：results/RESULTS.md 表 1–表 4、表 3b；results/profile_phases.csv",
                  size=10, color=FAINT))

    p = os.path.join(out, "fig5_step_cost.svg")
    open(p, "w", encoding="utf-8").write(wrap_svg(H, b, "decode 一步的耗时构成与三个优化攻击面"))
    return p


# ---------------------------------------------------------------- 图 6
def fig_launch_timeline(out):
    """
    eager 336 次提交 vs CUDA graph 1 次提交的时序对比。

    常量来源（RESULTS.md 表 2 / 表 3b）：
      eager decode 每 step 墙钟 ≈ 30 ms，且与 batch 无关
      graph 后 bs=1 每 step 6.6 ms（其中 CPU 提交 0.3 ms）
      eager 的 336 次提交里，真正 cuLaunchKernel 只占 4 ms，其余是 Python/guard/dispatch
    """
    W, H = 920, 576
    b = [rect(0, 0, W, H, BG)]
    b += head("eager 与 CUDA graph 的提交时序：不是快一点，是两种形态",
              "同一次 decode 前向 · 28 层 × 12 个 op = 336 次 Python 提交 · GPU 净执行时间两边相同（≈6.6 ms）")

    # 时间轴
    X0, X1 = 132, 876
    TMAX = 34.0                       # ms
    YAX = 490
    def tx(t):
        return X0 + (X1 - X0) * t / TMAX

    for t in range(0, 36, 5):
        x = tx(t)
        b.append(line(x, 104, x, YAX, color=GRID, w=1))
        b.append(text(x, YAX + 16, f"{t}", size=10.5, color=FAINT, anchor="middle"))
    b.append(text((X0 + X1) / 2, YAX + 36, "时间（ms）", size=11, color=MUTED, anchor="middle"))
    b.append(line(X0, YAX, X1, YAX, color=AXIS, w=1))

    # ---- 上：eager
    b.append(text(30, 112, "eager", size=13.5, weight="700", color=RED))
    b.append(text(30, 130, "每 step 30 ms", size=10.5, color=MUTED))
    b.append(text(30, 145, "与 batch 无关", size=10.5, color=FAINT))

    b.append(text(30, 196, "CPU 提交", size=10.5, color=MUTED))
    b.append(text(30, 211, "336 次", size=10.5, color=FAINT, weight="600"))

    # CPU：浅底 + 336 根深红细竖条，0 ~ 30.4 ms
    import random
    rnd = random.Random(7)
    cy = 200
    xe = tx(30.4)
    b.append(rect(tx(0), cy - 10, xe - tx(0), 20, "#fee2e2", rx=2))
    n_sub = 336
    for i in range(n_sub):
        x = tx(30.4 * i / n_sub)
        wdt = max((X1 - X0) * 30.4 / TMAX / n_sub - 0.2, 0.7)
        hh = 14 + rnd.random() * 5
        b.append(rect(x, cy - hh / 2, wdt, hh, RED, rx=0, opacity=0.9))
    b.append(text(xe - 8, cy + 4, "30.4 ms", size=10.5, color="#fff", weight="700", anchor="end"))

    # GPU：28 段碎块，中间被 launch 切开
    b.append(text(30, 246, "GPU 执行", size=10.5, color=MUTED))
    b.append(text(30, 261, "碎成 28 段", size=10.5, color=FAINT, weight="600"))
    gy = 250
    seg_w = (X1 - X0) * 6.6 / TMAX / 28
    for i in range(28):
        x = tx(30.4 * i / 28) + 1.0
        b.append(rect(x, gy - 9, max(seg_w, 1.6), 18, GREEN, rx=1.5, opacity=0.9))

    b.append(text(X0, 288,
                  "墙钟 30.4 ms，其中 GPU 只忙 6.6 ms —— 空闲 78%，全部在等 CPU 把下一个 kernel 交上去",
                  size=11.5, color=RED, weight="600"))

    # ---- 下：graph
    b.append(rect(X0 - 6, 302, X1 - X0 + 12, 174, SLATE_L, rx=8, opacity=0.75))
    b.append(text(30, 330, "CUDA graph", size=13.5, weight="700", color=BLUE))
    b.append(text(30, 348, "每 step 6.6 ms", size=10.5, color=MUTED))
    b.append(text(30, 363, "bs=1", size=10.5, color=FAINT))

    b.append(text(30, 396, "CPU 提交", size=10.5, color=MUTED))
    b.append(text(30, 411, "1 次", size=10.5, color=FAINT, weight="600"))
    b.append(rect(tx(0), 392, max(tx(0.32) - tx(0), 3), 16, RED, rx=2))
    b.append(text(tx(0.32) + 8, 404, "cudaGraphLaunch ≈ 0.3 ms，随后 CPU 立即返回",
                  size=10.5, color=RED, weight="600"))

    b.append(text(30, 452, "GPU 执行", size=10.5, color=MUTED))
    b.append(text(30, 467, "连续 1 段", size=10.5, color=FAINT, weight="600"))
    b.append(rect(tx(0.4), 447, tx(6.9) - tx(0.4), 18, GREEN, rx=2))
    b.append(text(tx(6.9) + 8, 460, "6.6 ms 连续跑完，中间没有任何等待",
                  size=10.5, color=GREEN, weight="600"))

    # ---- 底部结论
    b.append(rect(X0 - 6, 540, X1 - X0 + 12, 26, BLUE_L, rx=6))
    b.append(text(X0 + 6, 558,
                  "同样 336 个 kernel、同样的 GPU 工作量：收益全部来自「把 336 次 CPU 提交压成 1 次」，"
                  "batch 一大带宽重新主导，收益递减",
                  size=11, color="#1e40af", weight="600"))

    p = os.path.join(out, "fig6_launch_timeline.svg")
    open(p, "w", encoding="utf-8").write(wrap_svg(H, b, "eager 与 CUDA graph 的提交时序对比"))
    return p


# ---------------------------------------------------------------- 图 7
def load_profile(csv_path):
    """读 profile_phases.csv，返回 (sweep, layers)"""
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    by_n, by_l = {}, {}
    for r in rows:
        if r.get("kind") != "prefill":
            continue
        s, t = float(r["submit_ms"]), float(r["tail_ms"])
        if r["phase"] == "sweep" and int(r["n"]) <= 3584:
            by_n.setdefault(int(r["n"]), []).append((s, t))
        elif r["phase"] == "layers28" or r["phase"].startswith("layers"):
            by_l.setdefault(int(r["layers"]), []).append(s)

    sweep = [(n,
              statistics.median([a for a, _ in v]),
              statistics.median([b for _, b in v]))
             for n, v in sorted(by_n.items())]
    layers = [(k, statistics.median(v)) for k, v in sorted(by_l.items())]
    return sweep, layers


def fig_prefill_submit(sweep, layers, out):
    """
    A: prefill 的 CPU 提交 vs token 数 —— 平线，与 n 无关
    B: 提交 vs 层数 —— 严格线性，1.12 ms/层

    数据来自 results/profile_phases.csv（sweep: 5 次/长度取中位；layers: 各 2 次取中位）
    """
    W, H = 920, 664
    b = [rect(0, 0, W, H, BG)]
    b += head("prefill 的慢不在 GPU：提交开销是一条平线，且严格按层线性",
              "插桩在 run_model 外层分离 CPU 提交与 GPU 执行 · 虚线为最小二乘拟合")

    # ================= A
    AX0, AX1 = 96, 860
    AY0, AY1 = 132, 326            # y 轴 132(顶) ~ 326(底)
    ymax = 130.0
    def ay(v): return AY1 - (AY1 - AY0) * v / ymax

    b.append(text(30, 108, "A · 提交与 token 数无关（bs=1）", size=13, weight="600"))

    for v in range(0, 131, 30):
        yy = ay(v)
        b.append(line(AX0, yy, AX1, yy, color=GRID if v else AXIS, w=1))
        b.append(text(AX0 - 10, yy + 4, f"{v}", size=10.5, color=FAINT, anchor="end"))
    b.append(text(AX0 - 10, AY0 - 16, "ms", size=10.5, color=MUTED, anchor="end"))
    b.append(text(AX1, AY0 - 16, "prompt 长度（token）", size=10.5, color=MUTED, anchor="end"))

    # x 均匀分布（token 数跨度大，等距分类更可读）
    n = len(sweep)
    def ax(i): return AX0 + (AX1 - AX0) * (i / (n - 1)) if n > 1 else AX0

    # tail 面积 + 线
    tail_pts = [(ax(i), ay(t)) for i, (_, _, t) in enumerate(sweep)]
    sub_pts = [(ax(i), ay(s)) for i, (_, s, _) in enumerate(sweep)]
    b.append('<path d="M %.1f,%.1f %s L %.1f,%.1f L %.1f,%.1f Z" fill="%s" opacity="0.5"/>' % (
        tail_pts[0][0], AY1,
        " ".join(f"L {x:.1f},{y:.1f}" for x, y in tail_pts),
        tail_pts[-1][0], AY1, tail_pts[0][0], AY1, AMBER_L))
    b.append(polyline(tail_pts, AMBER, w=2.4))
    b.append(polyline(sub_pts, BLUE, w=2.4))
    b.append(line(AX0, ay(29.9), AX1, ay(29.9), color=BLUE, w=1.2, dash="5 4", opacity=0.55))

    for i, (nn, s, t) in enumerate(sweep):
        b.append(circle(ax(i), ay(t), 3.6, AMBER))
        b.append(circle(ax(i), ay(s), 3.6, BLUE))
        b.append(text(ax(i), AY1 + 18, f"{nn}", size=10.5, color=MUTED, anchor="middle"))

    # 图例
    lx, ly = AX0 + 12, AY0 + 10
    b.append(rect(lx, ly - 10, 250, 52, "#fff", rx=6, stroke=GRID))
    b.append(line(lx + 12, ly + 2, lx + 36, ly + 2, color=BLUE, w=2.4))
    b.append(text(lx + 44, ly + 6, "CPU 提交（submit）", size=10.5, color=INK))
    b.append(line(lx + 12, ly + 24, lx + 36, ly + 24, color=AMBER, w=2.4))
    b.append(text(lx + 44, ly + 28, "GPU 执行（tail）", size=10.5, color=INK))

    # 标注：蓝线注释放在平线的左段上方（右段会被橙色曲线穿过）
    b.append(text(AX0 + 40, ay(29.9) - 9, "≈30 ms，16 → 2048 token 几乎不变",
                  size=10.5, color=BLUE, anchor="start", weight="600"))
    b.append(text(ax(n - 1) - 8, ay(sweep[-1][2]) - 12, "GPU 到 3584 才真正主导",
                  size=10.5, color=AMBER, anchor="end", weight="600"))

    b.append(text(30, 372,
                  "16 token 的 prefill 要 30 ms，GPU 只占 1.3 ms —— 96% 的时间 GPU 在等 Python 喂 kernel。",
                  size=11.5, color=SLATE))

    # ================= B
    BX0, BX1 = 96, 860
    BY0, BY1 = 434, 584
    b.append(text(30, 408, "B · 提交开销严格按层线性（seq_len=2048，截断 model.layers）", size=13, weight="600"))

    bymax = 36.0
    def by(v): return BY1 - (BY1 - BY0) * v / bymax

    for v in range(0, 37, 9):
        yy = by(v)
        b.append(line(BX0, yy, BX1, yy, color=GRID if v else AXIS, w=1))
        b.append(text(BX0 - 10, yy + 4, f"{v}", size=10.5, color=FAINT, anchor="end"))
    b.append(text(BX0 - 10, BY0 - 16, "ms", size=10.5, color=MUTED, anchor="end"))
    b.append(text(BX1, BY0 - 16, "层数", size=10.5, color=MUTED, anchor="end"))

    lmax = max(l for l, _ in layers)
    def bx(l): return BX0 + 40 + (BX1 - BX0 - 90) * l / lmax

    # 拟合
    xs = [l for l, _ in layers]
    ys = [s for _, s in layers]
    nb = sum(xs) / len(xs)
    yb = sum(ys) / len(ys)
    den = sum((x - nb) ** 2 for x in xs) or 1.0
    k = sum((x - nb) * (y - yb) for x, y in zip(xs, ys)) / den
    c0 = yb - k * nb
    b.append(line(bx(0), by(c0), bx(lmax), by(c0 + k * lmax), color=BLUE, w=1.3, dash="5 4", opacity=0.6))

    # 拟合说明放左上空白区（线从左下升到右上，左上必空）
    b.append(text(BX0 + 56, BY0 + 16, f"拟合 {k:.2f} ms/层，0 层截距 {c0:+.1f} ms → 开销几乎全在层内",
                  size=10.5, color=BLUE, weight="600"))

    pts = [(bx(l), by(s)) for l, s in layers]
    b.append(polyline(pts, GREEN, w=2.2))
    for (l, s) in layers:
        b.append(circle(bx(l), by(s), 4, GREEN))
        b.append(text(bx(l), by(s) - 12, f"{s:.1f}", size=10.5, color=GREEN, anchor="middle", weight="600"))
        b.append(text(bx(l), BY1 + 18, f"{l}", size=10.5, color=MUTED, anchor="middle"))

    b.append(text(30, H - 24,
                  "结论：prefill 的 30 ms 里 ≈0 花在 embedding / lm_head / 框架入口，全部在 28 层的 336 次 Python 提交上",
                  size=10.5, color=FAINT))
    b.append(text(30, H - 8,
                  "（370 次真 launch API 仅 4 ms；169 次 @torch.compile 入口 16 ms —— 省钱工具变成了收税工具）",
                  size=10.5, color=FAINT))

    p = os.path.join(out, "fig7_prefill_submit.svg")
    open(p, "w", encoding="utf-8").write(wrap_svg(H, b, "prefill 提交开销的平线与逐层线性"))
    return p


# ---------------------------------------------------------------- 索引页
CARDS = [
    ("fig1_throughput_vs_concurrency.svg", "图 1 · 吞吐随并发扩展",
     "并发 1 → 64 提升 20.1×；32 → 64 只剩 1.26×，已撞到带宽墙。"),
    ("fig2_cuda_graph.svg", "图 2 · CUDA graph 消融",
     "batch 越大加速比越小：graph 抹掉的是每 step 固定的提交开销，batch 一大就重新变成带宽受限。"),
    ("fig3_ablation.svg", "图 3 · 参数消融（并发 64）",
     "block_size 与 max_num_seqs=64 无影响；mem 0.75 与 seqs 16 是真正让吞吐掉下来的两个旋钮。"),
    ("fig4_prefix_cache.svg", "图 4 · prefix caching 开关对照",
     "2048 token 共享前缀：关闭 83.7 ms → 命中 40.8 ms（2.05×），稳态命中率 100%；未命中代价 0.08 ms。"),
    ("fig5_step_cost.svg", "图 5 · decode 一步的耗时构成",
     "6.69 ms 里 94% 在等显存；1.19 GB 权重分三段，lm_head 占 26% 却只产 bs 行 logits —— 量化第一刀在这里。"),
    ("fig6_launch_timeline.svg", "图 6 · eager 与 CUDA graph 的时序对比",
     "同样 336 个 kernel、同样的 GPU 工作量：收益全部来自把 336 次 CPU 提交压成 1 次。"),
    ("fig7_prefill_submit.svg", "图 7 · prefill 的开销归因",
     "提交是 30 ms 的平线（与 token 数无关），且严格按层线性 1.12 ms/层；0 层截距≈0。"),
    ("fig8_prefill_compile.svg", "图 8 · 步 2：整层编译的收益与边界",
     "提交 38→17 ms，但端到端只在 512 与 prefix 命中兑现（−39%/−43%），2048/3584 不动 —— 长 prompt 只能走量化。"),
]

KPIS = [
    ("2,987", "tok/s @ 并发 64"),
    ("148.3", "tok/s 单流"),
    ("188", "GB/s 有效带宽<br>≈ 峰值 256 的 73%"),
    ("51%", "prefix cache<br>TTFT 降幅"),
    ("−55%", "prefill CPU 提交<br>38→17 ms（步 2）"),
]


def write_index(outdir):
    blocks = []
    i = 0
    for fn, title, note in CARDS:
        p = os.path.join(outdir, fn)
        if not os.path.exists(p):
            continue
        i += 1
        svg = open(p, encoding="utf-8").read()
        blocks.append(
            f'<section class="card"><div class="cap"><span class="idx">{i}</span>'
            f'<h2>{esc(title)}</h2></div>{svg}<p class="note">{esc(note)}</p></section>')

    kpi_html = "".join(
        f'<div class="kpi"><div class="v">{v}</div><div class="l">{l}</div></div>'
        for v, l in KPIS)

    doc = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>nano-vllm 性能报告 · 全部图表</title>
<style>
  :root {{ color-scheme: light; }}
  body {{ margin:0; padding:40px 20px 60px; background:#f8fafc; color:#1e293b;
         font-family:'Microsoft YaHei','PingFang SC',system-ui,sans-serif; }}
  .wrap {{ max-width:900px; margin:0 auto; }}
  h1 {{ font-size:22px; margin:0 0 6px; letter-spacing:-.2px; }}
  .sub {{ color:#64748b; font-size:13.5px; margin-bottom:26px; line-height:1.7; }}
  .kpis {{ display:grid; grid-template-columns:repeat(5,1fr); gap:12px; margin-bottom:28px; }}
  .kpi {{ background:#fff; border:1px solid #e2e8f0; border-radius:10px; padding:14px 16px; }}
  .kpi .v {{ font-size:22px; font-weight:700; color:#2563eb; letter-spacing:-.5px; }}
  .kpi .l {{ font-size:11.5px; color:#64748b; margin-top:3px; line-height:1.5; }}
  .card {{ background:#fff; border:1px solid #e2e8f0; border-radius:12px;
           padding:18px 18px 14px; margin-bottom:18px; }}
  .cap {{ display:flex; align-items:center; gap:9px; margin-bottom:6px; }}
  .idx {{ width:20px;height:20px;border-radius:6px;background:#dbeafe;color:#2563eb;
          font-size:12px;font-weight:700;display:grid;place-items:center; }}
  h2 {{ font-size:14px; margin:0; font-weight:600; }}
  svg {{ width:100%; height:auto; display:block; }}
  .note {{ color:#64748b; font-size:12.5px; margin:8px 2px 0; line-height:1.6; }}
  footer {{ color:#94a3b8; font-size:11.5px; margin-top:26px; line-height:1.7; }}
  @media (max-width:700px) {{ .kpis {{ grid-template-columns:repeat(2,1fr); }} }}
</style></head><body><div class="wrap">
<h1>nano-vllm 性能报告 · 全部图表</h1>
<div class="sub">Qwen3-0.6B · RTX 4070 Laptop 8GB（256 GB/s）· WSL2 Ubuntu 22.04 · torch 2.5.1+cu124 · flash-attn 2.8.3.post2<br>
图 1–4 为数据图（由 <code>scripts/charts.py</code> 从 bench_results.csv 生成）；图 5–7 为归因图（由 <code>scripts/figs_concept.py</code> 生成）。</div>
<div class="kpis">{kpi_html}</div>
{''.join(blocks)}
<footer>数据：results/bench_results.csv、results/bench_results_v2.csv、results/profile_phases.csv<br>
所有图均为纯标准库手写 SVG（无 matplotlib 依赖），可直接嵌入 README。<br>
重建索引：<code>python scripts/figs_concept.py</code></footer>
</div></body></html>"""

    p = os.path.join(outdir, "index.html")
    open(p, "w", encoding="utf-8").write(doc)
    return p


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--out", default=os.path.join(here, "results", "figs"))
    ap.add_argument("--profile", default=os.path.join(here, "results", "profile_phases.csv"))
    ap.add_argument("--skip-index", action="store_true")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    print(fig_step_cost(a.out))
    print(fig_launch_timeline(a.out))

    if os.path.exists(a.profile):
        sweep, layers = load_profile(a.profile)
        print(f"  sweep 点: {[(n, round(s,1), round(t,1)) for n, s, t in sweep]}")
        print(f"  layers 点: {[(l, round(s,2)) for l, s in layers]}")
        print(fig_prefill_submit(sweep, layers, a.out))
    else:
        print(f"!! 缺少 {a.profile}，跳过 fig7")

    if not a.skip_index:
        print(write_index(a.out))


if __name__ == "__main__":
    main()
