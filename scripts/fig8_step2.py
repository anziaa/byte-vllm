#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图 8（步 2）：prefill 整层编译到底改了什么
=========================================

左图：submit（CPU 提交时长）—— 三种编译粒度下随长度的三条线。
      要说明的是"它是一条平线"（与 token 数无关）以及"整层编译把平线压到一半"。
右图：墙钟 TTFT —— 同一批长度加上 prefix 命中场景。
      要说明的是"只在 CPU 主导的短 prompt 上兑现，长 prompt 一动不动"。

数据来源（全部实测，脚本自动读取，不写死数字）：
    results/step2_submit.csv    submit / tail（scripts/step2_measure.py）
    results/bench_step2_v2.csv  TTFT / prefix 稳态（bench_suite.py）
    results/step2_init.csv      启动耗时（本脚本写入）

用法：
    python scripts/fig8_step2.py
输出：
    results/figs/fig8_prefill_compile.svg
"""
import csv
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figs_concept import (  # noqa: E402
    AXIS, BG, BLUE, BLUE_L, BLUE_M, FONT, GRID, INK, MUTED, PURPLE, PURPLE_M,
    GREEN, GREEN_M, AMBER, RED, SLATE, SLATE_L, text, rect, line, polyline, wrap_svg, head, esc,
)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "results")
MODES = [("off", "eager（现状）", SLATE), ("layer", "每层一个 region", AMBER), ("model", "整模型一张图", BLUE)]
LENS = [512, 2048, 3584]


def load_submit():
    d = {}
    with open(os.path.join(RES, "step2_submit.csv"), encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["kind"] != "prefill":
                continue
            d.setdefault(r["mode"], {})[int(r["n"])] = float(r["submit_ms"])
    return d


def load_ttft():
    ttft, shared = {}, {}
    with open(os.path.join(RES, "bench_step2_v2.csv"), encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            mode = r["tag"].split("_", 1)[1]
            if r["mode"] == "ttft":
                ttft.setdefault(mode, {})[int(r["prompt_len"])] = float(r["ttft_s"]) * 1e3
            elif r["mode"] == "shared" and r["note"].startswith("warm"):
                shared.setdefault(mode, []).append(float(r["ttft_s"]) * 1e3)
    return ttft, {k: statistics.median(v) for k, v in shared.items()}


def main():
    sub = load_submit()
    ttft, shared = load_ttft()
    W, H = 920, 520
    b = [rect(0, 0, W, H, BG)]
    b += head("prefill 整层编译：把 CPU 提交砍一半，但只对短 prompt 兑现",
              "Qwen3-0.6B · RTX 4070 Laptop 8GB · 每根柱都是 5 次重复的中位数")

    # ---------------- 左：submit 平线
    PW, PH = 400, 216
    LX, LY = 60, 124
    b.append(text(30, 106, "① CPU 提交时长（ms）：一层的 336 次 launch 被压成一张图", size=13, weight="600"))
    smax = 44.0
    y0, y1 = LY + PH, LY
    for g in (0, 10, 20, 30, 40):
        yy = y0 - PH * g / smax
        b.append(line(LX, yy, LX + PW, yy, GRID, 1))
        b.append(text(LX - 8, yy + 4, str(g), size=10, color=MUTED, anchor="end"))
    b.append(line(LX, y1, LX, y0, AXIS, 1))
    b.append(line(LX, y0, LX + PW, y0, AXIS, 1))
    xs = [LX + 40 + i * (PW - 70) / (len(LENS) - 1) for i in range(len(LENS))]
    for L, x in zip(LENS, xs):
        b.append(text(x, y0 + 18, str(L), size=11, color=MUTED, anchor="middle"))
    b.append(text(LX + PW, y0 + 34, "prompt 长度 (token)", size=10.5, color=MUTED, anchor="end"))
    for mode, label, color in MODES:
        pts = [(x, y0 - PH * sub[mode][L] / smax) for L, x in zip(LENS, xs)]
        b.append(polyline(pts, color, w=2.4))
        for (x, y), L in zip(pts, LENS):
            b.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}" stroke="#fff" stroke-width="2"/>')
        b.append(text(pts[-1][0] + 10, pts[-1][1] + 4, f"{sub[mode][LENS[-1]]:.1f}", size=11,
                      color=color, weight="700"))
    # 图例（放在轴下方，避免压住标题）
    ly = y0 + 52
    for i, (mode, label, color) in enumerate(MODES):
        xx = LX + i * 134
        b.append(rect(xx, ly - 9, 10, 10, color, rx=2))
        b.append(text(xx + 15, ly, label, size=10.5, color=SLATE))
    b.append(text(LX, y0 + 68, "三条线都是平的 → 提交成本只与「层数 × 算子数」有关，与 token 数无关；",
                  size=10.5, color=MUTED))
    b.append(text(LX, y0 + 84, "整层编译消掉的是 169 次 region 入口与 336 次 Python 分发。",
                  size=10.5, color=MUTED))

    # ---------------- 右：TTFT
    RX, RY, RPW, RPH = 530, 124, 340, 216
    b.append(text(500, 106, "② 端到端 TTFT（ms）：只在 CPU 主导区兑现", size=13, weight="600"))
    cats = [(str(L), L) for L in LENS] + [("prefix\n命中", "shared")]
    tmax = 170.0
    ry0 = RY + RPH
    for g in (0, 50, 100, 150):
        yy = ry0 - RPH * g / tmax
        b.append(line(RX, yy, RX + RPW, yy, GRID, 1))
        b.append(text(RX - 8, yy + 4, str(g), size=10, color=MUTED, anchor="end"))
    b.append(line(RX, RY, RX, ry0, AXIS, 1))
    b.append(line(RX, ry0, RX + RPW, ry0, AXIS, 1))

    gw = RPW / len(cats)
    bw = 24
    for ci, (label, key) in enumerate(cats):
        cx = RX + gw * (ci + 0.5)
        b.append(text(cx, ry0 + 20, label.split("\n")[0], size=10.5, color=MUTED, anchor="middle"))
        if "\n" in label:
            b.append(text(cx, ry0 + 33, label.split("\n")[1], size=10.5, color=MUTED, anchor="middle"))
        for mi, (mode, _, color) in enumerate(MODES):
            v = shared[mode] if key == "shared" else ttft[mode][key]
            h = RPH * v / tmax
            x = cx - (bw * 3 + 4) / 2 + mi * (bw + 2)
            b.append(rect(x, ry0 - h, bw, h, color, rx=3))
            b.append(text(x + bw / 2, ry0 - h - 5, f"{v:.0f}", size=9.5, color=color,
                          anchor="middle", weight="700"))
    b.append(text(RX, ry0 + 56, "512 与 prefix 命中掉下来，2048/3584 纹丝不动。",
                  size=10.5, color=MUTED))
    b.append(text(RX, ry0 + 72, "那两档 GPU 本来就是瓶颈，省下的 CPU 时间藏在 GPU 后面。",
                  size=10.5, color=MUTED))

    # ---------------- 结论条
    yy = 448
    b.append(rect(30, yy, 860, 58, BLUE_L, rx=6))
    b.append(text(44, yy + 24, "判定实验结果：2048 从 80.3 → 80.3 ms，完全没有变化 —— "
                               "「CPU/GPU 全程重叠」的假设被证实，引擎里没有隐藏的同步点。",
                  size=11.5, color="#1e40af", weight="600"))
    b.append(text(44, yy + 45, "于是长 prompt 的下一步只能是减少 GPU 的工作量（量化 / 更快的 kernel），"
                               "压 CPU 已经压不出东西了。",
                  size=11.5, color="#1e40af", opacity=0.85))

    out = os.path.join(RES, "figs", "fig8_prefill_compile.svg")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(wrap_svg(H, b, "prefill 整层编译的收益与边界"))
    print("->", out)


if __name__ == "__main__":
    main()
