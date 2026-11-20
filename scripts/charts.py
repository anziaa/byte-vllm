#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 bench_results.csv 生成 4 张 SVG 图（纯标准库，无第三方依赖）

用法:
    python scripts/charts.py                    # 默认读 results/bench_results.csv
    python scripts/charts.py --csv results/bench_results.csv --out results/figs

为什么不用 matplotlib: 这台机器单连接被限速 ~200KB/s, 装 mpl 要拉 numpy/pillow
几十 MB, 而图表需求很固定 —— 手写 SVG 零依赖、可重复生成、放大不糊。
"""
import argparse
import csv
import html
import os
from collections import OrderedDict

# ---------------------------------------------------------------- 主题
INK = "#1e293b"
MUTED = "#64748b"
GRID = "#e2e8f0"
AXIS = "#cbd5e1"
BLUE = "#2563eb"
BLUE_L = "#dbeafe"
AMBER = "#d97706"
GREEN = "#059669"
GREY = "#94a3b8"
RED = "#dc2626"
FONT = "'Microsoft YaHei','PingFang SC','Noto Sans CJK SC',system-ui,sans-serif"

W, H = 780, 462
ML, MR, MT, MB = 62, 24, 46, 58


def esc(s):
    return html.escape(str(s), quote=True)


def text(x, y, s, size=12, color=INK, anchor="start", weight="400", family=FONT):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{color}" '
            f'text-anchor="{anchor}" font-weight="{weight}" font-family="{family}">{esc(s)}</text>')


def line(x1, y1, x2, y2, color=AXIS, w=1, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{w}"{d}/>'


def rect(x, y, w, h, fill, rx=3, stroke="none", opacity=None):
    o = f' opacity="{opacity}"' if opacity else ""
    return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(w,0):.1f}" height="{max(h,0):.1f}" '
            f'rx="{rx}" fill="{fill}" stroke="{stroke}"{o}/>')


def polyline(pts, color, w=2.2, dash=None):
    d = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    da = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<polyline points="{d}" fill="none" stroke="{color}" stroke-width="{w}" stroke-linejoin="round" stroke-linecap="round"{da}/>'


def circle(x, y, r=4, fill=BLUE, stroke="#fff", sw=2):
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'


def nice_max(v):
    """取一个好看的坐标上限"""
    if v <= 0:
        return 1
    import math
    exp = math.floor(math.log10(v))
    base = 10 ** exp
    for m in (1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
        if v <= base * m:
            return base * m
    return base * 10


def hframe(title, subtitle, xmax, xlabel, xticks=5):
    """横向条形图用的坐标系：值在 x 轴，y 轴是分类。返回 (svg, x 映射, 区域)"""
    pw, ph = W - ML - MR, H - MT - MB
    x0, x1 = ML, ML + pw

    def sx(v):
        return x0 + (v / xmax) * pw if xmax else x0

    s = [f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>']
    s.append(text(ML, 26, title, 15, INK, weight="600"))
    if subtitle:
        s.append(text(ML, 44, subtitle, 11.5, MUTED))
    for i in range(xticks + 1):
        v = xmax * i / xticks
        x = sx(v)
        s.append(line(x, MT + 12, x, MT + ph, GRID if i else AXIS))
        s.append(text(x, MT + ph + 18, f"{v:,.0f}", 11, MUTED, anchor="middle"))
    s.append(text(x1, MT + ph + 36, xlabel, 11, MUTED, anchor="end"))
    return "\n".join(s), sx, (x0, MT, pw, ph)


def frame(title, subtitle, ymax, ylabel, ytick_fmt=lambda v: f"{v:g}", yticks=5):
    """返回 (svg 前缀, y 映射函数, plot 区域)"""
    pw, ph = W - ML - MR, H - MT - MB
    y0, y1 = MT + ph, MT  # y0=底部, y1=顶部

    def sy(v):
        return y0 - (v / ymax) * ph if ymax else y0

    s = [f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>']
    s.append(text(ML, 26, title, 15, INK, weight="600"))
    if subtitle:
        s.append(text(ML, 44, subtitle, 11.5, MUTED))

    for i in range(yticks + 1):
        v = ymax * i / yticks
        y = sy(v)
        s.append(line(ML, y, ML + pw, y, GRID if i else AXIS))
        s.append(text(ML - 9, y + 4, ytick_fmt(v), 11, MUTED, anchor="end"))
    _lx = ML - 46
    _ly = MT + ph / 2
    s.append(f'<text x="{_lx}" y="{_ly:.0f}" font-size="11" fill="{MUTED}" '
             f'text-anchor="middle" transform="rotate(-90 {_lx} {_ly:.0f})" '
             f'font-family="{FONT}">{esc(ylabel)}</text>')
    s.append(line(ML, y0, ML + pw, y0, AXIS))
    return "\n".join(s), sy, (ML, MT, pw, ph)


def wrap(body, note=""):
    n = text(ML, H - 5, note, 10.5, MUTED) if note else ""
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
            f'viewBox="0 0 {W} {H}">\n{body}\n{n}\n</svg>\n')


# ---------------------------------------------------------------- 读数据
def load(path):
    d = OrderedDict()
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            tag = r["tag"]
            d.setdefault(tag, {"ttft": {}, "throughput": {}, "shared": []})
            if r["mode"] == "ttft":
                d[tag]["ttft"][int(r["prompt_len"])] = float(r["ttft_s"])
            elif r["mode"] == "throughput":
                d[tag]["throughput"][int(r["concurrency"])] = float(r["out_tok_s"])
            elif r["mode"] == "shared":
                d[tag]["shared"].append(float(r["ttft_s"]))
    return d


# ---------------------------------------------------------------- 图 1
def fig_throughput(d, out):
    tp = d["baseline"]["throughput"]
    cs = sorted(tp)
    ymax = nice_max(max(tp.values()) * 1.18)
    body, sy, (px, py, pw, ph) = frame(
        "吞吐 vs 并发（baseline：CUDA graph 开 · block 256 · mem 0.85）",
        "Qwen3-0.6B · prompt 256 tok / 输出 128 tok · RTX 4070 Laptop 8GB",
        ymax, "tok/s")

    n = len(cs)
    xs = [px + pw * (i + 0.5) / n for i in range(n)]
    pts = [(xs[i], sy(tp[c])) for i, c in enumerate(cs)]

    # 面积 + 折线
    area = f"{pts[0][0]:.1f},{py+ph:.1f} " + " ".join(f"{x:.1f},{y:.1f}" for x, y in pts) + f" {pts[-1][0]:.1f},{py+ph:.1f}"
    body += f'<polygon points="{area}" fill="{BLUE}" opacity="0.08"/>'
    body += polyline(pts, BLUE)

    # 每请求吞吐（虚线）
    per = [(xs[i], sy(tp[c] / c)) for i, c in enumerate(cs)]
    body += polyline(per, GREY, 1.6, dash="5 4")

    for i, c in enumerate(cs):
        x, y = pts[i]
        body += circle(x, y, 4.5, BLUE)
        body += text(x, y - 12, f"{tp[c]:.0f}", 12, BLUE, anchor="middle", weight="600")
        body += text(x, py + ph + 20, f"C={c}", 12, INK, anchor="middle")
        x2, y2 = per[i]
        body += circle(x2, y2, 3, GREY, sw=1.5)

    # 扩展比标注
    body += text(px + pw / 2, py + ph + 40,
                 f"扩展比 1→64：{tp[cs[-1]]/tp[cs[0]]:.1f}×　（32→64 仅 {tp[cs[-1]]/tp[32]:.2f}×，已逼近带宽上限）",
                 11.5, MUTED, anchor="middle")
    body += text(px + 8, sy(tp[cs[-1]]) - 26, "● 输出吞吐", 11, BLUE)
    body += text(px + 108, sy(tp[cs[-1]]) - 26, "┄ 每请求吞吐（带宽被分摊）", 11, MUTED)

    open(out, "w", encoding="utf-8").write(wrap(body, "数据：results/bench_results_v2.csv（TTFT 已修正）· 单次运行"))


# ---------------------------------------------------------------- 图 2
def fig_cudagraph(d, out):
    eager, graph = d["eager_on"]["throughput"], d["baseline"]["throughput"]
    cs = sorted(graph)
    ymax = nice_max(max(graph.values()) * 1.2)
    body, sy, (px, py, pw, ph) = frame(
        "CUDA graph 开关对比", "enforce_eager=True（逐 kernel 发射） vs False（录制回放）",
        ymax, "tok/s")

    n = len(cs)
    gw = pw / n
    bw = gw * 0.26
    for i, c in enumerate(cs):
        cx = px + gw * (i + 0.5)
        for j, (val, color) in enumerate([(eager[c], AMBER), (graph[c], BLUE)]):
            bx = cx - bw * 1.08 + j * bw * 1.16
            y = sy(val)
            body += rect(bx, y, bw, py + ph - y, color)
            body += text(bx + bw / 2, y - 7, f"{val:.0f}", 11, color, anchor="middle", weight="600")
        sp = graph[c] / eager[c]
        body += text(cx, min(sy(eager[c]), sy(graph[c])) - 26, f"{sp:.2f}×", 12.5, GREEN, anchor="middle", weight="700")
        body += text(cx, py + ph + 20, f"C={c}", 12, INK, anchor="middle")

    body += rect(px, py + 6, 11, 11, AMBER, 2) + text(px + 17, py + 16, "eager", 11.5, INK)
    body += rect(px + 72, py + 6, 11, 11, BLUE, 2) + text(px + 89, py + 16, "CUDA graph", 11.5, INK)
    body += text(px + 190, py + 16, "绿色 = 加速比（batch 越大，收益越小 → 带宽接棒成为瓶颈）", 11, GREEN)
    open(out, "w", encoding="utf-8").write(wrap(body, "说明：每个配置独立子进程运行一次，避免 CUDA graph 状态污染"))


# ---------------------------------------------------------------- 图 3
def fig_ablation(d, out):
    base = d["baseline"]["throughput"][64]
    rows = [
        ("baseline", "baseline · block 256 / seqs 512 / mem 0.85", base),
        ("block_512", "block_size 512", d["block_512"]["throughput"][64]),
        ("seqs_64", "max_num_seqs 64", d["seqs_64"]["throughput"][64]),
        ("mem_0.75", "mem_util 0.75 · KV 池仅 30K，装不下 128 blocks", d["mem_0.75"]["throughput"][64]),
        ("seqs_16", "max_num_seqs 16 · 批大小被钉死", d["seqs_16"]["throughput"][64]),
        ("eager_on", "enforce_eager=True · 关 CUDA graph", d["eager_on"]["throughput"][64]),
    ]
    rows_sorted = sorted(rows, key=lambda r: -r[2])
    xmax = nice_max(max(r[2] for r in rows) * 1.05)
    body, sx, (px, py, pw, ph) = hframe(
        "参数消融（并发 64 · 输出吞吐）", "同一硬件同一模型，每次只改标注的那一个旋钮",
        xmax, "输出吞吐 (tok/s)")

    n = len(rows_sorted)
    slot = ph / n
    bh = min(slot * 0.58, 32)
    for i, (tag, desc, val) in enumerate(rows_sorted):
        y = py + i * slot + (slot - bh) / 2
        rel = val / base
        color = BLUE if tag == "baseline" else (GREEN if rel >= 0.98 else (AMBER if rel >= 0.7 else RED))
        bw = sx(val) - px
        body += rect(px, y, bw, bh, color)
        body += text(px + 10, y + bh / 2 + 4.5, desc, 11.5,
                     "#ffffff" if bw > 300 else INK)
        body += text(px + bw + 9, y + bh / 2 + 4.5, f"{val:.0f}", 12, color, weight="600")
        body += text(px + bw + 56, y + bh / 2 + 4.5, f"{rel:.2f}×", 11.5,
                     MUTED if abs(rel - 1) < 0.02 else color, weight="600")
    open(out, "w", encoding="utf-8").write(wrap(body, "红色 = 明显劣化 · 灰色 ×1.00 = 该旋钮无影响（<2%）"))


# ---------------------------------------------------------------- 图 4
def _group(ablation, kind):
    """按开关状态归组：cache_off / cache_off_full / i1_off 都算 off。

    早期实验用 cache_off*，交替实验用 i<N>_off —— 两种命名都要覆盖，
    否则只统计到一半数据，均值会被单批次带偏。
    """
    out = []
    for tag, rs in (ablation or {}).items():
        t = tag.lower()
        if t == kind or t.startswith(f"cache_{kind}") or t.endswith(f"_{kind}"):
            out.extend(rs)
    return out


def fig_prefix_cache(d, out, ablation=None):
    """图 4: prefix caching 开关对照。

    关键口径问题：进程首轮混着 CUDA 首次调用开销，不能和稳态均值直接比。
    所以拆成两组，各自内部同口径：
      代价组 —— 全部用"进程首轮"（关闭 vs 开启但未命中）
      收益组 —— 全部用"稳态"（关闭 vs 开启且命中，排除首次命中的 graph 重捕获）
    """
    if ablation:
        off = _group(ablation, "off")
        on = _group(ablation, "on")
        off_first = [r["ttft"] for r in off if r["idx"] == 0]
        on_first = [r["ttft"] for r in on if r["idx"] == 0]
        off_stable = [r["ttft"] for r in off if r["idx"] >= 1]
        on_stable = [r["ttft"] for r in on if r["idx"] >= 2]
        _wrows = [r for r in on if r["idx"] >= 2]
        hits = sum(r["hit"] for r in _wrows)
        looks = sum(r["look"] for r in _wrows)

        if off_first and on_first and off_stable and on_stable:
            m = lambda xs: sum(xs) / len(xs) * 1000
            v_off1, v_on1 = m(off_first), m(on_first)
            v_off2, v_on2 = m(off_stable), m(on_stable)
            vals = [v_off1, v_on1, v_off2, v_on2]
            ymax = nice_max(max(vals) * 1.26)
            body, sy, (px, py, pw, ph) = frame(
                "Prefix caching：未命中不亏，命中才赚",
                "两组各自同口径对比（左边都是进程首轮，右边都是稳态），2048 token 共享前缀",
                ymax, "TTFT (ms)", lambda v: f"{v:.0f}")

            # 两组柱：组内 2 柱，组间留 26px
            gap = 26
            gw = (pw - gap) / 4
            bw = gw * 0.60
            xs = [px + gw * (i + 0.5) + (gap / 2 if i >= 2 else -gap / 2) for i in range(4)]
            colors = [GREY, "#b6c0cc", GREY, GREEN]
            heads = [
                ("关闭", "进程首轮 · 算满 2048 tok", GREY),
                ("开启 · 未命中", "进程首轮 · 0 命中", MUTED),
                ("关闭", "稳态 · 算满 2048 tok", GREY),
                ("开启 · 命中", f"稳态命中 {hits/looks*100:.0f}% · 只算 16 tok", GREEN),
            ]
            for i, v in enumerate(vals):
                cx = xs[i]
                y = sy(v)
                body += rect(cx - bw / 2, y, bw, py + ph - y, colors[i])
                body += text(cx, y - 9, f"{v:.1f} ms", 12.5, colors[i], anchor="middle", weight="700")
                body += text(cx, py + ph + 20, heads[i][0], 12, INK, anchor="middle", weight="600")
                body += text(cx, py + ph + 35, heads[i][1], 10.2, MUTED, anchor="middle")

            # 组标签
            mid1 = (xs[0] + xs[1]) / 2
            mid2 = (xs[2] + xs[3]) / 2
            body += text(mid1, py + 18, "未命中的代价", 11.5, MUTED, anchor="middle", weight="600")
            body += text(mid2, py + 18, "命中后的收益", 11.5, GREEN, anchor="middle", weight="600")
            body += line(xs[0] - bw / 2, py + 24, xs[1] + bw / 2, py + 24, AXIS, 1.4)
            body += line(xs[2] - bw / 2, py + 24, xs[3] + bw / 2, py + 24, GREEN, 1.4, dash="4 3")

            # 组内差值标注
            d1 = v_on1 - v_off1
            body += text(xs[1] + bw / 2 + 5, sy(v_on1) + 14, f"{d1:+.1f} ms", 11, MUTED, weight="600")
            body += text(xs[3] + bw / 2 + 5, sy(v_on2) + 16,
                         f"↓{(1 - v_on2/v_off2)*100:.0f}%", 13, GREEN, weight="700")
            body += text(xs[3] + bw / 2 + 5, sy(v_on2) + 31,
                         f"{v_off2/v_on2:.2f}×", 11, GREEN, weight="600")

            open(out, "w", encoding="utf-8").write(wrap(
                body,
                "微基准：未命中仅 0.08 ms（hash 10µs + 写 8 块 71µs），占 TTFT 0.1%"))
            return

    # 回退: 只有 baseline 的冷/热两柱
    sh = d["baseline"]["shared"]
    labels = [("冷启动", "需算满 2048 tok"), ("命中 · 第 2 轮", "仅 prefill 16 tok"), ("命中 · 第 3 轮", "仅 prefill 16 tok")]
    vals = [v * 1000 for v in sh[:3]]
    ymax = nice_max(max(vals) * 1.25)
    body, sy, (px, py, pw, ph) = frame(
        "Prefix caching 收益（2048 token 共享前缀）", "命中后实际只 prefill 16 个 token",
        ymax, "TTFT (ms)", lambda v: f"{v:.0f}")

    n = len(vals)
    gw = pw / n
    bw = gw * 0.36
    for i, v in enumerate(vals):
        cx = px + gw * (i + 0.5)
        color = GREY if i == 0 else GREEN
        y = sy(v)
        body += rect(cx - bw / 2, y, bw, py + ph - y, color)
        body += text(cx, y - 8, f"{v:.1f} ms", 12, color, anchor="middle", weight="600")
        body += text(cx, py + ph + 20, labels[i][0], 12, INK, anchor="middle")
        body += text(cx, py + ph + 35, labels[i][1], 10.5, MUTED, anchor="middle")
        if i:
            body += text(cx - bw / 2 - 6, py + ph - (py + ph - y) / 2, f"↓{(1-v/vals[0])*100:.0f}%",
                         12, GREEN, anchor="end", weight="700")
    body += line(px, py + 8, px + pw, py + 8, AXIS, 1, dash="3 4")
    body += text(px + pw - 2, py + 2, "剩余 ~25ms 为固定开销（调度 + pinned H2D + 1 次 decode）",
                 10.5, MUTED, anchor="end")
    open(out, "w", encoding="utf-8").write(wrap(body, "改造 A 加开关后，此图将变成「关闭 / 开冷 / 开热」三柱对照"))


def load_shared_detail(*paths):
    """读 shared 明细 -> {tag: [{ttft, hit, look, cold, idx}]}

    支持传多个 CSV：改造 A 的两批实验（顺序跑 + 交替跑）本来就是同一套测量。
    idx 是"进程内第几轮"，用来区分 进程首轮 / 首次命中过渡 / 稳态。
    """
    d = OrderedDict()
    for path in paths:
        if not path or not os.path.exists(path):
            continue
        with open(path, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                if r.get("mode") != "shared":
                    continue
                lst = d.setdefault(r["tag"], [])
                lst.append({
                    "ttft": float(r["ttft_s"]),
                    "hit": int(r.get("hit_blocks") or 0),
                    "look": int(r.get("lookup_blocks") or 0),
                    "cold": str(r.get("note", "")).startswith("cold"),
                    "idx": len(lst),
                })
    return d or None


def write_index(outdir, d, ablation=None):
    """把 4 张 SVG 拼成一页，方便直接看/截图"""
    tp = d["baseline"]["throughput"]
    sh = d["baseline"]["shared"]

    # prefix cache 的对外数字优先用改造 A 的开关对照（比"冷 vs 热"严格）
    pc_note = f"2048 token 共享前缀：TTFT {sh[0]*1000:.1f}ms → {sh[1]*1000:.1f}ms（↓{(1-sh[1]/sh[0])*100:.0f}%）。"
    pc_kpi = f"{(1-sh[1]/sh[0])*100:.0f}%"
    if ablation:
        _off = _group(ablation, "off")
        _on = _group(ablation, "on")
        _off1 = [r["ttft"] for r in _off if r["idx"] == 0]
        _on1 = [r["ttft"] for r in _on if r["idx"] == 0]
        _off2 = [r["ttft"] for r in _off if r["idx"] >= 1]
        _on2 = [r["ttft"] for r in _on if r["idx"] >= 2]
        _wrows = [r for r in _on if r["idx"] >= 2]
        _hits = sum(r["hit"] for r in _wrows)
        _looks = sum(r["look"] for r in _wrows)
        if _off1 and _on1 and _off2 and _on2:
            a, b = sum(_off2)/len(_off2), sum(_on2)/len(_on2)
            c, e = sum(_off1)/len(_off1), sum(_on1)/len(_on1)
            pc_note = (f"稳态：关闭 {a*1000:.1f}ms → 命中 {b*1000:.1f}ms，降幅 {(1-b/a)*100:.0f}%（{a/b:.2f}×），"
                       f"稳态命中率 {_hits/_looks*100:.0f}%。"
                       f"未命中代价：同为进程首轮 {c*1000:.1f} vs {e*1000:.1f}ms（{(e/c-1)*100:+.1f}%），"
                       f"在 ±4~6ms 噪声内；微基准实测仅 0.08ms。")
            pc_kpi = f"{(1-b/a)*100:.0f}%"
    cards = [
        ("fig1_throughput_vs_concurrency.svg", "图 1 · 吞吐随并发扩展",
         f"并发 1 → 64 吞吐提升 {tp[64]/tp[1]:.1f}×；32 → 64 只剩 {tp[64]/tp[32]:.2f}×，说明已撞到带宽墙。"),
        ("fig2_cuda_graph.svg", "图 2 · CUDA graph 消融",
         "batch 越大加速比越小：graph 抹掉的是每 step 固定的 kernel launch 开销，batch 一大就重新变成带宽受限。"),
        ("fig3_ablation.svg", "图 3 · 参数消融（并发 64）",
         "block_size 与 max_num_seqs=64 无影响；mem 0.75 与 seqs 16 是真正会让吞吐掉下来的两个旋钮。"),
        ("fig4_prefix_cache.svg", "图 4 · prefix caching 开关对照", pc_note),
    ]
    blocks = []
    for i, (fn, title, note) in enumerate(cards, 1):
        p = os.path.join(outdir, fn)
        if not os.path.exists(p):
            continue
        svg = open(p, encoding="utf-8").read()
        blocks.append(
            f'<section class="card"><div class="cap"><span class="idx">{i}</span>'
            f'<h2>{esc(title)}</h2></div>{svg}<p class="note">{esc(note)}</p></section>')

    html_doc = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>nano-vllm 基线性能报告</title>
<style>
  :root {{ color-scheme: light; }}
  body {{ margin:0; padding:40px 20px 60px; background:#f8fafc; color:#1e293b;
         font-family:'Microsoft YaHei','PingFang SC',system-ui,sans-serif; }}
  .wrap {{ max-width:840px; margin:0 auto; }}
  h1 {{ font-size:22px; margin:0 0 6px; letter-spacing:-.2px; }}
  .sub {{ color:#64748b; font-size:13.5px; margin-bottom:26px; }}
  .kpis {{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-bottom:28px; }}
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
</style></head><body><div class="wrap">
<h1>nano-vllm 基线性能报告</h1>
<div class="sub">Qwen3-0.6B · RTX 4070 Laptop 8GB（256 GB/s）· WSL2 Ubuntu 22.04 · torch 2.5.1+cu124 · flash-attn 2.8.3.post2</div>
<div class="kpis">
  <div class="kpi"><div class="v">{tp[64]:,.0f}</div><div class="l">tok/s @ 并发 64</div></div>
  <div class="kpi"><div class="v">{tp[1]:,.0f}</div><div class="l">tok/s 单流</div></div>
  <div class="kpi"><div class="v">187</div><div class="l">GB/s 实测有效带宽<br>≈ 峰值 256 的 73%</div></div>
  <div class="kpi"><div class="v">{pc_kpi}</div><div class="l">prefix cache<br>TTFT 降幅</div></div>
</div>
{''.join(blocks)}
<footer>数据：results/bench_results_v2.csv（6 配置 × 10 项测量，单次运行）<br>
图 4 另含 results/bench_cache_ablation.csv + bench_ab_interleave.csv（12 个独立进程 × 5 轮，开关交替运行）<br>
已知偏差：TTFT@512 被进程内首次调用开销污染，故图中只用 2048 前缀；未记录笔记本功耗墙与实际频率。</footer>
</div></body></html>"""
    p = os.path.join(outdir, "index.html")
    open(p, "w", encoding="utf-8").write(html_doc)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join("results", "bench_results.csv"))
    ap.add_argument("--out", default=os.path.join("results", "figs"))
    ap.add_argument("--ablation", nargs="+",
                    default=[os.path.join("results", "bench_cache_ablation.csv"),
                             os.path.join("results", "bench_ab_interleave.csv")],
                    help="prefix cache 开关对照数据（可传多个 CSV，合并统计）")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    d = load(a.csv)
    fig_throughput(d, os.path.join(a.out, "fig1_throughput_vs_concurrency.svg"))
    fig_cudagraph(d, os.path.join(a.out, "fig2_cuda_graph.svg"))
    fig_ablation(d, os.path.join(a.out, "fig3_ablation.svg"))
    abl = load_shared_detail(*(a.ablation if isinstance(a.ablation, list) else [a.ablation]))
    fig_prefix_cache(d, os.path.join(a.out, "fig4_prefix_cache.svg"), abl)
    idx = write_index(a.out, d, abl)
    print(f"已生成 4 张图 + 一页汇总 -> {a.out}")
    print(f"  打开: {os.path.abspath(idx)}")


if __name__ == "__main__":
    main()
