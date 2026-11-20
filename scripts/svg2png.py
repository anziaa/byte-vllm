"""把 results/figs/*.svg 转成 2x PNG，供 Markdown 正文内嵌（SVG 作为源文件保留）。

为什么需要这一步：
    SVG 在 GitHub / VS Code / 浏览器里都能显示，但不少 Markdown 阅读器
    （以及把 md 直接贴进 IM、导出 PDF 的场景）只认 PNG，SVG 会退化成链接。
    所以正文引用 PNG、仓库保留 SVG，两者由本脚本同步。

用法：
    python scripts/svg2png.py            # 全量转换
    python scripts/svg2png.py fig1 fig5  # 只转文件名含这些子串的

实现：不依赖 cairosvg / inkscape —— 本机装它们太慢。
      用 Edge 无头浏览器的 --screenshot，device-scale-factor=2 拿 2 倍图。
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "results" / "figs"
OUT = FIGS / "png"
SCALE = 2

EDGE_CANDIDATES = [
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
]

HTML = """<!DOCTYPE html><html><head><meta charset="utf-8">
<style>html,body{{margin:0;padding:0;background:#fff;overflow:hidden}}
img{{display:block;width:{w}px;height:{h}px}}</style></head>
<body><img src="{src}"></body></html>"""


def find_browser() -> Path:
    for p in EDGE_CANDIDATES:
        if p.exists():
            return p
    env = shutil.which("msedge") or shutil.which("chrome")
    if env:
        return Path(env)
    raise SystemExit("找不到 Edge / Chrome，无法转换")


def svg_size(svg: Path) -> tuple[int, int]:
    head = svg.read_text(encoding="utf-8", errors="replace")[:600]
    m_w = re.search(r'\bwidth="(\d+(?:\.\d+)?)"', head)
    m_h = re.search(r'\bheight="(\d+(?:\.\d+)?)"', head)
    if not (m_w and m_h):
        vb = re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', head)
        if not vb:
            raise SystemExit(f"{svg.name}: 读不到尺寸")
        return int(float(vb.group(1))), int(float(vb.group(2)))
    return round(float(m_w.group(1))), round(float(m_h.group(1)))


def convert(browser: Path, svg: Path) -> Path:
    w, h = svg_size(svg)
    OUT.mkdir(parents=True, exist_ok=True)
    dst = OUT / (svg.stem + ".png")
    # 页面必须和 svg 同目录、用相对路径 —— Chrome/Edge 默认禁止 file:// 页面跨目录取资源
    page = FIGS / "_svg2png_tmp.html"
    page.write_text(HTML.format(w=w, h=h, src=svg.name), encoding="utf-8")
    try:
        cmd = [
            str(browser),
            "--headless=new",
            "--disable-gpu",
            "--hide-scrollbars",
            "--no-first-run",
            "--force-device-scale-factor=%d" % SCALE,
            "--window-size=%d,%d" % (w, h),
            "--screenshot=%s" % dst,
            page.as_uri(),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
    finally:
        page.unlink(missing_ok=True)
    if not dst.exists():
        raise SystemExit(f"{svg.name}: 截图失败")
    return dst


def main() -> None:
    browser = find_browser()
    pats = sys.argv[1:]
    svgs = sorted(FIGS.glob("*.svg"))
    if pats:
        svgs = [s for s in svgs if any(p in s.name for p in pats)]
    if not svgs:
        raise SystemExit("没有匹配的 svg")
    for s in svgs:
        png = convert(browser, s)
        print(f"  {s.name:42s} -> png/{png.name}  ({png.stat().st_size // 1024} KB)")
    print(f"共 {len(svgs)} 张，输出目录：{OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
