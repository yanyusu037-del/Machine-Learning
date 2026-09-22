#!/usr/bin/env python3
"""
html-pagebreak-fix: 修复 Jupyter nbconvert 生成的 HTML 中「长代码被强制推到下一页」的分页问题。

用法:
    python fix_pagebreak.py <html文件>
    python fix_pagebreak.py *.html
    python fix_pagebreak.py homework1-kmeans.html

原理:
    nbconvert 模板自带 `.jp-OutputArea-child { break-inside: avoid-page }`，
    导致代码输出区域在页底放不下时被整段推到下一页、留下大片空白。
    本脚本注入 `break-inside: auto !important` 覆盖模板规则，
    让代码/输出在页底自然截断（从中间跨页），不主动推页。

幂等: 多次运行安全（先移除旧 override 再插入）。
"""

import re
import sys
import os

OVERRIDE = """<style id="pagebreak-override">
@media print {
  /* 允许代码/输出在页底自然截断，不主动推到下一页 */
  .jp-Cell, .jp-CodeCell, .jp-MarkdownCell,
  .jp-InputArea, .jp-CodeMirrorEditor, .jp-Editor,
  .jp-OutputArea, .jp-OutputArea-child, .jp-OutputArea-prompt,
  .jp-CellBody, .jp-CellInput, .jp-CellOutput,
  .input, .inner_cell, .output, .output_wrapper, .output_area,
  pre, .CodeMirror, .jp-CodeMirrorEditor pre {
    break-inside: auto !important;
    page-break-inside: auto !important;
  }
}
</style>
"""


def fix(path):
    with open(path, encoding="utf-8") as f:
        s = f.read()

    # 移除旧 override（幂等）
    s = re.sub(r'<style id="pagebreak-override">.*?</style>\s*', '', s, flags=re.DOTALL)

    # 在 </head> 前插入
    s = s.replace('</head>', OVERRIDE + '</head>', 1)

    with open(path, "w", encoding="utf-8") as f:
        f.write(s)

    size_before = os.path.getsize(path)
    ok_css = 'break-inside: auto !important' in s
    ok_id = 'pagebreak-override' in s
    print(f"  {path}: size={size_before}B, override={'OK' if ok_id else 'MISSING'}, auto={'OK' if ok_css else 'FAIL'}")
    return ok_id and ok_css


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    paths = sys.argv[1:]
    failed = []
    for p in paths:
        if not os.path.exists(p):
            print(f"  SKIP {p}: not found")
            continue
        if not fix(p):
            failed.append(p)

    if failed:
        sys.exit(f"failed: {failed}")
    print(f"\ndone: {len(paths)} file(s) patched")


if __name__ == "__main__":
    main()
