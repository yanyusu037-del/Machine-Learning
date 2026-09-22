import re, os

html_path = "/root/Machine Learning/homework1-kmeans.html"
with open(html_path, encoding="utf-8") as f:
    s = f.read()

# 移除旧 override（如果有）
s = re.sub(r'<style id="pagebreak-override">.*?</style>\s*', '', s, flags=re.DOTALL)

override = """<style id="pagebreak-override">
@media print {
  /* 所有代码/输出相关元素：允许在页底自然截断，不主动推到下一页 */
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

# 找 </head> 插入
s = s.replace('</head>', override + '</head>', 1)

with open(html_path, 'w', encoding='utf-8') as f:
    f.write(s)

# 验证
checks = {
    'override present': 'pagebreak-override' in s,
    'break-inside auto !important': 'break-inside: auto !important' in s,
    'template avoid-page still there (neutralized)': 'avoid-page' in s,
}
for k, v in checks.items():
    print(f'  {"OK" if v else "FAIL"} {k}')
print(f'size: {os.path.getsize(html_path)} bytes')
