# Machine Learning 作业仓库

课程作业合集：以 **scikit-learn 源码为素材**，对经典机器学习算法的实现做逐层解析（markdown 提炼 + 可复现的实测代码）。

---

## 环境

| 项 | 值 |
|---|---|
| 环境路径 | `/root/MSA/ENTER/envs/vmunet` |
| 解释器 | `/root/MSA/ENTER/envs/vmunet/bin/python` |
| 版本 | Python 3.8 + scikit-learn 1.3.2 |

> **规则**：所有脚本一律用项目环境（`vmunet`）的解释器运行，**不允许直接使用 base 环境**。
> `jupyter` / `nbconvert` 也在该环境的 `bin` 下。

---

## 作业清单

| # | 主题 | 源码对象 | Notebook | HTML |
|---|---|---|---|---|
| 1 | KMeans 源码解析 | `sklearn.cluster.KMeans` | [homework1-kmeans.ipynb](homework1-kmeans.ipynb) | [homework1-kmeans.html](homework1-kmeans.html) |
| 2 | KNN 源码解析 | `sklearn.neighbors` | [homework2-KNN.ipynb](homework2-KNN.ipynb) | [homework2-KNN.html](homework2-KNN.html) |

### 作业一：KMeans

逐层解析 KMeans 的 Python 实现：参数与属性 → `fit` 主流程 → k-means++ 初始化 → Lloyd / Elkan 迭代核心 → 收敛与择优细节 → `predict` / `transform` / `score` → 顶层 `k_means` 函数。

### 作业二：KNN

重点放在 sklearn 实现里**「不太一样」的地方**（课堂上讲过的多数投票 / 距离加权只作呼应）：

- **数学层**：全程平方距离 + 延迟开方、`radius *= radius`、`effective_metric_` 的坍缩特化、`0<p<1` 是半度量导致树算法被禁、`weights='distance'` 的零距离特判、半径结果的稳定排序；
- **数据结构层**：Top-k 用 `np.argpartition` 而非全排序、`algorithm="auto"` 的硬编码启发式、稀疏距离矩阵当图输入（KNN 可以没有坐标）、半径版返回变长 object 数组；
- **架构层**：`fit` 只建索引且索引可跨模型复用、`NeighborsBase` 抽象基类 + Mixin 组合、`kneighbors` 的四条执行路径、离群点的「全零概率」判定。

---

## 通用工作流

每份作业遵循同一套流程：

1. **规划** —— 在 `.trae/documents/` 下写 `<主题>_plan.md`（可选分析方向、notebook 章节结构、验证步骤）
2. **编写 notebook** —— markdown 做提炼，code cell 贴源码片段 + 实测验证
3. **全量执行** —— 把输出固化进 notebook，确认 **0 error**
4. **导出 HTML** —— 由 notebook 生成静态页面
5. **注入分页 CSS** —— 用 `fix_pagebreak.py` 修复打印分页

对应命令（在项目根目录执行）：

```bash
PY=/root/MSA/ENTER/envs/vmunet/bin/python
JUP=/root/MSA/ENTER/envs/vmunet/bin/jupyter

# 1) 全量执行，把输出写回 notebook
$JUP nbconvert --to notebook --execute --inplace homework2-KNN.ipynb

# 2) 导出 HTML（--execute 表示导出前再跑一遍，确保输出最新）
$JUP nbconvert --to html --execute homework2-KNN.ipynb

# 3) 注入打印分页 CSS
$PY fix_pagebreak.py homework2-KNN.html
```

> 仅改动了 markdown（不影响输出）时，第 2 步可去掉 `--execute`，直接复用 notebook 里已固化的输出，更快也不改变结果。

---

## fix_pagebreak.py（所有作业共用）

**问题**：nbconvert 模板自带 `.jp-OutputArea-child { break-inside: avoid-page }`，导致代码 / 输出在页底放不下时被**整段推到下一页**，留下大片空白。

**做法**：在 HTML 的 `</head>` 前注入 `<style id="pagebreak-override">`，把 `.jp-Cell` / `.jp-CodeCell` / `.jp-OutputArea-child` / `pre` 等元素的 `break-inside` 与 `page-break-inside` 覆盖为 `auto !important`，让长代码在页底**自然截断**（从中间跨页），不主动推页。

```bash
python fix_pagebreak.py homework2-KNN.html   # 单个文件
python fix_pagebreak.py *.html               # 批量（通配符）
```

脚本**幂等**：会先移除旧的 override 再重新插入，重复运行安全。

注入的样式内容如下，也可直接手写进任意 HTML 的 `</head>` 之前，不依赖脚本：

```html
<style id="pagebreak-override">
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
```

---

## 提交约定

- Git 提交信息使用 Conventional Commits 前缀，如 `feat: hw2`、`docs: plan for code show`、`fix: ...`
- notebook 与它导出的 HTML **一起提交**，保证两份文件始终对应
- 临时调试脚本不提交（用完即删）