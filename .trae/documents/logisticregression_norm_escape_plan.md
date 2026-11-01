# hw6 逻辑回归：`||w||` 表格转义修复计划

## Summary

用户问：第 2 节表格行

```
| `liblinear` | C++ 内部 `0.5||w||^2 + C * sum loss` | 整体除以 C |
```

改成 `` `0.5\|\|w\|\|^2` `` 之后「为什么还不行」。

实测（vmunet 环境，nbconvert 7.16.6 / **mistune 3.3.4**）结论：

| 写法 | 表格是否建成 | code span 里实际显示 |
|---|---|---|
| `` `0.5||w||^2` ``（原样） | **否**，整块降级成一段 `<p>` | — |
| `` `0.5\|\|w\|\|^2` ``（用户尝试） | 是 | **`0.5\|\|w\|\|^2`，反斜杠可见** |
| `0.5\|\|w\|\|^2`（不加反引号） | 是 | 无 code span，显示 `0.5||w||^2` |
| `` `0.5‖w‖²` ``（Unicode） | 是 | `0.5‖w‖²`，干净 |

原因：`\|` 能骗过拆格，但骗不过 code span。所以本计划改用 **Unicode `‖`（U+2016）+ `²`**，与 hw4 已有写法一致，并同步第 2 节公式块与第 10 节正文，最后重新导出 HTML。

## Current State Analysis

### 为什么两种写法都不对（源码级）

- `mistune/plugins/table.py::_split_table_cells` 只把 `\|` 当成"不是分隔符"，切片 `text[start:pos]` **原样保留反斜杠**，没有 GFM 那种"拆格后把 `\|` 还原成 `|`"的步骤。
- `_process_row` 里 `if len(cells) != len(aligns): return None`；一旦某行拆出的列数和表头不同，`parse_table` 直接走 `_parse_invalid_pipe_table`，**整张表按普通段落输出**。
- `\|` 落在 code span 内时，CommonMark 规定 code span 不做反斜杠转义，于是反斜杠被原样渲染出来。
- 参考：GitHub 的 GFM 实现会删掉 `\|`（包括 code span 内），所以同一段在 GitHub 上看起来是对的，在 Jupyter 里不对——这是渲染器差异，不是写法本身"少转义了一次"。

### 现状证据

- 全篇扫描（按未转义 `|` 计数）：**只有 cell 6（第 2 节）这一张表坏**，管道数 `[4,4,4,4,4,8]`；其余 6 张表全部 `OK`。
- 已生成的 [homework6-LogisticRegression.html](file:///%5C%5Cwsl.localhost/Ubuntu-22.04/root/Machine%20Learning/homework6-LogisticRegression.html#L7973-L7978) 第 7973–7978 行确实是 `<p>| 后端 | 实际目标 | ...`，全篇 `<table>` 只有 7 个。
- 项目先例：[homework4-LinearRegression.ipynb](file:///%5C%5Cwsl.localhost/Ubuntu-22.04/root/Machine%20Learning/homework4-LinearRegression.ipynb#L29) 第 29 行用 `α‖w‖²`、`α‖w‖₁`，对应 HTML 第 7554 行原样渲染，无需转义。
- hw6 全篇目前 0 处 LaTeX / MathJax，本次不引入。

### 待改的 3 处（`.ipynb` JSON 行号）

| 行 | 位置 | 现状 |
|---|---|---|
| 324 | 第 2 节 ` ```text ` 公式块 | `sum_i sample_weight_i * loss_i(w) + \|\|w\|\|^2 / (2C)` |
| 334 | 第 2 节表格最后一行 | 8 个裸竖线，表格塌掉 |
| 1372 | 第 10 节正文 inner code | `liblinear 是 \`C * sum_i loss_i + 0.5\|\|w\|\|^2\`` |

## Proposed Changes

### 1. `homework6-LogisticRegression.ipynb`（只改 markdown，不动 outputs）

用 Edit 做三处精确替换（每处 old_string 在文件中唯一）：

**a. 第 324 行**（fenced 公式块）

```
old:  "sum_i sample_weight_i * loss_i(w) + ||w||^2 / (2C)\n",
new:  "sum_i sample_weight_i * loss_i(w) + ‖w‖² / (2C)\n",
```

**b. 第 334 行**（坏掉的表格行）

```
old:  "| `liblinear` | C++ 内部 `0.5||w||^2 + C * sum loss` | 整体除以 C |\n",
new:  "| `liblinear` | C++ 内部 `0.5‖w‖² + C * sum loss` | 整体除以 C |\n",
```

**c. 第 1372 行**（第 10 节正文）

```
old:  liblinear 是 `C * sum_i loss_i + 0.5||w||^2`。
new:  liblinear 是 `C * sum_i loss_i + 0.5‖w‖²`。
```

改三处而不是只改表格行的理由：第 2 节和第 10 节在讲同一个目标函数，同一篇里留两种记号会让读者以为是两套式子；Unicode 在三处（fenced block、table cell、inner code）都安全。

**不改**：被引用的 sklearn 源码字符串（`_linear_loss.py` 等原样）、hw4 / hw5 / README、上游方向清单 `logisticregression_code_analysis_plan.md`。

### 2. 重新导出 HTML

```
wsl -d Ubuntu-22.04 -- bash -lc "cd '/root/Machine Learning' && /root/MSA/ENTER/envs/vmunet/bin/jupyter nbconvert --to html homework6-LogisticRegression.ipynb"
wsl -d Ubuntu-22.04 -- bash -lc "cd '/root/Machine Learning' && /root/MSA/ENTER/envs/vmunet/bin/python fix_pagebreak.py homework6-LogisticRegression.html"
```

**不加 `--execute`**：本次仅 markdown 文本变化，`fix_pagebreak.py` 已验证幂等（先删旧 override 再插新的）。

### 3. 明确不做

- 不重跑 notebook（不执行 `--to notebook --execute --inplace`），不重置任何 `execution_count` / outputs。
- 不引入 MathJax `$\lVert w\rVert$`，不与 hw4 记号分叉。
- 不批量替换全仓库的 `||·||`，只处理本 notebook 这 3 处。

## Assumptions & Decisions

- **决定 1：记号用 Unicode `‖w‖²`**。理由：唯一在 table cell 的 code span 里也零转义、零渲染差异的写法，且有 hw4 先例。
- **决定 2：范围取全篇 3 处**，保持第 2 / 10 节同一记号。
- **决定 3：不重跑代码**。markdown 改动不影响输出。
- **备选（若必须保留 ASCII 双竖线）**：把公式移出 code span，直接写 `0.5\|\|w\|\|^2`（不加反引号），实测渲染为 `0.5||w||^2`；代价是丢失等宽字体，且与同段其他 inner code 风格不一致。本计划不采用。

## Verification

1. 改完 `.ipynb` 后重跑表格扫描脚本：cell 6 的管道计数应从 `[4,4,4,4,4,8]` 变为全 `4`，且全篇无 `BROKEN`。
2. 重新导出后检查 HTML：
   - `<table>` 出现次数 **7 → 8**；
   - 第 2 节表格里存在 `<td>C++ 内部 <code>0.5‖w‖² + C * sum loss</code></td>`；
   - 全篇搜不到 `\|\|` 这类可见反斜杠；
   - `pagebreak-override` 计数仍为 1。
3. 输出未受影响：HTML 中仍能搜到 `== 目标函数的三种尺度（_logistic.py）==`、`111.928737` 等既有运行输出，`execution_count` 数值不变。
