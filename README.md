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
| 3 | 朴素贝叶斯源码解析 | `sklearn.naive_bayes` | [homework3-NaiveBayes.ipynb](homework3-NaiveBayes.ipynb) | [homework3-NaiveBayes.html](homework3-NaiveBayes.html) |
| 4 | 线性回归源码解析 | `sklearn.linear_model` | [homework4-LinearRegression.ipynb](homework4-LinearRegression.ipynb) | [homework4-LinearRegression.html](homework4-LinearRegression.html) |
| 5 | 感知机源码解析 | `sklearn.linear_model.Perceptron` | [homework5-Perceptron.ipynb](homework5-Perceptron.ipynb) | [homework5-Perceptron.html](homework5-Perceptron.html) |

### 作业一：KMeans

逐层解析 KMeans 的 Python 实现：参数与属性 → `fit` 主流程 → k-means++ 初始化 → Lloyd / Elkan 迭代核心 → 收敛与择优细节 → `predict` / `transform` / `score` → 顶层 `k_means` 函数。

### 作业二：KNN

重点放在 sklearn 实现里**「不太一样」的地方**（课堂上讲过的多数投票 / 距离加权只作呼应）：

- **数学层**：全程平方距离 + 延迟开方、`radius *= radius`、`effective_metric_` 的坍缩特化、`0<p<1` 是半度量导致树算法被禁、`weights='distance'` 的零距离特判、半径结果的稳定排序；
- **数据结构层**：Top-k 用 `np.argpartition` 而非全排序、`algorithm="auto"` 的硬编码启发式、稀疏距离矩阵当图输入（KNN 可以没有坐标）、半径版返回变长 object 数组；
- **架构层**：`fit` 只建索引且索引可跨模型复用、`NeighborsBase` 抽象基类 + Mixin 组合、`kneighbors` 的四条执行路径、离群点的「全零概率」判定。

### 作业三：朴素贝叶斯

`naive_bayes.py` 是纯 Python（1530 行，无 `.so`），五个变体共享同一骨架，因此能真正 dump 到每一行。只挑课堂上没讲透的实现决策：

- **模板方法**：`_BaseNB` 只留一个抽象钩子 `_joint_log_likelihood`，就换来了 `predict` / `predict_proba` / `predict_log_proba` / `predict_joint_log_proba` 四个公开方法（`predict` 连归一化都省了，`logsumexp` 负责数值稳定）；
- **在线统计**：GaussianNB 用 Chan-Golub-LeVeque 公式合并均值/方差，支持 `partial_fit`；`epsilon_` 的「先减后加」避免平滑量被卷进合并公式反复累加；
- **代数化简**：BernoulliNB 用恒等式消掉 `(1-X)` 稠密矩阵，换来 CSR 稀疏输入支持；ComplementNB 靠一个负号把 argmin 折进基类写死的 argmax；
- **权重的落点**：`sample_weight` 被折叠进 one-hot 标签矩阵，五个变体的 `_count` 里一行权重判断都没有；
- **异构结构**：CategoricalNB 的 `feature_log_prob_` 是 list-of-arrays，列数随 `partial_fit` 见到的新类别用 `np.pad` 动态增长。

### 作业四：线性回归

围绕 `linear_model` 里 OLS 与 SGD 两条线展开，重点在「实现里不一样的地方」：

- **OLS 不用正规方程**：`LinearRegression` 按 X 的类型与形状分发到三类后端（`scipy.linalg.lstsq`、`scipy.sparse.linalg.lsqr`、`linalg.solve`），多目标时还有 `_decision_function` 之外的策略选择；
- **截距的处理**：稠密输入走「去均值 → 求解 → 补回」，稀疏输入用隐式中心化，并对稀疏截距施加衰减；
- **样本权重**：用 `√w` 重缩放把 WLS 折叠成 OLS，避免改动求解后端；
- **SGD 的配置台**：`learning_rate` / `penalty` 字符串先解析成枚举与调度器，最终落到 C 内循环；
- **优化状态**：`t_` / ASGD / 验证集掩码 / 截距衰减四项状态各自的语义与边界。

### 作业五：感知机

`Perceptron` 是 `SGDClassifier` 的一组预设档位——它的训练循环不在自己身上。逐层拆开「减法 API」背后的东西：

- **调用关系**：`fit → _fit → _partial_fit → _fit_binary / _fit_multiclass → fit_binary → _plain_sgd32/64 → Hinge.dloss`，行号由程序化扫描得到；
- **参数被缩减的方式**：`loss` / `learning_rate` 固定、`eta0` 默认 1.0（可改），`penalty` / `average` 从签名里移除；`loss_functions` 表里 `perceptron → Hinge(0.0)` 与 `hinge → Hinge(1.0)` 只差一个阈值；
- **对偶形式**：代码里没有对偶路径，但 `eta0=1 且无正则` 时 `w = Σ c_i y_i x_i` 可事后重建（实测偏差 `6e-14`），开正则或改步长后恒等式失效；
- **循环可复现**：在严格不可分数据上，手工教科书循环与 sklearn 逐元素近似一致（`updates=18583` vs `t_=204001`），判据实测为 `y·(wᵀx+b) <= 0`（等于 0 也更新）；
- **状态与开关**：`t_` 是按迭代轮数与输入样本数维护的计数属性、`n_jobs` 只在多分类生效、早停与 `tol` 共用同一套耐心计数但监控指标不同、稀疏路径截距单步增量乘以 `0.01`、`class_weight="balanced"` 与重采样不等价。

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