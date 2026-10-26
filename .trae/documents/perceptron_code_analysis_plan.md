# 感知机源码解析 —— 「值得分析的硬核差异点」方向清单

> 说明：本文是**方向清单**，不是 notebook 实施计划。覆盖范围只限 sklearn 的 `Perceptron` 本身（用户已确认），不展开 PassiveAggressive / hinge-SVM 主线；公式本体（`w ← w + ηyx`）不推导。
> 版本：`vmunet` 环境，Python 3.8.20 / **scikit-learn 1.3.2** / numpy 1.24.4 / scipy 1.10.1。本文行号仅用于定位，日后做 notebook 一律用 `inspect.getsource` 取源码，**不写死行号**。

## Summary

用户提问：**「感知机有什么类似的内容可以做的」**。澄清结果：

- **交付形式**：先出方向清单（同 hw3 朴素贝叶斯、hw4 线性回归第一版）；随后用户确认，已按本清单产出 `homework5-Perceptron.ipynb` 与 `.html`。
- **覆盖范围**：只讲 `Perceptron` 本身。
- **筛选标准**（沿用 hw2 / hw3 / hw4）：优先「课堂上讲不透 / 没讲、且是 sklearn 工程实现的真实决策」。
- **核心叙事**：sklearn 的 `Perceptron` **类没有单独实现训练循环**，而是复用 `BaseSGDClassifier` 的 SGD 框架——它只是**一组被锁死的默认配置** + 损失表里 `Hinge(θ=0)` 这一行。行数要分两把尺子量：**文件 221 行**、**类体 213 行**（L9–221）；类体里 docstring 占 158 行（L10–167），其余 55 行才是实现（L169–221，且不含文件顶部的 8 行 import）。

---

## Current State Analysis

### 工作区现状

- 已有 `homework1-kmeans`、`homework2-KNN`、`homework3-NaiveBayes`、`homework4-LinearRegression` 的 `.ipynb` + `.html`，以及 `fix_pagebreak.py`、`README.md`。
- 已有四份计划：`kmeans_code_match_plan.md`、`knn_code_analysis_plan.md`、`naivebayes_code_analysis_plan.md`、`linearregression_code_analysis_plan.md`。
- 感知机此前没有任何文件；本文是首份材料。README 的作业表本就滞后（hw3 / hw4 未回填），本次不动 README。

### 环境（本轮实测）

| 项 | 值 |
|---|---|
| 环境 | `/root/MSA/ENTER/envs/vmunet` |
| 版本 | Python 3.8.20 / scikit-learn **1.3.2** / numpy 1.24.4 / scipy 1.10.1 |
| 调用 | `wsl -d Ubuntu-22.04 -- /root/MSA/ENTER/envs/vmunet/bin/python ...` |

> 用户规则：一律用项目环境 `vmunet`，不允许 base 环境。

### 源码地图

| 文件 | 行数 | 可分析性 |
|---|---|---|
| `sklearn/linear_model/_perceptron.py` | 221（文件）/ 213（类体） | 纯 Python；类体 L9–221，其中 L10–167 docstring（158 行），L169–221 是实现（55 行，不含顶部 import） |
| `sklearn/linear_model/_stochastic_gradient.py` | 2574 | 纯 Python，外层流程全可 dump |
| `sklearn/linear_model/_sgd_fast.pyx` | 不存在 | 本地包没有 `.pyx` |
| `sklearn/linear_model/_sgd_fast.pxd` | 26 | 只声明 `LossFunction / Regression / Classification / Log / SquaredLoss`，**不暴露 Hinge** |
| `_sgd_fast.cpython-38-x86_64-linux-gnu.so` | — | 只有二进制，更新内循环不可 `inspect.getsource` |
| `sklearn/linear_model/_base.py` | — | `make_dataset`（L136–186）、`SPARSE_INTERCEPT_DECAY`（L52–54）可 dump |

**方法边界**：凡涉及 Cython 内循环（更新式、loss 标量、epoch 停止判定）的结论，一律用**行为实测反推**，不假装引用源码。

### 已实测确认的关键事实

1. **等价性**：`Perceptron(random_state=0)` 与
   `SGDClassifier(loss="perceptron", penalty=None, eta0=1.0, learning_rate="constant", random_state=0)`
   在 60 个样本上 `coef_` / `intercept_` / `n_iter_` **逐元素完全相等**（`np.array_equal` 为 True；`n_iter_=15`，`t_=901`）。注意区分措辞：`np.allclose` 只能支持「**数值近似一致**」，不要说成「完全一致」。
2. **损失映射**：`_stochastic_gradient.py` L499–509 的 `loss_functions` 表中，
   `"perceptron": (Hinge, 0.0)`（L502），而 `"hinge": (Hinge, 1.0)`（L500）。同一个 `Hinge` 类，只差一个阈值参数。
3. **Hinge 阈值行为**（构造 `X=[[1],[5]]`, `y=[1,0]`, `sample_weight=[1,0]`, `coef_init=0`, `intercept_init=m`）：
   - `m=0.5`：Perceptron **不更新**（coef 仍为 0，b=0.5）；hinge **更新**；
   - `m=2.0`：两者都不更新；
   - `m=-0.5`：两者都更新；Perceptron 步长 1.0（`eta0=1` + `learning_rate="constant"`），hinge 默认 `learning_rate="optimal"` 下步长 10.0（对照时必须显式改 constant）。
   → `Hinge(θ=0)` 的判定退化为「零分界」，`Hinge(θ=1)` 是「间隔 1 的线性 SVM」。
4. **教科书循环可复现（严格不可分数据）**：`make_blobs(n_samples=200, centers=2, random_state=7, cluster_std=2.0)` 的两簇只是**有重叠**，重叠不等于不存在线性分界，**不能用来证明不可分**。因此在末尾追加 **2 组「相同特征、相反标签」的矛盾样本对**（`[[0,0],[0,0],[5,5],[5,5]]` / `[-1,1,-1,1]`），矛盾对存在即保证任何线性模型都不可能全分对。此时手工循环
   `if y*(x@w+b) <= 0: w += y*x; b += y`，配合 `shuffle=False, tol=None, max_iter=1000`，
   与 sklearn 的 `coef_` / `intercept_` **数值近似一致（`np.allclose`）**（共 18583 次更新；sklearn `n_iter_=1000`、`t_=204001`、`coef[:2]=[8.3589, -7.1631]`、`b=75.0`）。
   把判据换成 `< 0` 则**首轮 0 次更新就「收敛」**（零初始化下 `z` 全为 0，`<` 无一命中），与 sklearn 不一致——这同时钉死了「折点取等号」和「零点不可微」两件事。
   同一份数据若用默认 `tol=1e-3`，60 样本的可分案例只跑 `n_iter_=7` 就停。
5. **`t_` 语义（限定适用范围）**：docstring（`_perceptron.py` L137–139）称「Number of weight updates performed during training. Same as `(n_iter_ * n_samples + 1)`」。**这个说法不能按字面读，但「处理样本数 + 1」也不是普遍精确的解释**：`t_` 是**按迭代轮数和输入样本数维护的计数属性**，仅在「单次 `fit`、二分类、无 Smart 分片」时才恰好等于 `n_iter_ × n_samples + 1`：
   - 二分类 60 样本 × 15 轮 → `t_=901`；
   - digits 多分类 1797 样本 × 22 轮 → `t_=39535`，但 22 是**各 OvA 二分类 `n_iter_` 的最大值**（`_fit_multiclass` L794–801），不对应任何一个二分类器的真实遍历量；
   - `early_stopping=True` 时 120 样本 × 11 轮 → `t_=1321`：**外层公式仍用完整输入样本数**，被切出去当验证集的样本照样计入；
   - P3 的非可分实验里 `t_=204001` vs 实际更新 18583 次——差距来自「轮数 × 样本数」这个定义本身。
   - `partial_fit` 两次 → `t_` 61 → 121，而 `n_iter_` 始终为 1。
   → 稳妥表述：**`t_` 是按迭代轮数和输入样本数维护的计数属性，不能直接解释为参数实际改变的次数。**
6. **稀疏截距衰减**：`make_dataset`（`_base.py` L178–184）对稀疏输入返回 `intercept_decay=SPARSE_INTERCEPT_DECAY=0.01`，稠密返回 1.0。
   同一份 `make_blobs(100, random_state=3, cluster_std=1.5)` 数据，`shuffle=False, max_iter=30, tol=None`：
   - 稠密：`intercept_=3.0`，`coef_≈[-9.90, -7.04]`；
   - 稀疏（`csr_matrix`）：`intercept_=0.14`，`coef_≈[-9.38, -8.87]`；
   - 两者 `t_` 都是 3001。
7. **停止条件（P6）**：`tol` 的判据**不是与上一轮比**，而是与**历史最优损失** `best_loss` 比。源码 docstring（`_stochastic_gradient.py` L996–1002）原文：*training will stop when `(loss > best_loss - tol)` for `n_iter_no_change` consecutive epochs.* 即**连续 `n_iter_no_change` 轮（默认 5）改善都不足 `tol` 才停**。`tol=None` → 阈值被换成 `-inf`（L450），永不早停，且因 `self.tol is not None` 不成立（L708–712）连 `ConvergenceWarning` 都不报。
   实测「耐心计数」随参数线性推迟停止（`n_iter_no_change=1/5/20/100` → `n_iter_=2/11/45/234`）。
8. **参数校验时机分两类**（已实测修正）：
   - 参数**根本不在签名里** → **构造期 TypeError**：`Perceptron(average=True)`、`Perceptron(loss="hinge")`。这才是「禁止改损失 / 禁止开 ASGD」的**直接原因**；类体里的 `_parameter_constraints.pop("loss")` / `.pop("average")` **只是移除这两个键对应的参数校验约束**（继承自基类，签名里既然没有就不该再校验），它本身不阻止任何调用。
   - 参数**越界** → **构造通过，`fit` 期才报错**（1.3 的参数校验挂在 `fit` 的 `@_fit_context` 装饰器上）：
     `Perceptron(eta0=0.0)` → `ValueError: eta0 must be > 0`（`_more_validate_params` L149–153）；
     `Perceptron(alpha=-1.0)` → `fit` 期 `InvalidParameterError`；
   - `Perceptron(penalty="l1")` → 正常 fit（会正则化，偏离教科书）。
9. **多分类**：digits 上 `coef_.shape=(10, 64)`、`intercept_.shape=(10,)`、`n_iter_=22`；`_fit_multiclass`（L757–810）用 joblib `Parallel(n_jobs=..., require="sharedmem")`，`n_iter_` 取各二分类的**最大值**（L794–801）。
10. **没有 `predict_proba`**：实例只有 `decision_function` / `predict` / `score`；`hasattr(p, "predict_proba")` 为 False；二分类 `decision_function` 返回 1D，多分类返回 2D。

---

## 「值得分析的硬核差异点」

### 前置：一次 `fit` 的调用链（读者应先看这张图）

`Perceptron` **自己没有 `fit`**（`"fit" in Perceptron.__dict__` 为 `False`，`Perceptron.fit.__qualname__ == "BaseSGDClassifier.fit"`）。整条链上每一层都属于基类或编译模块：

```
Perceptron.fit                        # 继承 BaseSGDClassifier.fit（L877，被 @_fit_context 装饰）
  → BaseSGDClassifier._fit            # L655：重置 t_、处理 warm_start、交给 _partial_fit
  → BaseSGDClassifier._partial_fit    # L571：算 class_weight，按 n_classes 分流
  → BaseSGDClassifier._fit_binary     # L723：二分类（也是 OvA 的单个 job）
    / BaseSGDClassifier._fit_multiclass  # L757：多分类，joblib 并行
  → fit_binary                        # L359：准备 dataset、解析随机种子
  → _plain_sgd64 / _plain_sgd32       # Cython `_sgd_fast._plain_sgd64`（.so）
  → Hinge.dloss                       # 就在 Cython 里逐样本调用
```

三个要点：① `Perceptron` 只是「一行配置」（`loss="perceptron"` + 几个默认值）；② 二分类 / 多分类在 `_partial_fit` 处分叉；③ 真正的训练循环在 `.so` 里，只有损失对象能通过 `py_loss` / `py_dloss` 从 Python 侧读到判据。`fit` 上挂的 `@_fit_context` 正是 P1 里「越界参数要到 `fit` 期才报错」的原因。

（行号由 `inspect.getsourcelines` + 文本扫描程序化取得，脚本已复核。）

### 主线（必做，P1–P5）

#### P1｜文件 221 行 / 类体 213 行：`Perceptron` 是「减法 API」的预设档位

- **位置**：`_perceptron.py` 文件共 221 行；类体 L9–221（213 行），其中 L10–167 是 docstring（158 行）、L169–179 约束字典、L181–221 `__init__`；文件顶部 L1–8 是 import（不在类体内）。
- **做法**：`__init__` 签名里**不放** `loss` / `average`，并硬编码
  `loss="perceptron"`、`learning_rate="constant"`、`eta0=1.0`、`power_t=0.5`，其余参数原样转发给基类；类体的 `_parameter_constraints` 先继承 `BaseSGDClassifier` 的约束表，再 `pop("loss")`、`pop("average")`——**只是移除这两个键的校验约束**，不是"禁止"的来源。
- **为什么不一样**：课堂上「感知机」是一种算法；在这里它是**一个估计器的预设档位**。真正的「减法」发生在**签名**上：`loss` 与 `average` 根本不存在，用户传进去就是构造期 `TypeError`；`pop()` 只是让校验表与签名保持一致（继承来的约束键若留着，反而会与 `_parameter_constraints` 的声明式校验自相矛盾）。默认 `penalty=None` 只是「不加正则」，但 `alpha=0.0001` / `l1_ratio=0.15` 仍留在签名里，形成「有参数、默认无效」的错位。
- **实测点**：
  - `Perceptron(average=True)` → `TypeError: __init__() got an unexpected keyword argument 'average'`；
  - `Perceptron(loss="hinge")` → 同样 TypeError；
  - `Perceptron(penalty="l1")` 可正常 fit，但更新式已带正则，不再是教科书感知机；
  - `Perceptron().average` 为 `False`（属性存在，但无法从构造器打开）。

#### P2｜「感知机损失」其实是 `Hinge(θ=0)`：和线性 SVM 只差一个阈值

- **位置**：`_stochastic_gradient.py` L499–509（`loss_functions`），L500 `hinge→(Hinge, 1.0)`、L502 `perceptron→(Hinge, 0.0)`；`_get_loss_function` L165–171。
- **做法**：`_get_loss_function` 走 `loss_class(*args)`，所以 `loss="perceptron"` 实例化出来的就是 `_sgd_fast.Hinge(0.0)`；`hinge` 则是 `Hinge(1.0)`。
- **为什么不一样**：课堂上「感知机 = 误分类驱动」「线性 SVM = 间隔驱动」是两个算法；实现里它们是**同一个 Cython 类 + 一个 double 参数**。更硬核的是：`loss` / `dloss` 是 `cdef` 方法，Python 侧**没有** `.loss` 名字（`hasattr(lf, "loss")` 为 False），编译模块只额外暴露了**专供测试**的 `py_loss` / `py_dloss` 包装。可以直接读：

  | `p`（= wᵀx+b） | `y` | Perceptron `py_loss` | Perceptron `py_dloss` | hinge `py_loss` | hinge `py_dloss` |
  |---|---|---|---|---|---|
  | 1.5 | 1 | 0.0 | 0.0 | 0.0 | 0.0 |
  | **1.0** | **1** | 0.0 | **0.0** | **0.0** | **−1.0** |
  | 0.5 | 1 | 0.0 | **0.0** | 0.5 | **−1.0** |
  | 0.0 | 1 | 0.0 | **−1.0** | 1.0 | −1.0 |
  | −0.5 | 1 | 0.5 | −1.0 | 1.5 | −1.0 |
  | 0.5 | −1 | 0.5 | +1.0 | 1.5 | +1.0 |

  → Perceptron 的 `dloss` 在 `z = p·y <= 0` 时非零（**`z = 0` 也算**），hinge 在 `z <= 1` 时非零（**`z = 1` 也算**，实测 `Hinge(1).py_dloss(1.0, 1.0) == -1.0`；再大一点立刻变 0）。**两个折点都取等号，且折点处 `py_loss = 0` 而 `py_dloss ≠ 0`**——这正是「零初始化时损失为 0 却仍更新」的原因。
- **实测点**（`X=[[1],[5]]`, `y=[1,0]`, `sample_weight=[1,0]`, `coef_init=0`, `intercept_init=m`）：

  | 初始决策值 m | Perceptron（θ=0） | hinge（θ=1） |
  |---|---|---|
  | 0.5 | 不更新（coef=0, b=0.5） | 更新 |
  | 2.0 | 不更新 | 不更新 |
  | −0.5 | 更新（coef=1.0, b=0.5） | 更新（默认 optimal 步长 10.0） |

  → 零分界 vs 间隔 1 的差别被一次实验直接看见。**对照注意**：hinge 必须显式设 `penalty=None, eta0=1.0, learning_rate="constant"`，否则默认 `optimal` 会把「步长」也混进结论。
- **延伸（很值得讲的一段）**：**零初始化时损失已经是 0，为什么还会更新？** `w=0, b=0` ⇒ 每个样本 `z = 0`；查表可知 `Hinge(0)` 在 `z=0` 处 `py_loss = 0` 但 `py_dloss = ∓1`。原因是 `max(0, -z)` 在 `z=0` **不可导**（折点），实现必须挑一个次梯度，而 sklearn 挑的是**非零的那一侧**。后果：零初始化后的第一个 epoch 里**每个样本都会触发一次更新**；这也正是判据写成 `<= 0` 而不是 `< 0` 的原因——零点不可微，实现选了非零次梯度，算法就"必须"在那里动一下。反过来，写成 `< 0` 则首轮 0 次更新、直接"收敛"（P3 已实测）。
  实测推演：`X=[[1],[2],[3],[4]]`、`y=[1,1,-1,-1]`、`fit_intercept=False`、`max_iter=1` ⇒ 一轮后 `coef_=[-2.0]`（第 1、3 个样本更新，第 2、4 个跳过），与手工推演逐个吻合。

#### P3｜教科书循环可逐元素复现：更新式、遍历顺序、停止时机

- **位置**：行为层面对 `_plain_sgd` 黑盒；外层 `fit_binary` L359–491、`_prepare_fit_binary` L327–356。
- **做法**：在 `shuffle=False`、`tol=None`、稠密、`penalty=None`、`early_stopping=False` 时，算法退化为「按行顺序遍历、误分类即更新、跑满 `max_iter`」。
- **为什么不一样**：默认配置下它**不是**教科书循环——每个 epoch 后洗牌、按 loss 下降幅度早停、稀疏输入还有截距衰减（P8）。只有把上述开关全部关掉，才能得到可逐元素对照的版本。
- **实测点**：
  - 数据：`make_blobs(n_samples=200, centers=2, random_state=7, cluster_std=2.0)` **再追加 2 组矛盾样本对**（同特征、相反标签）以保证**严格线性不可分**——光靠 `make_blobs` 的重叠区不能证明不可分；
  - 手工 `if y*(x@w+b) <= 0: w += y*x; b += y` 与 sklearn 的 `coef_` / `intercept_` **数值近似一致（`np.allclose`，不是逐位相等）**；
  - 手工循环共 18583 次更新，而 sklearn `n_iter_=1000`、`t_=204001`（再次说明 `t_` 不是更新次数，见 P4）；
  - **判据的等号**：把手工循环改成 `< 0`，**首轮 0 次更新、1 个 epoch 就退出**（零初始化下 `z` 全为 0，`<` 无一命中），与 sklearn 不一致；写成 `<= 0` 才一致；
  - **单样本边界**：`X=[[0],[0]]`、`y=[1,0]`、`sample_weight=[1,0]`、`coef_init=0`、`intercept_init=0` → Perceptron 的 `intercept_` 从 0 变为 1，确认「决策值恰为 0 也更新」；
  - 60 样本默认 `tol=1e-3` 只跑 `n_iter_=7`。

#### P4｜文档里的「更新次数」应该怎么理解：`t_` 是按「轮数 × 输入样本数」维护的计数

- **位置**：`_perceptron.py` L137–139（属性文档）；`_stochastic_gradient.py` L692（`fit` 重置 `t_=1.0`）、L740 / L800（`t_ += n_iter_ * X.shape[0]`）、L623–624（`partial_fit` 首次置 1.0）。
- **做法**：`fit` 每次把 `t_` 重置为 1.0，训练结束后按「轮数 × **完整输入样本数**」累加；`partial_fit` 不重置，跨调用继续累加。
- **为什么不一样**：docstring 写「Number of weight updates performed during training. Same as `(n_iter_ * n_samples + 1)`」。这句话不能按字面读，**但「处理样本数 + 1」也不是普遍精确的解释**：`t_` 是**按迭代轮数和输入样本数维护的计数属性**，只有在「单次 `fit` + 二分类」时才恰好等于那个公式；多分类时用的是各 OvA `n_iter_` 的**最大值**，早停时用的仍是**完整样本数**（验证集那部分照算）。稳妥表述：**`t_` 不能直接解释为参数实际改变的次数。**
- **实测点**：
  - 901 = 15 × 60 + 1（二分类）；
  - 39535 = 22 × 1797 + 1（digits 多分类，22 是各二分类的最大轮数）；
  - **1321 = 11 × 120 + 1**（`early_stopping=True`，外层仍用完整的 120 个样本）；
  - P3 的非可分实验：`t_=204001` vs 实际更新 **18583** 次；
  - `partial_fit` 两次：`t_` 61 → 121，`n_iter_` 恒为 1（`max_iter` 被固定为 1）；
  - `warm_start=True` 也照样在 `fit` 开头重置 `t_`。

#### P5｜多分类 = OvA + joblib：`n_jobs` 只在多分类生效，`n_iter_` 取最大值

- **位置**：`_stochastic_gradient.py` L627–646（>2 类分派）、L757–810（`_fit_multiclass`）、L773–792（`Parallel(require="sharedmem")`）、L794–801（`n_iter_ = max`）。
- **做法**：每个类各训一个「本类 vs 其余」的二分类器；先在外层用原始多分类标签算好验证掩码，再对每个类并行调用 `fit_binary`；每个 job 的随机种子在并行前统一抽取。
- **为什么不一样**：没有 softmax、没有多类联合更新；每个二分类器**独立早停**，所以各列 `coef_` 对应不同的迭代历史，最后只保留 `n_iter_` 的最大值。`n_jobs` 在二分类路径上完全没用（`_fit_binary` 不读它），容易被误当成全局加速参数。
- **实测点**：digits 上 `coef_.shape=(10, 64)`、`intercept_.shape=(10,)`、`n_iter_=22`、`t_=39535`；`require="sharedmem"` 意味着 joblib 只能用线程类后端（共享内存），不是进程隔离。

### 扩展（选做，P6–P8）

#### P6｜`tol` 与**历史最优** loss 比较，连续 `n_iter_no_change` 轮不改善才停；`tol=None` 静默跑满 `max_iter`

- **位置**：docstring `_perceptron.py` L42–46；`_stochastic_gradient.py` L450（`tol=None` → `-inf`）、L994–1002（`tol` 的 docstring 原文）、L708–720（警告条件）、L466–467（`n_iter_no_change` 传入 C）。
- **做法**：判据**不是「本轮的 loss 与上一轮比」**，而是与**历史最优损失**比。源码 docstring 原文：*training will stop when `(loss > best_loss - tol)` for `n_iter_no_change` consecutive epochs.* 即「连续 `n_iter_no_change` 轮（默认 5）都没把历史最优 loss 压低超过 `tol`，才停」。`tol=None` 时阈值被换成 `-inf`，上式永不成立，等于永不早停。
- **为什么不一样**：课堂上「收敛」通常指「不再误分类 / 权重不再变」；这里是**训练损失相对历史最优的改善幅度**，而且带「耐心计数」。非可分数据上 loss 可能长期缓慢下降，于是出现「还在误分类但已经停止」的情况。
- **实测点**：
  - `tol=None, max_iter=3` 无任何警告（因 `self.tol is not None` 不成立）；`tol=1e-3, max_iter=3` 报 `ConvergenceWarning`（仅在 `n_iter_ == max_iter` 时触发）；
  - 扫耐心计数，`n_iter_no_change = 1 / 5 / 20 / 100` → `n_iter_ = 2 / 11 / 45 / 234`，严格单调递增，直接证明「连续多轮不改善才停」。
- **注意**：loss 的具体归一化（是否除以样本数）在 C 内循环里，本地不可读；正文只写「每个 epoch 的 loss 标量」，不要断言「平均」或具体公式。

#### P7｜early_stopping：换成监控验证集 score，**与 P6 共用同一套耐心计数**

- **位置**：`_stochastic_gradient.py` L257–310（`StratifiedShuffleSplit`）、L312–324（`_ValidationScoreCallback`）、L763–765（多分类先按原标签切）、L1060–1065（docstring：验证 score 连续 `n_iter_no_change` 轮提升不足 `tol`）、L147–148（`partial_fit` 禁用）。
- **做法**：`early_stopping=True` 时用 `validation_fraction` 做分层切分；切分发生在 OvA 之前，每个二分类器收到同一个 bool 掩码；回调每次把当前 `coef_/intercept_` 塞进克隆估计器，算验证集 `score`；判据是「验证 score 连续 `n_iter_no_change` 轮提升不足 `tol`」。
- **为什么不一样**：它和 P6 **不是两套独立机制，而是同一套耐心计数、只是换了监控指标**——P6 看训练 loss（越小越好），P7 看验证集 score（越大越好）。同一份数据上两者的停止轮数可以不同（实测 `n_iter_no_change=20` 分别是 45 与 26）。`partial_fit` 直接禁止早停。
- **实测点**：`early_stopping=True` → `n_iter_=11`、`t_=1321`（= 11 × 120 + 1，外层用完整样本数）；`partial_fit + early_stopping` → `ValueError: early_stopping should be False with partial_fit`。

#### P8｜稀疏输入的截距每步只走 1%：`SPARSE_INTERCEPT_DECAY=0.01`

- **位置**：`_base.py` L52–54（常量与注释）、L136–186（`make_dataset`，分支在 L178–184）；`fit_binary` L432–434（返回值传入 C）。
- **做法**：稠密输入 `intercept_decay=1.0`；稀疏输入 `intercept_decay=0.01`，每次截距更新都乘这个系数。源码注释给的理由是「avoid intercept oscillation」。
- **为什么不一样**：同一份数据、同一组超参，`csr_matrix` 与稠密数组会得到**不同的模型**，且不是数值误差。感知机默认不加正则，截距被压慢后，系数需要「补偿」它，于是 `coef_` 也跟着变。
- **实测点**（100 样本，`shuffle=False, max_iter=30, tol=None`）：

  | 输入 | `intercept_` | `coef_` 前两维 | `t_` |
  |---|---|---|---|
  | 稠密 ndarray | 3.0 | ≈[-9.90, -7.04] | 3001 |
  | `csr_matrix` | 0.14 | ≈[-9.38, -8.87] | 3001 |

  > 与 hw4 的 R13 同源（都是 `make_dataset` 的这个分支）。hw4 已讲过机制，这里只补**感知机上的后果**：无正则时截距与系数同时漂移，读者不应把它当成「稀疏只是存储格式」。

### 选读（P9–P10）

#### P9｜`class_weight` 不是重采样，而是把权重乘进 C 循环

- **位置**：`_stochastic_gradient.py` L603–606（`compute_class_weight`）、L432–434 与 L474–475（`pos_weight` / `neg_weight` 入 C）；`fit` docstring L894–898（`sample_weight` 会与 `class_weight` 相乘）。
- **做法**：`class_weight="balanced"` 只算出每类一个标量权重；二分类把正、负类权重分别作为 `pos_weight` / `neg_weight` 传入内循环，多分类则每个 OvA job 用「本类权重 + 1.0」。数据不复制、不改行数。权重是**乘性**的：实测把所有样本的 `sample_weight` 乘 2，`coef_` 也精确翻倍（比值 `[2.0, 2.0]`）。
- **为什么不一样**（**理由已修正**）：教科书的「加权」常被理解成过采样 / 重采样；这里是**损失项的乘性权重**。它与「复制少数类样本」**不等价**，但理由**不是**「权重比例凑不齐」——比例其实凑得齐：`balanced` 权重比 5.25 = 21/4，把**多数类复制 4 份、少数类复制 21 份**（252×4 = 48×21 = 1008），两类「总权重质量」与比例就完全对齐了。真正的区别在**在线训练**：
  - `balanced` 放大的是**单次更新的幅度**（0.5952 / 3.125 的乘性系数），一次更新仍算一步；
  - 复制改变的是**访问次数与遍历顺序**（每个 epoch 样本数不同、洗牌后排列不同），一次更新只算一步。
  感知机每步都在改 `w`，后续样本的判定依赖当前 `w`，所以「放大一次更新」与「重复访问样本」不会走出同一条轨迹。
- **实测点**（300 样本、85:15 不平衡，`max_iter=200, tol=None`）：
  - 类别样本数 `[252, 48]`，`balanced` 权重 `[0.5952, 3.125]`（比例 1 : 5.25）；
  - `class_weight="balanced"`：`coef=[8.829, -5.29]`、训练 score **0.720**；
  - 复制「多数×4 / 少数×21」（两类各 1008 个样本，比例 1 : 5.25）：`coef=[1.506, -1.128]`、训练 score **0.813**；
  - 两者 `coef_` 不同（`np.allclose` 为 False），且**比值 `[0.171, 0.213]` 不是常数**——连等比缩放都不成立，说明轨迹形状本身就不同。
  - `class_weight="balanced" + partial_fit` → `ValueError: class_weight 'balanced' is not supported for partial_fit...`。

#### P10｜没有 `predict_proba`：只有 `decision_function` 的符号 / argmax

- **位置**：`LinearClassifierMixin`（`_base.py`）；`Perceptron` 未覆盖该方法。
- **做法**：二分类 `decision_function` 返回 1D 分数，`predict` 取符号；多分类返回 2D，`predict` 取 `argmax`。
- **为什么不一样**：感知机输出的是**未校准的分数**，不是概率。同为 SGD 家族，只有 `log_loss` / `modified_huber` 能提供 `predict_proba`；`Perceptron` 与 `hinge` 都不能。这是 API 层最容易被误用的一点（下游若直接拿 `decision_function` 当置信度，会得到不可解释的数值）。
- **实测点**：`hasattr(p, "predict_proba")` 为 False；二分类 `decision_function(X).shape == (n_samples,)`；digits 上为 `(n_samples, 10)`。

---

## 与 hw4（线性回归 / SGD）的重叠与去重

| hw4 已讲 | 本文对应 | 处理方式 |
|---|---|---|
| R8/R14：`_sgd_fast` 是编译模块、32/64 dtype 分派、Python 只做配置台 | P2/P3 的方法边界 | 不复述结构，只引用「内循环不可读」这一前提 |
| R9：`learning_rate='optimal'` 与 `alpha` 的耦合 | P2 的对照实验 | 只提一句「对照 hinge 必须改 constant」，不展开学习率表 |
| R10：`fit` 重置 `t_`、`partial_fit` 累积 | P4 | 只取「docstring 说更新次数、实测是样本数」这个新结论；重置/累积一句带过 |
| R11：`average=True`（ASGD） | P1 | 只说 `Perceptron` 把 `average` 从 API 删掉，不展开 ASGD |
| R12：验证掩码在训练/评估两侧的行为 | P7 | 保留「多分类先切分、所有 OvA 共享掩码」这个感知机特有结论 |
| R13：`SPARSE_INTERCEPT_DECAY=0.01` | P8 | 机制不重复，只补感知机上的实测后果 |
| R20–R23：Ridge / Lasso | — | 无关，不引用 |

**原则**：同一结论只讲一次；hw4 讲机制，本文讲「在感知机上会看到什么」。

---

## 取舍

- **主线（必做）**：**P1–P5**。这五条构成完整叙事——「预设档位 → 损失只是 Hinge(0) → 可复现的教科书循环 → `t_` 的真实语义 → 多分类 OvA」。
- **扩展（选做）**：**P6–P8**。停止条件、早停、稀疏截距衰减；其中 P8 与 hw4 有重叠，按需取用。
- **选读**：**P9–P10**。`class_weight` 与 API 边界，适合收尾。
- **不做**：
  - PassiveAggressive（PA-I/PA-II）与 `pa1`/`pa2` 学习率——同基类但属于另一估计器，用户已确认不扩；
  - 感知机收敛性证明、线性可分性定理；
  - Cython 内循环的源码引用（本地无 `.pyx`）；
  - 重复 hw4 已写的 SGD 配置台 / Dataset 抽象 / ASGD / 学习率表；
  - 改 README、提交 git。

---

## 已交付 notebook 的结构（28 cell / 14 code）

| # | 节标题 | 覆盖 |
|---|---|---|
| 1 | 感知机（教科书版 vs 文件/类体行数） | — |
| 2 | 实现全貌：一次 `fit` 的调用链 | 前置图 |
| 3 | 定位源码：文件 221 行，类体 213 行 | — |
| 4 | P1｜减法 API：签名里没有 `loss` 与 `average` | P1 |
| 5 | P2｜「感知机损失」就是 `Hinge(θ=0)`（含 `z=1` 行） | P2 |
| 5+ | 延伸：零初始化时损失为 0 为什么还更新 | P2/P3 |
| 6 | P3｜教科书循环可逐元素复现（矛盾对保证不可分） | P3 |
| 7 | P4｜文档里的「更新次数」应该怎么理解 | P4 |
| 8 | P5｜多分类 = OvA + joblib | P5 |
| 9 | P6 / P7｜同一套耐心机制，监控指标不同 | P6/P7 |
| 10 | P8｜稀疏输入的截距每步只走 1% | P8 |
| 11 | P9 / P10｜权重不等价于重采样；感知机没有概率 | P9/P10 |
| 12 | 小结 | — |

代码要求（沿用 hw1–hw4 惯例）：
- 用 `inspect.getsource(...)` 取源码，不写死行号；print 前用注释标出 markdown 引用的关键行段；
- 图表标题 / 图例用纯 ASCII（避免 CJK glyph missing）；
- 不引入新依赖，全部在 `vmunet` 下可执行；
- 若产出，工作流：`nbconvert --to notebook --execute --inplace` → `--to html --execute` → `fix_pagebreak.py`。

---

## Assumptions & Decisions

1. **交付形式**：先出方向清单；用户确认后已产出 notebook + HTML（28 cell / 14 code）。
2. **覆盖范围**（用户已选）：只讲 `Perceptron` 本身；PA、hinge-SVM 只作为 P2 的对照，不单列主线。
3. **版本**：vmunet 为 sklearn **1.3.2**，与 hw1–hw4 一致；本文行号与结论均在该版本下验证。
4. **源码不可见部分**：`_sgd_fast` 本地只有 `.so`；`.pxd` 仅 26 行且不含 `Hinge` 声明。P2/P3/P4/P6 的内循环结论**以行为实测为准**，文中不得出现「源码第 N 行写了更新式」这类表述。
5. **与 hw4 的关系**：P8 与 hw4 R13 同源、P4 与 hw4 R10 部分重叠；已在上表给出「去重说明」。
6. **代码改动范围**：本清单 markdown + `homework5-Perceptron.ipynb/.html`；README 未改。

---

## Verification

1. 本文每条 P# 都含：源码位置、与课堂/教科书的差异、可执行的实测方法；无「待补充」占位（P9 的对照实验已完成，见下方数字）。
2. 文中数字与下列实测一致（可重跑复核）：

```bash
# 等价性 + t_ + 多分类
wsl -d Ubuntu-22.04 -- /root/MSA/ENTER/envs/vmunet/bin/python - <<'PY'
import numpy as np
from sklearn.linear_model import Perceptron, SGDClassifier
from sklearn.datasets import make_blobs, load_digits
X, y = make_blobs(n_samples=60, centers=2, random_state=0)
p = Perceptron(random_state=0).fit(X, y)
s = SGDClassifier(loss="perceptron", penalty=None, eta0=1.0,
                  learning_rate="constant", random_state=0).fit(X, y)
assert np.array_equal(p.coef_, s.coef_) and np.array_equal(p.intercept_, s.intercept_)
assert p.t_ == p.n_iter_ * X.shape[0] + 1 == 901
Xd, yd = load_digits(return_X_y=True)
m = Perceptron(random_state=0, max_iter=50).fit(Xd, yd)
assert m.coef_.shape == (10, 64) and m.t_ == m.n_iter_ * Xd.shape[0] + 1 == 39535
print("ok")
PY

# 教科书循环复现（shuffle=False, tol=None）
wsl -d Ubuntu-22.04 -- /root/MSA/ENTER/envs/vmunet/bin/python - <<'PY'
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=200, centers=2, random_state=7, cluster_std=2.0)
y = np.where(y == 0, -1, 1)
w = np.zeros(X.shape[1]); b = 0.0
for _ in range(1000):
    for i in range(len(y)):
        if y[i] * (X[i] @ w + b) <= 0:
            w += y[i] * X[i]; b += y[i]
sk = Perceptron(tol=None, shuffle=False, random_state=0, max_iter=1000).fit(X, y)
assert np.allclose(w, sk.coef_[0]) and np.allclose(b, sk.intercept_[0])
print("ok")
PY

# 稀疏截距衰减 + 警告行为 + 参数校验时机
wsl -d Ubuntu-22.04 -- /root/MSA/ENTER/envs/vmunet/bin/python - <<'PY'
import numpy as np, scipy.sparse as sp, warnings
from sklearn.linear_model import Perceptron
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=100, centers=2, random_state=3, cluster_std=1.5)
kw = dict(shuffle=False, random_state=0, max_iter=30, tol=None)
d = Perceptron(**kw).fit(X, y); s = Perceptron(**kw).fit(sp.csr_matrix(X), y)
assert d.intercept_[0] == 3.0 and abs(s.intercept_[0] - 0.14) < 1e-9
with warnings.catch_warnings(record=True) as w1:
    warnings.simplefilter("always")
    Perceptron(max_iter=3, tol=None, random_state=0).fit(X, y)
    assert not w1
with warnings.catch_warnings(record=True) as w2:
    warnings.simplefilter("always")
    Perceptron(max_iter=3, tol=1e-3, random_state=0).fit(X, y)
    assert any(issubclass(x.category, UserWarning) for x in w2)
for kwargs, exc in [({"average": True}, TypeError), ({"loss": "hinge"}, TypeError), ({"alpha": -1.0}, Exception)]:
    try:
        Perceptron(**kwargs); raise AssertionError("should raise")
    except exc:
        pass
print("ok")
PY
```

3. 文中不出现「本地可读 `_sgd_fast.pyx`」之类的错误表述；`_sgd_fast.pxd` 的 26 行、不含 `Hinge` 的事实不能写反。
4. 「待确认的 notebook 结构」明确标注不执行；`README.md` 未改动；无新增 `ipynb` / `html`。

---

## Risks

- **版本 / 行号漂移**：行号只作定位提示；若 `vmunet` 升级，需重核 `_perceptron.py` 的 docstring 长度与 `_stochastic_gradient.py` 的 `loss_functions` 行号。
- **Cython 黑盒**：P2 的阈值语义、P6 的 loss 判定都靠实验反推；文中已标明「行为证据，不是源码引用」。
- **P3 的可复现条件很窄**：`shuffle=False` + `tol=None` + 稠密 + 无 penalty + 无 early_stopping；缺一不可，正文已列全。
- **P8 与 hw4 重复风险**：只保留感知机上的后果，机制一句带过并指向 hw4。
- **P9 的「复制样本 vs balanced」**：已实测，结论是「**不等价**」而非「近似等价」。理由**不是**权重比例凑不齐（5.25 = 21/4，多数类×4、少数类×21 即可对齐），而是**在线训练中「放大单次更新幅度」≠「重复访问样本」**；两组的 `coef` 比值 `[0.171, 0.213]` 不是常数，可直接作为证据。
