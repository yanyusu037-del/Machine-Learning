# 线性回归源码解析 —— 「值得分析的硬核差异点」方向清单（v2）

> v2 修订说明：v1 的部分技术结论不准确，经用户对照 sklearn 1.3.2 源码逐条审阅后修正（R1/R7/R10/R12/R17/R24/R26、验证方案、三处措辞、篇幅分层）。编号 R1–R26 保持不变，便于对照。

## Summary

用户提问：**「线性回归有什么源码值得分析」**。澄清结果：

- **交付形式**：先只出「方向清单」，暂不产出 notebook / HTML（同朴素贝叶斯那次）。
- **覆盖范围**：**不限于 OLS**；**SGD 梯度下降（`SGDRegressor`）也要纳入**。
- **筛选标准**（沿用作业 2 / 3）：优先「课堂上讲不透 / 没讲、且是 sklearn 工程实现的真实决策」，不展开最小二乘/正则化/SGD 的教科书公式推导。

### 优先级分层（本版核心调整）

| 层级 | 内容 | 编号 | 说明 |
|---|---|---|---|
| **主线（必做）** | OLS | R1–R7 | 与教科书正面冲突，最贴「线性回归」字面 |
| **主线（必做）** | SGD 梯度下降 | R8–R14 | 用户点名；「Python 配置台 + C 内循环」结构 |
| **扩展（选做）** | Ridge / Lasso | R20–R23 | 多 solver 路由 + 坐标下降黑盒 |
| **进阶选读** | GLM / Newton | R15–R19 | 理论最深，按需选读 |
| **延伸（挑 1 个）** | 稳健 / 贝叶斯 / 其他 | R24–R26 | 收尾用 |

产出对象（本次核实到的全部源文件，均在 vmunet 环境）：

| 线 | 文件 | 行数 | 可分析性 |
|---|---|---|---|
| A. OLS | `sklearn/linear_model/_base.py` | 920 | 纯 Python，可 dump |
| B. SGD | `sklearn/linear_model/_stochastic_gradient.py` | ~2570 | 外层 Python 可 dump；内循环在 `_sgd_fast` 编译模块 |
| C. 共享损失 / GLM | `_linear_loss.py`、`_glm/glm.py`、`_glm/_newton_solver.py` | 525 / 901 / 525 | 纯 Python，可 dump |
| D. 正则化族 | `_ridge.py`、`_coordinate_descent.py` | 2560 / ~3100 | 外层 Python 可 dump；坐标下降内循环在 `_cd_fast` 编译模块 |
| E. 延伸（选做） | `_huber.py`、`_bayes.py`、`_theil_sen.py`、`_quantile.py`、`_ransac.py` | — | 纯 Python，可 dump |

---

## Current State Analysis

### 工作区现状（`\\wsl.localhost\Ubuntu-22.04\root\Machine Learning`）
- 已有 `homework1-kmeans.ipynb/.html`、`homework2-KNN.ipynb/.html`、`homework3-NaiveBayes.ipynb/.html`、`fix_pagebreak.py`、`README.md`。
- 已有三份计划：`.trae/documents/kmeans_code_match_plan.md`、`knn_code_analysis_plan.md`、`naivebayes_code_analysis_plan.md`。
- **线性回归尚无任何文件**，本文件 = 方向清单。
- README 的作业表目前只列到作业 2（作业 3 未回填），本次不改 README。

### 环境（本轮已实测确认，**回应「版本不一定一致」的提醒**）
- 项目环境 `/root/MSA/ENTER/envs/vmunet`，**Python 3.8 / scikit-learn 1.3.2 / numpy 1.24.4 / scipy 1.10.1**。
- **结论：版本与作业 1–3 完全一致，未发生漂移**，本文档引用的行号在 1.3.2 下有效。
- 关于 `normalize` 的版本说明（v1 此处不准确，已修正）：
  - 1.3.2 的 **`LinearRegression` 构造函数已经不含 `normalize`**（签名只有 `fit_intercept / copy_X / n_jobs / positive`，`_base.py` L637–648），所以**不能**统一写成「到 1.4 才移除」。
  - 仍在的 `normalize` 是**其它估计器**的公开参数（Lasso / Ridge / ElasticNet / LARS 等）以及**内部辅助函数** `_deprecate_normalize`（L57–133）与 `_preprocess_data(normalize=...)`（L193）。
  - 因此：本文档中与 `normalize` 有关的行号（R2/R4 附近）在 **这些其它估计器升级到 ≥1.4 时**才会位移，`LinearRegression` 自身不受影响。
- 建议（沿用作业 2 约定）：notebook 一律用 `inspect.getsource(obj)` 取源码，**不写死行号**。
- 用户规则：一律用 vmunet 环境，不得用 base。调用方式：
  `wsl -d Ubuntu-22.04 -- /root/MSA/ENTER/envs/vmunet/bin/python <script.py>`

### 可分析性边界（关键）
- **纯 Python 可 dump**：`_base.py`、`_glm/*`、`_linear_loss.py`、`_ridge.py`、`_huber.py`、`_bayes.py`、`_least_angle.py`、`_coordinate_descent.py`、`_stochastic_gradient.py` 的**外层流程**。
- **本地安装包中只有 `.so`（无法用 `inspect.getsource` 查看内循环）**：`_cd_fast`（坐标下降内循环）、`_sag_fast`（SAG/SAGA）、`_sgd_fast`（SGD 内循环 `_plain_sgd32/64` + 各 `LossFunction`/`Penalty`/`LearningRate` 类）。
  → 这三者一律走「接口 / 分派点 / docstring / 调用点」黑盒分析。**如需深入其实现，可另行阅读对应版本的上游 Cython 源码（`.pyx`）**，但那不属于本项目可 dump 的范围。
- 与作业 2（KNN 大量 `.so`）类似、与作业 3（朴素贝叶斯纯 Python）形成对照：**本次是「Python 分派台 + C 内循环」的混合结构**，本身就是可讲的一个工程主题。

---

## 「值得分析的硬核差异点」

### 一、主线 A —— OLS（`_base.py`）：与教科书正面冲突

> docstring 自己点破（L608–610）：*"From the implementation point of view, this is just plain Ordinary Least Squares (**scipy.linalg.lstsq**) or Non Negative Least Squares (**scipy.optimize.nnls**) wrapped as a predictor object."*

| 编号 | 方向 | 源码位置 | 为什么不一样 |
|---|---|---|---|
| **R1** | **OLS 不用正规方程：三类求解后端 × 多目标处理策略** | `LinearRegression.fit` L709–752 | 教科书写 `w=(XᵀX)⁻¹Xᵀy`；sklearn **从不这么算**。**三类后端**：① `positive=True` → `scipy.optimize.nnls`（L709–717）；② `X` 稀疏 → `scipy.sparse.linalg.lsqr`（L718–749）；③ 稠密 → `scipy.linalg.lstsq`（**SVD**，L750–752）。**多目标不是一个独立后端，而是「策略」**：稠密 `lstsq` **原生支持 2D y**（L751，返回 `(n_features, n_targets)` 再 `.T`）；只有 `nnls`（L714–717）与稀疏 `lsqr`（L745–749）因为只吃 1D `y`，才**逐列** `joblib.Parallel` 拼回 |
| **R2** | **截距不是「加一列 1」，而是「去均值 + 事后重构」** | `_preprocess_data` L248–286；`_set_intercept` L388–396 | X、y 各自减**加权均值** → 拟合**无截距**模型 → 再由 `intercept_ = y_offset − X_offset·coef_` 反推（L394）。好处是不必显式扩列、SVD 作用在原始列空间上。（**修正**：本路径未启用 `normalize`，`X_scale` 恒为全 1，故 L393 的 `np.divide(coef_, X_scale)` 实际是**恒等操作**；它是为 `normalize=True` 的老路径与其它模型保留的统一写法。`fit_intercept=False` 时直接 `intercept_=0.0`，L396） |
| **R3** | **稀疏数据的「隐式中心化」：用 `LinearOperator` 不物化中心化矩阵** | `fit` L718–739 | 稀疏 X 若真去中心化会**破坏稀疏性**（变稠密）。解法：定义 `matvec(b)=X·b − b·(X_offset/X_scale)`、`rmatvec(b)=Xᵀ·b − (X_offset/X_scale)·b.sum()`，把「已中心化的 X」表示成一个**只提供矩阵向量乘的算子**，喂给 `lsqr`。数学等价、内存不爆 |
| **R4** | **样本权重用 √w 重缩放，把 WLS 折叠成 OLS** | `_rescale_data` L303–356 | 恒等式：`(y−Xw)ᵀS(y−Xw) = ‖√S y − √S X w‖²`（docstring L308–312 直接写出）。于是加权最小二乘 = 对 `√S·X, √S·y` 做普通最小二乘。教科书只给 WLS 的正规方程，不讲这层变换。附带 `inplace` 优化（L338–341）：仅当 `_preprocess_data` 已复制时才原地改 |
| **R5** | **`rank_` / `singular_` 只在「稠密 且 `positive=False` 且走 `lstsq` 分支」时赋值** | 属性文档 L574–578；赋值点 L751 | `linalg.lstsq` 顺带返回 `rank_`、`singular_`。稀有路径（稀疏 LSQR、NNLS）拿不到奇异值，`positive=True` 甚至在 `_validate_data` 阶段就禁掉稀疏（L676），故这两个属性**同时**要求「稠密 + `positive=False` + lstsq」。文档只说「Only available when `X` is dense」，是这句的精确化 |
| **R6** | **`positive=True` 是另一个算法（NNLS），多目标要逐列并行** | L709–717 | `scipy.optimize.nnls`（active-set）**只接受 1D y**，多目标只能 `Parallel(n_jobs=n_jobs_)(delayed(nnls)(X, y[:, j]))` 逐列（L714–717）。同一参数在不同形状下走完全不同的求解器 |
| **R7** | **`n_jobs` 的触发条件：多目标 **且**（稀疏 **或** `positive=True`）** | docstring L552–558 + 实现 L714–717 / L745–749 | **修正**：`n_jobs` 唯一被使用处就是 R6 的两处 `Parallel`。当「单目标 或 稠密且 `positive=False`」时不走 joblib —— 但这**只是说该路径不使用 joblib 并行，不代表底层只有一个线程**（BLAS/LAPACK 仍可能多线程）。文档用一段话解释了这个「条件性参数」 |

### 二、主线 B —— SGD 梯度下降（`_stochastic_gradient.py`）：Python 配置台 + C 内循环

| 编号 | 方向 | 源码位置 | 为什么不一样 |
|---|---|---|---|
| **R8** | **真正的梯度更新在编译模块，Python 侧只是「配置台」** | 导入 L33–44；`_get_plain_sgd_function` L494–495；调用 L1684–1714 | `_plain_sgd32/64` 来自 `_sgd_fast`（本地包内为 `.so`，用 `inspect.getsource` 看不到内循环）。Python 只负责：把 `loss` 字符串映射成 Cython 的 `LossFunction` 对象（L165–171）、把 `penalty`/`learning_rate` 字符串映射成**整数枚举**（`PENALTY_TYPES` L55、`LEARNING_RATE_TYPES` L46–53），然后一次性把 25 个参数传进 C 层 |
| **R9** | **学习率是「整数枚举 + 时间表」，`'optimal'` 还会反向约束 `alpha`** | `_more_validate_params` L145–159；默认值 L1397–1399 | `learning_rate='optimal'` 时要求 `alpha>0`（L154–159），因为最优步长 `η=1/(α(t+t₀))` 直接**用 alpha 算**——正则强度兼职学习率尺度。且**回归器默认与基类不同**：`BaseSGDRegressor` 默认 `invscaling / eta0=0.01 / power_t=0.25`（L1397–1399），而 `BaseSGD` 默认 `optimal / power_t=0.5`（L111–113） |
| **R10** | **`t_` 是「累计样本数」；且 `fit` 每次重置、`partial_fit` 才延续** | `_fit` L1540–1541；`partial_fit`→`_partial_fit` L1506；`_fit_regressor` L1653–1654 / L1716 | `self.t_ += self.n_iter_ * X.shape[0]`（L1716）是**累计样本数**而非轮数，供 invscaling/optimal 时间表使用。**修正**：`_fit` 在每次 `fit` 调用开头**无条件** `self.t_ = 1.0`（L1540–1541，位于 `warm_start` 分支**之后**，所以 `warm_start=True` 也照样重置）；而 `partial_fit` 直接走 `_partial_fit`（L1506）**不重置**，`_fit_regressor` 仅在无 `t_` 时才补 1.0（L1653–1654）。**「跨调用累积」只对 `partial_fit` 成立** |
| **R11** | **`average=True` = 平均 SGD（ASGD）** | 分配 L242–255；启用判定 L1718–1728 | 除 `coef_` 外另存 `_standard_coef` / `_average_coef`，对**迭代过程中的权重做历史平均**；且只有 `average <= t_ − 1` 时才真正用平均值，否则回退到普通解（L1722–1728）。课堂一般只讲 SGD 本体，不讲 averaging |
| **R12** | **训练循环用掩码跳过验证样本，但验证回调仍提取验证子集** | `_make_validation_split` L257–310；`_make_validation_score_cb` L312–324 | **修正**：验证集用同一数组上的 bool mask 表示（L275、L309），训练循环据此跳过；但回调**确实做了布尔索引** `X[validation_mask] / y[validation_mask] / sample_weight[validation_mask]`（L318–322）来构造评估对象。因此**不能**说「X 不复制」；准确说法是「训练侧不复制、验证侧按掩码提取子集」 |
| **R13** | **稀疏 vs 稠密的截距衰减不同：`intercept_decay` = 1.0 vs 0.01** | `_base.make_dataset` L178–184；常量 L52–54 | 稀疏数据截距更新乘 `SPARSE_INTERCEPT_DECAY=0.01`（源码注释：*avoid intercept oscillation*）。纯工程 corner case，教科书不会提 |
| **R14** | **`make_dataset` 的 Dataset 抽象：32/64 位按 dtype 分派** | `_base.make_dataset` L136–186 | `X.dtype==float32` → `ArrayDataset32/CSRDataset32`，否则 `...64`。C 内循环靠它**随机访问单样本**（`SequentialDataset`）。这是「Python 配置 → C 数据视图」的桥梁，也是 R8 黑盒能工作的前提 |

### 三、扩展 —— Ridge / Lasso（`_ridge.py` + `_coordinate_descent.py`）

| 编号 | 方向 | 源码位置 | 为什么不一样 |
|---|---|---|---|
| **R20** | **Ridge 的 `solver='auto'` 是一张硬编码路由表** | `_ridge_regression` L562–597 | 规则写死：`positive→lbfgs`、`return_intercept→sag`、**稠密→cholesky**、**稀疏→sparse_cg**（L562–571）；并配约束「positive 只能 lbfgs」（L579–584）、「lbfgs 只能 positive」（L586–590）、「只有 sag 能直接拟合截距」（L592–597） |
| **R21** | **`_solve_sparse_cg` 按「特征多还是样本多」切换核岭 / 线性岭** | L83–118 | `n_features > n_samples` 时对 `XXᵀ+αI` 用共轭梯度（**核形式**，`w=Xᵀ·解`，L103–110）；否则对 `XᵀX+αI` 用 CG（L111–118）。全程靠 `LinearOperator` + 迭代，**不显式求逆**。另：`_solve_lsqr` 里 `damp=√α`（L174–180）——LSQR 的阻尼是 alpha 的平方根 |
| **R22** | **Cholesky 失败自动降级到 SVD** | L681–696 | `try: _solve_cholesky(...) except linalg.LinAlgError: solver='svd'`——把「矩阵奇异/病态」当**运行期可恢复分支**处理（核形式 L688–690、普通形式 L694–696） |
| **R23** | **Lasso/ElasticNet 的主体是坐标下降，四个内循环全在 `_cd_fast`（编译模块）** | `enet_path` L584–635 | 按数据形态分派：稀疏 → `sparse_enet_coordinate_descent`（L589–604）、多任务 → `..._multi_task`（L605–608）、预计算 Gram → `..._gram`（L609–626）、否则 `..._coordinate_descent`（L627–630）。**Python 侧只做**：`alpha` 网格（`_alpha_grid` L552）、warm-start 路径、`dual_gap` 尺度校正（L640）。典型黑盒：能讲「何时进哪条支路」，内循环需读上游 Cython 源码 |

### 四、进阶选读 —— GLM / Newton（`_linear_loss.py` + `_glm/`）

| 编号 | 方向 | 源码位置 | 为什么不一样 |
|---|---|---|---|
| **R15** | **截距被塞进 `coef` 向量尾部：`n_dof = n_features + fit_intercept`** | `weight_intercept` L99–136；`loss_gradient` L271–302 | 截距不是一个独立标量，而是 `coef[-1]`（多分类再 reshape）。这样 `grad` 能一次算完（L288–298）。把「拟合截距」降解为「多一个自由度」 |
| **R16** | **GLM = 同一套优化器解不同的 link / 分布** | `glm.py` fit L260–308 | `PoissonRegressor`(log link) / `GammaRegressor` / `TweedieRegressor` 只在 `_get_loss()`（L451–458）换 `base_loss`；`solver` 走 `'lbfgs'`（→ `scipy.optimize.minimize(L-BFGS-B)` L262–283）或 `'newton-cholesky'` / `NewtonSolver` 子类（L284–306） |
| **R17** | **Newton 解法：backtracking line search + 双收敛判据** | `_glm/_newton_solver.py` L199–343 | **修正**：`line_search` 最多尝试 21 次步长（`beta=0.5, sigma=1/2¹¹`，L212）；**21 次尝试均未满足接受条件时**，才 `use_fallback_lbfgs_solve=True` 转 L-BFGS（L286–297）。`check_convergence` 同时检查**两条**：① `max|gradient| <= tol`（L315–321）；② Newton decrement `½·d² = ½·gᵀH⁻¹g <= tol`（L323–331，引 Boyd & Vandenberghe §9.5.1）。**两条都过才判定收敛** |
| **R18** | **GLM 把 `sample_weight` 归一到「和为 1」** | L231–241；初值 L251–255 | 目标函数对 `w` 的常数缩放不变，故令 `Σw=1`（注释 L231–240 写出推导），使 `obj = Σ w·loss + ½α‖w‖²`。初值取 `coef[-1] = link(ȳ)`（L253）——斜率置 0、截距取「均值经 link」的 warm start |
| **R19** | **GLM 的 `score` 是 D²（deviance 解释率），不是 R²** | L364–437 | 自定义实现：`D² = 1 − (dev + c)/(dev_null + c)`（L437），其中 `dev_null` 用**零模型 deviance**（`raw_prediction ≡ link(ȳ)`，L430–436）。`RegressorMixin` 默认给的是 R²，GLM 显式覆盖成 deviance 版本 |

### 五、延伸（选做 1 个）—— 稳健 / 贝叶斯 / 其他

| 编号 | 方向 | 源码位置 | 为什么值得讲 |
|---|---|---|---|
| **R24** | **HuberRegressor：联合优化「系数 + 截距 + 尺度 σ」，用带下界约束的 L-BFGS-B** | `_huber.py` L277–352（`optimize.minimize` L325–333） | **修正**：不是 IRLS。待估向量把 **σ 作为最后一位**（L312–317，初值置 1），并给 σ 加**非负下界** `bounds[-1][0] = eps*10`（L322–323），交给 `scipy.optimize.minimize(method="L-BFGS-B", jac=True)` 一次解完（L325–333）；`_huber_loss_and_gradient`（L18–125）超 `ε·σ` 的点走线性损失。可视化直观 |
| **R25** | **BayesianRidge：证据最大化（evidence maximization）** | `_bayes.py` L423 `_update_coef_`、L446 `_log_marginal_likelihood` | α、λ 由**边际似然**自动更新，**不需要交叉验证**；`predict(return_std=True)` 给的是**预测标准差**（L393），据此画的是**预测区间**（prediction interval），不是参数/均值的置信带。与 R20（Ridge 靠 CV 调 α）形成「两种确定 α 的哲学」对照 |
| **R26** | **TheilSen：子样本最小二乘解 → 参数向量的空间中位数；及另两种「不最小二乘」哲学** | `_theil_sen.py` docstring L212–218、`_spatial_median` L81–134；`_quantile.py`、`_ransac.py` | **修正**：不是「成对斜率取中位数」。多元实现是「**对子样本分别求最小二乘解，再对所有解（参数向量）求空间中位数（L1 中位数）**」，`_spatial_median` 用**修正 Weiszfeld 迭代**求（L81–134）；模块 docstring L34–35 自述其属于「iteratively re-weighted least squares」家族。Quantile=分位数回归（线性规划）；RANSAC=随机采样一致。可作「线性回归的边界」收尾 |

### 取舍

- **主线（必做）**：**R1–R14**（OLS `_base.py` + SGD `_stochastic_gradient.py`）。这两条是本科程「线性回归」的核心，也是最贴课堂差异的地方。
- **扩展（选做一组）**：**R20–R23**（Ridge/Lasso 的多 solver 路由 + 坐标下降黑盒）。
- **进阶选读**：**R15–R19**（GLM/Newton）。理论最深、篇幅最重，按需选读，不默认展开。
- **延伸（挑 1 个）**：**R24–R26** 中挑 1 个（默认 **R24 Huber**，实现最短、效果最直观）。
- **不做**：教科书公式推导（最小二乘、岭/Lasso 的正则化动机、SGD 的收敛证明）；`.so` 内循环的复现（改读上游 Cython 源码即可，但不属本项目 dump 范围）；`_least_angle.py`(LARS)/`_omp.py` 全量铺开（稀释深度，如需只引一句）。

---

## 待确认：若后续要做成 notebook（本节为备选，未获确认前不执行）

> 用户当前选择「先不要产出文件」。以下仅为后续备选结构，按上述优先级排序。

| # | 优先级 | 节标题 | 覆盖 | code cell 做什么 |
|---|---|---|---|---|
| 1 | — | 线性回归的两张「面孔」：课上 vs sklearn | — | 点明只挑「实现里不一样的地方」 |
| 2 | — | 定位源码：linear_model 的可分析边界 | — | 打印 `sklearn.__version__`；列出 `linear_model/` 并分类 `.py` / 编译模块 |
| 3 | 主线 | OLS 不用正规方程：三类后端 × 多目标策略 | R1/R6/R7 | dump `LinearRegression.fit`（切函数体）；同数据分别走 dense / sparse；`positive` 另设「约束活跃 / 不活跃」两组 |
| 4 | 主线 | 截距的去均值重构 + 隐式中心化算子 | R2/R3 | dump `_preprocess_data`/`_set_intercept`；稀疏 X 拟合，验证与稠密结果一致（条件良好数据 + 合理容差） |
| 5 | 主线 | 样本权重 = √w 重缩放 | R4 | dump `_rescale_data`；实测「整数重复样本」≡ `sample_weight` |
| 6 | 主线 | SGD 的配置台：字符串→枚举→C 内循环 | R8/R9/R14 | 打印 `LEARNING_RATE_TYPES`/`PENALTY_TYPES`；`SGDRegressor.__init__` 默认 vs `BaseSGD` 默认对比；实测 `learning_rate='optimal'` + `alpha=0` 报错 |
| 7 | 主线 | SGD 的优化状态：`t_` / ASGD / 验证掩码 | R10/R11/R12/R13 | dump `_fit`/`_fit_regressor` 关键行；对比 `fit` 重置 vs `partial_fit` 延续 `t_`；`average=True` 前后 `coef_` 差异 |
| 8 | 扩展 | 正则化族：多 solver 路由 + 坐标下降黑盒 | R20/R21/R22 | 打印不同输入下 `solver='auto'` 实际落到的 solver；`Ridge` 各 solver 的 `coef_` 一致性；`Lasso` 走 `_cd_fast` 的调用点 |
| 9 | 进阶 | 共享损失抽象：截距塞进 coef | R15 | dump `weight_intercept`/`loss_gradient`；手工核对 `grad[:n_features] = Xᵀ·g + λ·w` |
| 10 | 进阶 | GLM：一套优化器解不同分布 | R16/R18/R19 | dump `glm.fit` 求解分支；`PoissonRegressor` fit 后打印 `coef_`/`intercept_`；`score` 与手算 D² 对照 |
| 11 | 进阶 | Newton 求解器：线搜索 + 双收敛判据 | R17 | dump `_newton_solver.solve`/`line_search`/`check_convergence`；实测 `verbose=2` 打印迭代过程 |
| 12 | 延伸 | Huber / BayesianRidge（选 1） | R24/R25 | 含离群点数据上 OLS vs Huber 对比图；`BayesianRidge.predict(return_std=True)` 画**预测区间** |
| 13 | — | 小结 | — | 串成一条线：**「Python 分派台 + C 内循环」**——分派条件（dtype/形状/稀疏/positive/多目标）决定算法，数值稳定性由 scipy 后端兜底 |

代码要求（沿用作业 1–3 惯例）：
- 一律用 `inspect.getsource(...)` 取源码，**不写死行号**；print 前用注释标出 markdown 引用的关键行段。
- `LinearRegression.fit`（~105 行含 docstring）、`_base.py` 的长函数按逻辑分段打印。
- 图表标题/图例用纯 ASCII（避免 CJK glyph missing）。
- 不引入新依赖，全部在 vmunet 下可执行。
- 若产出，工作流：`nbconvert --to notebook --execute --inplace` → `--to html --execute` → `fix_pagebreak.py`。

---

## Assumptions & Decisions

1. **筛选标准**：优先「课堂没讲透 / 没讲」的实现决策；最小二乘、正则化、SGD 的公式本体不展开。
2. **交付形式**（用户已定）：**只出方向清单**；本文档即交付物，暂不产出 notebook / HTML。
3. **优先级**（本版核心）：**OLS + SGD 为必做主线（R1–R14）**；Ridge/Lasso 扩展（R20–R23）；GLM/Newton 进阶选读（R15–R19）；R24–R26 挑 1。
4. **版本**（本轮实测）：vmunet 仍是 **sklearn 1.3.2**，与作业 1–3 一致；本文行号在 1.3.2 下有效。`LinearRegression` 构造函数在 1.3.2 **已无 `normalize`**；`_deprecate_normalize` 仅为其它估计器的内部辅助函数。
5. **编译模块**：`_cd_fast` / `_sag_fast` / `_sgd_fast` 本地包内只有 `.so`，`inspect.getsource` 看不到内循环；如需深入改读上游 Cython 源码。
6. **Python 环境**：只用 `/root/MSA/ENTER/envs/vmunet`（用户规则）。
7. 行号只写在本文档定位用；日后做 notebook 一律用 `inspect.getsource` 取源码，避免版本漂移。

---

## Verification

1. 本文档引用的类/函数名与 1.3.2 源码一致（已逐条对照 `_base.py` / `_stochastic_gradient.py` / `_linear_loss.py` / `_glm/*` / `_ridge.py` / `_coordinate_descent.py` / `_huber.py` / `_bayes.py` / `_theil_sen.py`）。
2. 若日后做 notebook：`nbconvert --execute` 全量 0 error；逐节核对「markdown 引用」与「code dump」一致（作业 1 暴露过的主要缺陷）。
3. 关键实测结论（**已按审阅意见修正**）：
   - **R1**：稠密/sparse 两条路径在条件良好数据 + 合理容差下 `coef_` 一致；**`positive=True` 不能一般性要求一致**——分两组：① 约束**不活跃**（无真正违反非负的列）时与 OLS 一致；② 约束**活跃**时系数不同。
   - **R5**：`rank_`/`singular_` 仅在「稠密 **且** `positive=False` **且** 走 `lstsq`」时存在；稀疏、NNLS 路径应验证其**不存在**。
   - **R2**：本路径 `X_scale` 为全 1，`_set_intercept` 的除法是恒等操作（可打印验证）。
   - **R4**：整数重复样本 ≡ `sample_weight`。
   - **R9**：`learning_rate='optimal'` + `alpha=0` 触发 ValueError。
   - **R10**：`fit` 每次把 `t_` 重置为 1（含 `warm_start=True`）；`partial_fit` 延续累计。
   - **R11**：`average=True` 与默认的 `coef_` 不同。
   - **R17**：`verbose=2` 能看到线搜索与 Newton decrement 日志；可观察「两条判据都过才收敛」。
   - **R20**：不同输入下 `solver='auto'` 落到 cholesky / sparse_cg。
   - **R25**：`return_std=True` 返回的是**预测标准差**，图注应写「预测区间」而非置信带。

## Risks

- **GLM/Newton（R15–R19）较深**：已降为「进阶选读」，默认不展开；若选读，退化为「dump `solve` 主循环 + 一句结论」，不做数学证明。
- **R23 的 `_cd_fast` 是编译模块**：只能讲「何时进哪条支路」，无法展示坐标下降更新式；如需公式，用 docstring 里的目标函数替代，或读上游 Cython 源码。
- **R7 的「n_jobs 条件触发」偏静态**：改为「代码里 `n_jobs` 唯一出现处」的静态证据，避免用「无加速」这种不稳定的负面实测。
- **R21/R22 的数值细节易脆**：`n_features>n_samples` 才切核形式、Cholesky 降级需构造奇异矩阵；构造困难则退化为「代码精读 + 注释」。
- **R1 的 positive 分组**：必须构造出「约束活跃」与「不活跃」两类数据，否则实测结论不可信。
