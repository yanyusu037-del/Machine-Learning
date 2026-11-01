# 逻辑回归源码解析 —— 「值得分析的硬核差异点」方向清单

> 说明：本文是**方向清单**，不是 notebook 实施计划。覆盖范围限 `LinearClassifierMixin` + `LogisticRegression` + `LogisticRegressionCV`；公式本体（sigmoid / 交叉熵 / 梯度推导）不展开。
> 版本：`vmunet` 环境，Python 3.8.20 / **scikit-learn 1.3.2** / numpy 1.24.4 / scipy 1.10.1。本文行号仅用于定位，日后做 notebook 一律用 `inspect.getsource` 取源码，**不写死行号**。

## Summary

用户提问：**「逻辑回归的代码有什么值得分享的，先列出主题我来审核」**。审核结果：保留 P1–P10（P11 仅作延伸阅读，不单列）。

- **交付形式**：先出方向清单（同 hw3 朴素贝叶斯、hw4 线性回归、hw5 感知机）；随后再做 notebook / HTML。
- **覆盖范围**：`LogisticRegression` + `LogisticRegressionCV`；`SGDClassifier(loss="log_loss")` 不在主线（hw5 已讲 SGD 框架）。
- **筛选标准**（沿用 hw2–hw5）：优先「课堂上讲不透 / 没讲、且是 sklearn 工程实现的真实决策」。
- **核心叙事**：`LogisticRegression` 不是「一个算法 + 若干优化器」，而是 **6 个 solver 字符串 → 5 套互不共享代码的后端**，再叠加 **3 种多分类实现**（liblinear C++ 内部 OvR / Python joblib OvR / multinomial 联合优化）和 **3 份 log loss 实现**（`_loss` Cython / `_sag_fast` Cython / liblinear C++）。类本身的 Python 代码只做参数校验、数据整形、结果拼装。

---

## Current State Analysis

**工作区**：已有 `homework1-kmeans`、`homework2-KNN`、`homework3-NaiveBayes`、`homework4-LinearRegression`、`homework5-Perceptron` 的 `.ipynb` + `.html`，以及 `fix_pagebreak.py`、`README.md`；`.trae/documents/` 下已有五份计划文档（kmeans、knn、naivebayes、linearregression、perceptron）。**逻辑回归此前没有任何文件**，本文是首份材料；README 的作业表目前只到 hw5，本次不动 README。

**环境**：全部结论在本机项目环境核对（用户规则：一律用 `vmunet`，不允许 base 环境）。

| 项 | 值 |
|---|---|
| 环境 | `/root/MSA/ENTER/envs/vmunet` |
| 版本 | Python 3.8.20 / scikit-learn **1.3.2** / numpy 1.24.4 / scipy 1.10.1 |
| 调用 | `wsl -d Ubuntu-22.04 -- /root/MSA/ENTER/envs/vmunet/bin/python ...` |

**源码地图**（1.3.2，本轮已核对）：

| 文件 | 行数 | 可分析性 |
|---|---|---|
| `sklearn/linear_model/_logistic.py` | 2114 | 纯 Python。`LogisticRegression` 类体约 625 行（docstring 约 284 行），`LogisticRegressionCV` 约 697 行；`_logistic_regression_path` 是所有非 liblinear 路径的公共前端 |
| `sklearn/linear_model/_sag.py` | 372 | 纯 Python 外层（步长估计、warm start 内存、dtype 分派） |
| `sklearn/linear_model/_sag_fast*.so` | — | 本地包**只有 `.so`**，SAG/SAGA 内循环不可 `inspect.getsource` |
| `sklearn/linear_model/_linear_loss.py` | 658 | 纯 Python；`LinearModelLoss` 提供 loss / gradient / hessp / hessian |
| `sklearn/linear_model/_glm/_newton_solver.py` | 525 | 纯 Python；`NewtonSolver` 抽象基类 + `NewtonCholeskySolver` |
| `sklearn/linear_model/_glm/glm.py` | 901 | 纯 Python；GLM 估计器，与逻辑回归**复用同一套损失与牛顿求解器**，但逻辑回归不继承它 |
| `sklearn/svm/_base.py` | 1262 | 纯 Python 包装；`_fit_liblinear` / `_get_liblinear_solver_type` 可 dump |
| `sklearn/svm/_liblinear*.so`、`_loss/_loss*.so` | — | C++ / Cython 内核黑盒；`_loss/loss.py`（1164 行）只是 Python 外壳与文档 |
| `sklearn/utils/optimize.py` | 267 | `_newton_cg` / `_check_optimize_result` 可 dump |

**方法边界**：凡涉及 liblinear C++ 内循环、`_sag_fast` 内循环、`_loss` 内核算的结论，一律用**行为实测反推**；正文不假装引用不可见源码。

**本轮实测的关键事实**：

1. **六个 solver 解的是同一个目标函数，但迭代次数不可比**。同一份 `make_classification(300, 8, random_state=0)` + `StandardScaler`，二分类，`tol=1e-6, max_iter=1000`：
   - `n_iter_`：lbfgs 10、liblinear 7、newton-cg 10、newton-cholesky 5、sag 36、saga 22（全部 shape `(1,)`）；
   - 与 lbfgs 的系数最大差：liblinear `2.9e-3`（`tol=1e-10` 时为 `5.5e-4`）、newton-cg `2.4e-5`、newton-cholesky `2.6e-5`、sag `2.4e-5`、saga `2.3e-5`；
   - 目标函数值：lbfgs `111.928737`、liblinear `111.928882`、newton-cholesky `111.928737089` —— 是优化精度差异，不是目标函数不同。

2. **`n_iter_` 的形状与语义随 solver / 多分类策略变化**。digits（1797×64，10 类）：
   - `lbfgs + multinomial`：`n_iter_ = [300]`，shape `(1,)`；
   - `lbfgs + ovr`：`[152 300 213 300 262 300 280 300 300 300]`，shape `(10,)`（每类独立早停）；
   - `liblinear`：`[11 14 12 13 12 13 12 12 17 14]`，shape `(10,)`；
   - `newton-cholesky`：`[11 11 11 11 11 11 12 12 9 11]`，shape `(10,)`。

3. **`n_iter_` 的 docstring 与 liblinear 实现不一致**。docstring（`_logistic.py` L1013–1016）写「For liblinear solver, only the maximum number of iteration across all classes is given」，实测返回的是**每类迭代数的数组**（digits 上 10 个值），不是最大值。

4. **`penalty=None` 的真实实现是「把 C 换成 inf，再把 penalty 改回 l2」**（L1191–1201），不是独立的无正则路径。实测 `C=1` 与 `C=2` 在 `penalty=None` 下系数**逐元素完全相同**，并发出「Setting penalty=None will ignore the C and l1_ratio parameters」警告。

5. **liblinear 的 `penalty=None` 走的是另一条报错路径**：`_check_solver` 只拦截字符串 `"none"`（L72–73），`None` 会一路传到 `_get_liblinear_solver_type`，最终抛出 liblinear 查表式错误「Unsupported set of arguments: ... penalty=None」。同一个「关正则」，字符串与 `None` 两种写法、两套报错。

6. **dtype 能力矩阵**：`lbfgs` 强制 `np.float64`（L1203–1206），其余 solver 接受 `[float64, float32]`；但 liblinear 内部再转 float64。实测 float32 输入下，lbfgs / liblinear 的 `coef_` 是 float64，newton-cg / newton-cholesky / sag / saga 保持 float32。

7. **`sample_weight` 的整体缩放不是无关紧要的**：`Σ sw·loss + ||w||²/(2C)` 中权重与 C 耦合。把 `sample_weight` 整体乘 7，newton-cholesky 与 lbfgs 的系数都变化 `≈5.86e-2`（两者结果一致，说明 newton-cholesky 的「权重归一化 + `l2_reg_strength = 1/(C·Σsw)`」重缩放（L300–315、L476）没有改变 C 的语义，只是抵消了数值尺度）。

8. **`LogisticRegressionCV` 的默认 C 网格与数据无关**：`Cs = np.logspace(-4, 4, Cs)`（L269）。实测 `Cs_ = [1e-4, 7.74e-4, 5.995e-3, 4.642e-2, 0.359, 2.783, 21.544, 166.81, 1291.55, 10000]`。

9. **边界行为已实测**：
   - `predict` 二分类阈值是**严格 `> 0`**：决策值恰为 0 判为 `classes_[0]`；`1e-12` 即判为 `classes_[1]`。
   - OvR `predict_proba` 是「各自 sigmoid 再按行归一化」，没有零和防护；把决策值人为压到 −800 时出现 `0/0 → NaN` 并抛 `RuntimeWarning`（正常数据不会触发）。
   - `newton-cholesky + multi_class="multinomial"` → `ValueError: Solver newton-cholesky does not support a multinomial backend.`（由 `_check_multi_class` 拦截）。
   - `sag + penalty="l1"` → `ValueError: Solver sag supports only 'l2' or 'none' penalties`。
   - `liblinear + n_jobs=2` → 警告「'n_jobs' > 1 does not have any effect when 'solver' is set to 'liblinear'」。

---

## 「值得分析的硬核差异点」

> 编号用于章节组织；**P1–P5 为主线，P6–P8 为扩展，P9–P10 为选读**。每条统一给「源码位置 / 做法·现象 / 为什么不一样 / 实测点」。

**主线（必做）P1–P5**

**P1｜六个 solver 字符串 = 五套互不共享代码的后端**

- **位置**：`_logistic.py` L53–75（`_check_solver`）、L1191–1206（C 与 dtype）、L1221–1243（liblinear 直接早退）、L1273–1330（`classes_` hack 与 joblib）、L446–554（`_logistic_regression_path` 的 solver 分派）；`_sag.py` L88+（`sag_solver`）；`svm/_base.py` L1066+（`_fit_liblinear`）；`_glm/_newton_solver.py`（`NewtonCholeskySolver`）；`utils/optimize.py::_newton_cg`。
- **做法·现象**：

  | solver | 后端 | 入口 | 迭代计数来源 |
  |---|---|---|---|
  | `liblinear` | liblinear C++（坐标下降） | `fit` 里直接 `_fit_liblinear`，**不经过** path 函数 | C++ 返回每类轮数 |
  | `lbfgs` | `scipy.optimize.minimize(method="L-BFGS-B")` | path 函数 L452 | `opt_res.nit` |
  | `newton-cg` | sklearn 自写 `_newton_cg` + `LinearModelLoss.gradient_hessian_product` | path 函数 L470 | CG 迭代数 |
  | `newton-cholesky` | sklearn `NewtonCholeskySolver`（1.2 新增） | path 函数 L477 | `sol.iteration` |
  | `sag` / `saga` | Cython `_sag_fast`（本地只有 `.so`） | `_sag.py::sag_solver` | epoch 数 |

- **为什么不一样**：读一个 `LogisticRegression` 类，实际是在读 5 份实现。`solver` 不只是「优化器开关」，它同时决定：能否看源码、支持什么 penalty、`tol` 的量纲、`n_iter_` 的语义、并行方式（进程 / 线程 / 无）。
- **实测点**：六个 solver 的 `n_iter_` 见「关键事实 1」；同一数据系数与 lbfgs 的最大差也在一张表里。notebook 可直接做「六 solver 同解」表格。

**P2｜C、`penalty=None`、`sample_weight`：同一个目标函数的三种写法**

- **位置**：`_logistic.py` L1191–1201（`penalty=None → C_=inf, penalty='l2'`）、L448 / L468（`l2_reg_strength = 1.0 / C`）、L300–315 与 L476（newton-cholesky 的权重重缩放）、`_linear_loss.py` L170–173（`0.5 * l2_reg_strength * ||w||²`）；liblinear 侧 `_get_liblinear_solver_type` L1025–1026。
- **做法·现象**：统一的未归一化目标可写成 `Σ sample_weight·log_loss + ||w||²/(2C)`。liblinear 的 C++ 目标写成 `0.5||w||² + C·Σloss`，数学上等价；`sag/saga` 把 `alpha=1/C` 再除 `n_samples`（`_sag.py` L256）；newton-cholesky 先把 `sample_weight` 归一到和为 1，再把 `l2_reg_strength` 改成 `1/(C·Σsw)`（源码注释原话：`nasty matter`）。
- **为什么不一样**：用户看到的 `C` 是同一个数字，但进入后端前被换了三种尺度；`penalty=None` 也不是「关正则分支」，而是「让 `1/C → 0`」。这解释了为什么不同 solver 的 `C` 结果在数值精度内一致，但 `sample_weight` 整体缩放会改变模型。
- **实测点**：`penalty=None` 下 `C=1` 与 `C=2` 系数逐元素相同；`sample_weight × 7` 后两种 solver 系数都变 `≈5.86e-2`；liblinear 与 lbfgs 的目标函数值差 `1.45e-4`（`tol=1e-10`）。

**P3｜`n_iter_` 有四种含义，而且 docstring 对 liblinear 写错了**

- **位置**：`_logistic.py` L1013–1016（docstring）、L460–465 / L470–471 / L486–487（三种计数器）、L488–514（liblinear）、L533–548（SAG epoch）、L1328–1329（组装）。
- **做法·现象**：`n_iter_` 统一是 `np.int32` 数组，但：lbfgs 是 L-BFGS 迭代数；newton-cg 是共轭梯度步数；newton-cholesky 是牛顿迭代数；sag/saga 是**整轮 epoch**；liblinear 是坐标下降轮数。二分类恒为 `(1,)`；OvR 多分类是 `(n_classes,)`；multinomial 是 `(1,)`。
- **为什么不一样**：同一个属性名承载了四种量纲，跨 solver 直接比较 `n_iter_` 没有意义。docstring 对 liblinear 的描述（只给各类最大值）与 1.3.2 实现（返回每类数组）不符 —— 与 hw5 的 `t_` docstring 问题同类，但这是另一个属性。
- **实测点**：digits 上 `lbfgs+ovr` 与 `liblinear` 都是 `(10,)`，但数值规律完全不同；`lbfgs+multinomial` 是 `(1,)`。`max_iter=200` 时 lbfgs multinomial 恰好 `[200]` 并伴随 `ConvergenceWarning`。

**P4｜多分类是三种实现，而不是一个 `multi_class` 开关**

- **位置**：`_check_multi_class` L78–93；`fit` L1219、L1273–1330；`_logistic_regression_path` L327–364（OvR 编码成 ±1 或 {0,1}）、L405–423（multinomial 目标）、L556–566（结果拼装）；`_fit_liblinear` L1228。
- **做法·现象**：
  1. **liblinear**：把原始多类标签一次性交给 C++，liblinear **内部**做 OvR，返回 `(n_classes, n_features+1)`；
  2. **lbfgs / newton-cg / sag / saga**：Python 层用 joblib `Parallel` 按类训练，每类一个二分类器，`n_iter_` 每类独立；
  3. **multinomial**：只有一条联合优化路径，`classes_` 被替换成 `[None]` 使 joblib 只迭代一次（L1273–1276）。
- **为什么不一样**：同一个 `fit` 里同时存在「C++ 内部 OvR」「Python 进程/线程级 OvR」「联合 softmax」三种语义；`multi_class="auto"` 不是「自动选最好」，而是一张 **solver 能力表**（liblinear / newton-cholesky 强制 OvR）。
- **实测点**：digits 三条路径 `coef_` 都是 `(10, 64)`，但 `n_iter_` 形状 `(1,)` vs `(10,)`；`newton-cholesky + multinomial` 直接 ValueError；liblinear 的 `n_jobs>1` 警告。

**P5｜二分类 multinomial 只保存一行系数：半参数化**

- **位置**：`_logistic_regression_path` L556–566（`if n_classes == 2: multi_w0 = multi_w0[1][np.newaxis, :]`）；`predict_proba` L1386–1393（`np.c_[-decision, decision]`）；`_base.py` L434（`scores.shape[1] == 1` 时压成 1D）。
- **做法·现象**：multinomial 二分类理论上应有两行系数 `(2, n_features)`，sklearn 只保留第 1 行，把第 0 行固定为 0；预测时用 `[-d, d]` 造出两类分数再 softmax。因此 `coef_.shape` 是 `(1, n_features)`，`decision_function` 返回 1D。
- **为什么不一样**：softmax 对「所有类分数加同一常数」不变，所以二分类只有**相对**参数可辨识；sklearn 用「第 0 类分数恒为 0」固定规范。读者若按教科书以为有两行系数，会在 `coef_` 形状上困惑。
- **实测点**：二分类 `multi_class="ovr"` 与 `"multinomial"` 的 `coef_` 都是 `(1, 8)`、`decision_function` 都是 `(3,)`、`predict_proba` 行和都是 1。

**扩展（选做）P6–P8**

**P6｜两种牛顿法：matrix-free Hessian-vector product vs 显式矩阵 + Cholesky**

- **位置**：`_linear_loss.py` L375–510（`gradient_hessian`）、L511–660（`gradient_hessian_product` / `hessp`）；`_glm/_newton_solver.py` L436–506（`NewtonCholeskySolver.inner_solve` 显式 `scipy.linalg.solve(assume_a="sym")`）、L199–298（回溯线搜索）、L302–343（双收敛判据）、L464–506（三种 fallback 到 L-BFGS）。
- **做法·现象**：`newton-cg` 只接受一个 `hessp`：二分类用 `X.T @ (hX @ s)`（`multi_dot` / 稀疏 `dia_matrix`），multinomial 用源码里那段带下标推导的实现；`newton-cholesky` 则分配 `(n_dof, n_dof)` 的 `hessian` 并做 Cholesky 分解。`NewtonSolver.solve` 的顺序是 `update_gradient_hessian → inner_solve → line_search → check_convergence`。
- **为什么不一样**：同叫「牛顿法」，一个内存 `O(n_features)`、一个内存 `O(n_dof²)`；一个每步解子问题用 CG，一个直接分解。后者还内置「Hessian 负对角 / 奇异 / 非下降方向 → 自动改走 L-BFGS」的自愈逻辑，这在教材里没有。
- **实测点**：`verbose=2` 可打印每步线搜索与收敛判据；构造共线特征可触发 `LinAlgWarning` 并观察 fallback 警告；`newton-cholesky` 在二分类上 `n_iter_=5`（六 solver 里最少）。

**P7｜`LogisticRegressionCV` 不是「循环调 LogisticRegression」**

- **位置**：`_logistic.py` L269（固定 C 网格）、L574–789（`_log_reg_scoring_path`）、L1829–1830（`max_squared_sum` 在全量 X 上计算）、L1880–1907（joblib 三维循环：类 × 折 × l1_ratio）、L1920–1946（张量 reshape）、L1964–2007（`refit=True` 用各折最优系数的均值热启动）、L2010–2025（`refit=False` 直接取各折最优系数的平均）。
- **做法·现象**：一次 `_logistic_regression_path` 在 C 序列上**沿路径 warm start**（`w0` 与 SAG 的 `warm_start_mem` 跨 C 复用，L446–570）；默认 C 网格是写死的 `logspace(-4, 4, n_Cs)`，与数据规模无关；`Cs` 可以是数组；`scoring` 默认是 `log_reg.score`（accuracy，L784–787）；`refit=False` 时 `coef_` 是各折最优系数的平均，**不对应任何真实训练出的模型**。
- **为什么不一样**：CV 版本的价值在「路径求解 + 热启动」，而不是「外层 for 循环」；此外 `sag/saga` 的步长用**全量数据**的 `max_squared_sum`（含验证折），只是步长上界，不影响模型但会影响收敛速度。
- **实测点**：`Cs_` 固定十个数（见「关键事实 8」）；`coefs_paths_` / `scores_` 是 `dict: 类标签 → (n_folds, n_Cs)`（源码注释 L1914–1919）；`refit=False` 手算平均可与 `coef_` 对齐。

**P8｜参数「静默忽略」清单：`intercept_scaling`、`warm_start`、`n_jobs`、`dual`**

- **位置**：`_logistic.py` L1203–1213（`intercept_scaling` 只在 liblinear 分支使用）、L1221–1227（liblinear 的 `n_jobs` 警告）、L1243（liblinear 在读取 `warm_start` 之前就 `return`，而 warm start 代码在 L1264）、L60–62（`dual` 非 liblinear 报错）；`svm/_base.py` L1025–1026（liblinear 魔数表：l2 primal 0 / l2 dual 7 / l1 primal 6）。
- **做法·现象**：`intercept_scaling` 在 lbfgs 下实测无警告且截距完全不变；`warm_start=True` 对 liblinear **结构性无效**（代码路径根本走不到）；`n_jobs` 对 liblinear 只发警告；`dual=True` 只 liblinear 支持；`l1_ratio` 非 elasticnet 时警告；`penalty="none"` 是 FutureWarning，`penalty=None` 是另一套行为（见 P2）。
- **为什么不一样**：参数是否存在（签名）、是否校验（`_parameter_constraints`）、是否生效（solver 分支）是三层独立的事；课堂上「设了参数就该生效」的假设在这里不成立。这条与 hw5 P5（`loss` / `average` 从签名里删掉）是同一类工程现象的不同变体。
- **实测点**：`liblinear warm_start=True` 两次 `fit` 的 `n_iter_` 都是 `[5]`（未续热）；`lbfgs intercept_scaling=100` 与默认截距 `np.allclose`；`liblinear n_jobs=2` 警告文本已实测。

**选读 P9–P10**

**P9｜概率输出：OvR 行归一化 vs softmax；二分类阈值严格 `> 0`**

- **位置**：`_base.py` L459–473（`_predict_proba_lr`：`expit` + `prob /= prob.sum(axis=1)`）、L452–455（`predict` 二分类 `astype(scores > 0, int)`，多分类 `argmax`）；`_logistic.py` L1383–1393（OvR / multinomial 分流、二分类 `np.c_[-d, d]`）。
- **做法·现象**：OvR 的各类 sigmoid 输出**本来不是联合概率**，sklearn 用行归一化强行凑成 1；multinomial 才用 `softmax`。二分类 `predict` 用严格 `>`，决策值恰为 0 判第 0 类；多分类 `argmax` 平票取首个。
- **为什么不一样**：同一个 `predict_proba`，两种数学来源；`predict` 的零点归属是一个真实实现决策（与 hw5 感知机更新判据 `<= 0` 形成对照）。
- **实测点**：`score=0 → classes_[0]`，`score=1e-12 → classes_[1]`；人工构造 −800 决策值时 OvR 概率出现 `NaN` + `RuntimeWarning`（正常数据不会触发，作为边界展示即可，不夸大）。

**P10｜同一份 log loss 的三份实现 + dtype 能力矩阵**

- **位置**：`_logistic.py` L23（`HalfBinomialLoss / HalfMultinomialLoss` 来自 `sklearn._loss.loss`）、L405–441（按 solver 选损失）；`_sag.py` L516–548（`loss="log" / "multinomial"` 传给 `_sag_fast`）；`svm/_base.py` L1026（liblinear `logistic_regression`）；`_logistic.py` L1292–1301（`n_threads` 的两个分支都写死为 1）。
- **做法·现象**：lbfgs / newton-cg / newton-cholesky 用 `_loss`（Cython + OpenMP，函数签名里有 `n_threads`）；sag/saga 用 `_sag_fast` 自己的 log / multinomial 分支；liblinear 用 C++ 的 LR loss。`n_threads` 一路传到 `LinearModelLoss`，但 `fit` 里两个分支都写死 `n_threads = 1`（源码 TODO 原话：「In the future, we would like to ...」）。
- **为什么不一样**：数学上是一个损失，工程上是三份代码；`float32` 只有非 lbfgs 路径保留，lbfgs 会静默升级到 float64；OpenMP 并行在 `LogisticRegression.fit` 里实际未启用。
- **实测点**：见「关键事实 6」的 dtype 表；`n_threads` 两个分支均为 1 可直接从 L1292–1301 读出。

**延伸（不单列）P11｜逻辑回归与 GLM 共用底层，但不继承 GLM**

- `_logistic.py` L42 从 `_glm.glm` 导入 `NewtonCholeskySolver`，L43 导入 `LinearModelLoss`；`_glm/glm.py`（901 行）的 `PoissonRegressor` / `TweedieRegressor` 复用同一套损失与牛顿框架。可作为 P6 的延伸阅读：1.3 的工程方向是「逻辑回归 = 二项 GLM 的损失 + 分类外壳」，但 `LogisticRegression` 仍是独立估计器。**不进主线**，避免把 hw6 扩成 GLM 专题。

---

## 去重与取舍

**与 hw2–hw5 的重叠**

| 已讲内容 | 出处 | hw6 的处理 |
|---|---|---|
| SGD 框架、`loss_functions` 表、`Hinge(0)`、`t_`、`tol/early_stopping` | hw5 P1–P7 | **不重复**；仅在 P3 提一句「`n_iter_` 与 `t_` 是不同属性」 |
| `make_dataset`、稀疏截距衰减 `SPARSE_INTERCEPT_DECAY=0.01` | hw4 R13 / hw5 P8 | **不重复**；`sag_solver` 也走 `make_dataset`，只作脚注 |
| Python 层 OvR + joblib | hw5 P5 | hw6 P4 升级为「三种多分类实现并存」，重点在 liblinear 内部 OvR 与 multinomial 的对照 |
| `class_weight` 折叠进 SGD 的 C 循环 | hw5 P9 | **不单列**；P2 只讲 `sample_weight` 的缩放语义 |
| 概率归一化 / logsumexp | hw3（NB 的「全零概率」） | hw6 P9 是 OvR sigmoid 行归一化，属于另一层，需在正文注明区别 |
| `predict_proba` 缺失 | hw5 P10 | hw6 是「有 `predict_proba` 但有两种来源」，正好接上 |

**主线（必做）P1–P5**

1. 六 solver 分发表（P1）
2. C / `penalty=None` / `sample_weight` 的目标函数缩放（P2）
3. `n_iter_` 的四语义与 docstring 冲突（P3）
4. 多分类三实现（P4）
5. 二分类 multinomial 半参数化（P5）

**扩展（选做）P6–P8**

6. 牛顿法两实现与失败自愈（P6）
7. `LogisticRegressionCV` 的路径求解（P7）
8. 参数静默忽略清单（P8）

**选读 P9–P10**

9. 概率输出与阈值（P9）
10. 三份 log loss + dtype 表（P10）

**明确不做**

- 不推导 sigmoid / 交叉熵 / 梯度 / 正则化的数学性质。
- 不读 liblinear C++ 内循环、不读 `_sag_fast` / `_loss` 的 Cython 内核（本地只有 `.so`）；相关结论只用行为实测。
- 不讲 `SGDClassifier(loss="log_loss")`（hw5 已覆盖 SGD 框架）。
- 不展开 GLM（`PoissonRegressor` / `TweedieRegressor`）与 `LogisticRegressionCV` 的 scorer 细节；P11 只作延伸阅读。
- 不重复 hw4 / hw5 已讲内容（见上表）。

---

## 待确认：若后续做 notebook 的结构

> 未获确认不执行。预计文件名：`homework6-LogisticRegression.ipynb` + `.html`。

| 节 | 内容 | 对应主题 |
|---|---|---|
| 0 | 环境与源码地图（哪些能 dump、哪些是 `.so`） | — |
| 1 | 六 solver 分发表 + 同解实测 | P1 |
| 2 | `C` / `penalty=None` / `sample_weight` 的三种缩放 | P2 |
| 3 | `n_iter_` 四语义与 docstring 冲突 | P3 |
| 4 | 多分类三实现（liblinear / joblib OvR / multinomial） | P4 |
| 5 | 二分类 multinomial 半参数化 | P5 |
| 6 | newton-cg 的 hessp vs newton-cholesky 的显式 Hessian | P6 |
| 7 | `LogisticRegressionCV` 路径与固定 C 网格 | P7 |
| 8 | 静默忽略参数清单（含 liblinear 魔数） | P8 |
| 9 | 概率输出与 `> 0` 阈值（含边界实测） | P9 |
| 10 | 三份 log loss 与 dtype / `n_threads` | P10 |
| 11 | 总结：一张「参数 → 生效位置」表 | 全部 |

---

## Assumptions & Decisions

1. **版本**：vmunet 的 sklearn **1.3.2**，与 hw1–hw5 一致；所有行号与结论均在本版本核对。
2. **范围**：`LogisticRegression` + `LogisticRegressionCV`；`multi_class` 的 `"auto" / "ovr" / "multinomial"` 三种取值都覆盖；不扩展到 GLM / SGD。
3. **可读性边界**：liblinear（C++ `.so`）、`_sag_fast`（Cython `.so`）、`_loss`（Cython `.so`）的内核不可读；相关结论全部标注为「行为实测」。
4. **数值证据**：本文所有数字来自本轮 vmunet 实跑；notebook 中用固定 `random_state` 复现，不写「大概 / 约」。
5. **交付节奏**：本轮只出方向清单；notebook / HTML 需用户再次确认后才做。
6. **不改代码**：本轮不新建 notebook、不导出 HTML、不改 `README.md`、不提交 git。

---

## Verification

1. 文件存在且包含 Summary / Current State Analysis / P1–P10 / 去重与取舍 / notebook 结构 / Assumptions / Verification / Risks。
2. 每条 P# 都有：源码位置、与课堂/教科书的差异、可执行实测；没有「待补充」占位。
3. 文中数字与下列实跑一致（可重跑复核）：

```bash
# 六 solver 同解 + n_iter_（二分类）
wsl -d Ubuntu-22.04 -- /root/MSA/ENTER/envs/vmunet/bin/python - <<'PY'
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
X, y = make_classification(n_samples=300, n_features=8, n_informative=5, random_state=0)
Xs = StandardScaler().fit_transform(X)
for s in ["lbfgs","liblinear","newton-cg","newton-cholesky","sag","saga"]:
    m = LogisticRegression(solver=s, tol=1e-6, max_iter=1000).fit(Xs, y)
    print(s, m.n_iter_, m.coef_.ravel()[:2])
PY
```

```bash
# n_iter_ 形状（digits 多分类）
wsl -d Ubuntu-22.04 -- /root/MSA/ENTER/envs/vmunet/bin/python - <<'PY'
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_digits
X, y = load_digits(return_X_y=True)
for kw in [dict(solver="lbfgs", multi_class="ovr", max_iter=300),
           dict(solver="lbfgs", multi_class="multinomial", max_iter=300),
           dict(solver="liblinear", max_iter=200),
           dict(solver="newton-cholesky", max_iter=200)]:
    m = LogisticRegression(**kw).fit(X, y)
    print(kw, m.coef_.shape, m.n_iter_, m.n_iter_.shape)
PY
```

```bash
# penalty=None 忽略 C；liblinear 的另一条报错
wsl -d Ubuntu-22.04 -- /root/MSA/ENTER/envs/vmunet/bin/python - <<'PY'
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
import numpy as np
X, y = make_classification(n_samples=100, n_features=5, random_state=0)
a = LogisticRegression(penalty=None, C=1).fit(X, y)
b = LogisticRegression(penalty=None, C=2).fit(X, y)
assert np.allclose(a.coef_, b.coef_) and np.allclose(a.intercept_, b.intercept_)
try:
    LogisticRegression(penalty=None, solver="liblinear").fit(X, y)
except ValueError as e:
    print(str(e)[:60])
print("ok")
PY
```

```bash
# dtype 能力矩阵
wsl -d Ubuntu-22.04 -- /root/MSA/ENTER/envs/vmunet/bin/python - <<'PY'
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=100, n_features=5, random_state=0)
X = X.astype("float32")
for s in ["lbfgs","liblinear","newton-cg","newton-cholesky","sag","saga"]:
    print(s, LogisticRegression(solver=s, max_iter=300).fit(X, y).coef_.dtype)
PY
```

4. 不出现「本地可读 `_sag_fast.pyx` / `_loss.pyx` / liblinear 源码」之类错误表述。
5. 明确标注 notebook 结构「未获确认不执行」；`README.md` 未改动；无新增 `ipynb` / `html`。

---

## Risks

- **行号漂移**：文中行号只作 1.3.2 的定位提示；若 vmunet 升级，需重新核对 `_logistic.py` 的 path 分派、CV 网格与 numpy 版本。
- **liblinear 非确定性**：docstring Notes（L1029–1034）明确说 C 实现用随机数选特征，同数据结果可能略有差异；P1/P8 的数值必须带 `random_state` 并说明「不保证逐位一致」。
- **浮点比较**：`float32` 路径与 `float64` 路径的 `allclose` 阈值要放对；P2 的 `sample_weight ×7` 结论只对 `tol=1e-10` 成立，需在 notebook 固定。
- **ConvergenceWarning 噪声**：digits 上 `max_iter=200` 的 lbfgs multinomial 必然触发警告；正文要把「警告」与「没收敛」分开讲。
- **CV 运行成本**：`LogisticRegressionCV` 默认 `Cs=10` × `cv=5` × 10 类，notebook 需缩小数据或固定 `cv=3`，否则执行时间不可控。
