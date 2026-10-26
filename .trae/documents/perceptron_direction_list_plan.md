# 感知机（Perceptron）方向清单 —— 实施计划

## Summary

用户提问：**「感知机有什么类似的内容可以做的」**。澄清结果：

- **交付形式**：只出**方向清单**（与 hw3 朴素贝叶斯、hw4 线性回归第一版一致），本次**不产出** notebook / HTML。
- **覆盖范围**：**只讲 sklearn 的 `Perceptron` 本身**，不展开 PassiveAggressive（PA-I/PA-II）、不把 SGD hinge 作为主线。
- **筛选标准**（沿用 hw2 / hw3 / hw4）：优先「课堂上讲不透 / 没讲、且是 sklearn 工程实现的真实决策」；感知机公式本体（`w ← w + ηyx`）不展开推导。
- **核心叙事**：`Perceptron` 不是独立算法，而是 `BaseSGDClassifier` 的**一组被锁死的默认配置** + 损失表里 `Hinge(θ=0)` 这一行；221 行里只有约 53 行是代码。

### 本次唯一产出

新建一份文档：

`\\wsl.localhost\Ubuntu-22.04\root\Machine Learning\.trae\documents\perceptron_code_analysis_plan.md`

不新建 notebook、不导出 HTML、不改 `README.md`、不提交 git。

---

## Current State Analysis

### 工作区现状

- 已有：`homework1-kmeans`、`homework2-KNN`、`homework3-NaiveBayes`、`homework4-LinearRegression` 的 `.ipynb` + `.html`，`fix_pagebreak.py`，`README.md`。
- 已有四份计划：`kmeans_code_match_plan.md`、`knn_code_analysis_plan.md`、`naivebayes_code_analysis_plan.md`、`linearregression_code_analysis_plan.md`。
- **感知机尚无任何文件**；本次只补一份方向清单。
- README 的作业表本来就滞后（hw3/hw4 未回填），本次**不动 README**。

### 环境（本轮已实测）

| 项 | 值 |
|---|---|
| 环境 | `/root/MSA/ENTER/envs/vmunet` |
| 版本 | Python 3.8.20 / scikit-learn **1.3.2** / numpy 1.24.4 / scipy 1.10.1 |
| 调用 | `wsl -d Ubuntu-22.04 -- /root/MSA/ENTER/envs/vmunet/bin/python ...` |

> 用户规则：一律用项目环境 `vmunet`，**不允许 base 环境**。

### 源码地图（1.3.2，本轮已核对）

| 文件 | 行数 | 可分析性 |
|---|---|---|
| `sklearn/linear_model/_perceptron.py` | 221 | 纯 Python；**L10–167 是 docstring（158 行）**，L169–221 才是代码 |
| `sklearn/linear_model/_stochastic_gradient.py` | 2574 | 纯 Python，外层流程全可 dump |
| `sklearn/linear_model/_sgd_fast.pyx` | 不存在 | 本地包**没有 `.pyx`** |
| `sklearn/linear_model/_sgd_fast.pxd` | 26 | 只声明 `LossFunction/Regression/Classification/Log/SquaredLoss`，**不暴露 Hinge** |
| `_sgd_fast.cpython-38-x86_64-linux-gnu.so` | — | 只有二进制；更新内循环不可 `inspect.getsource` |
| `sklearn/linear_model/_base.py` | — | `make_dataset` L136–186、`SPARSE_INTERCEPT_DECAY` L52–54 可 dump |

**方法边界**：凡涉及 Cython 内循环（更新式、loss 值、epoch 停止判定）的结论，**一律用行为实测反推，不假装引用源码**。

### 已实测确认的关键事实（写入方向清单的证据基础）

1. **等价性**：`Perceptron(random_state=0)` 与
   `SGDClassifier(loss="perceptron", penalty=None, eta0=1.0, learning_rate="constant", random_state=0)`
   在 60 个样本上 `coef_` / `intercept_` / `n_iter_` **逐元素完全相等**（`np.array_equal` 为 True；`n_iter_=15`，`t_=901`）。
2. **损失映射**：`_stochastic_gradient.py` **L499–509** 的 `loss_functions` 表中，
   `"perceptron": (Hinge, 0.0)`（**L502**），而 `"hinge": (Hinge, 1.0)`（L500）。同一个 `Hinge` 类，只差一个阈值参数。
3. **Hinge 阈值行为**（构造 `X=[[1],[5]]`, `y=[1,0]`, `sample_weight=[1,0]`, `coef_init=0`, `intercept_init=m`）：
   - `m=0.5`：Perceptron **不更新**（coef 仍为 0，b=0.5）；hinge **更新**；
   - `m=2.0`：两者都不更新；
   - `m=-0.5`：两者都更新；Perceptron 步长 1.0（`eta0=1` + `learning_rate='constant'`），hinge 默认 `learning_rate='optimal'` 下步长 10.0（对照时必须显式改 constant）。
   → 结论：`Hinge(θ=0)` 的判定退化为「零分界」，`Hinge(θ=1)` 是「间隔 1 的线性 SVM」。
4. **教科书循环可复现**：`make_blobs(n_samples=200, centers=2, random_state=7, cluster_std=2.0)`，标签映射为 ±1，手工循环
   `if y*(x@w+b) <= 0: w += y*x; b += y`，配合 `shuffle=False, tol=None, max_iter=1000`，
   与 sklearn 的 `coef_` / `intercept_` **完全一致**（共 14526 次更新；sklearn `n_iter_=1000`、`t_=200001`）。
   同样数据用默认 `tol=1e-3` 时，60 样本的案例只跑 `n_iter_=15` 就停。
5. **`t_` 语义**：docstring（`_perceptron.py` L137–139）称「Number of weight updates… Same as `(n_iter_ * n_samples + 1)`」，
   但实测它就是**处理过的样本数 + 1**：
   - 二分类 60 样本 × 15 轮 → `t_=901 = 15*60+1`；
   - digits 多分类 1797 样本 × 22 轮 → `t_=39535 = 22*1797+1`；
   - `early_stopping=True` 时 100 样本 × 6 轮 → `t_=601`；
   - `partial_fit` 两次 → `t_` 101 → 201，而 `n_iter_` 始终为 1。
6. **稀疏截距衰减**：`make_dataset`（`_base.py` L178–184）对稀疏输入返回 `intercept_decay=SPARSE_INTERCEPT_DECAY=0.01`，稠密返回 1.0。
   同一份 `make_blobs(100, random_state=3, cluster_std=1.5)` 数据，`shuffle=False, max_iter=30, tol=None`：
   - 稠密：`intercept_=3.0`，`coef_≈[-9.90, -7.04]`；
   - 稀疏（`csr_matrix`）：`intercept_=0.14`，`coef_≈[-9.38, -8.87]`；
   - 两者 `t_` 都是 3001。
7. **停止条件**：`tol=None, max_iter=3` **不报警**；`tol=1e-3, max_iter=3` 报 `ConvergenceWarning`（`_stochastic_gradient.py` L708–720）。
8. **参数校验三处不一致**：
   - `Perceptron(average=True)` / `Perceptron(loss="hinge")` → **构造期 TypeError**（__init__ 根本没有这两个参数）；
   - `Perceptron(eta0=0.0)` → **构造通过**，`fit` 时 ValueError「eta0 must be > 0」（L149–153）；
   - `Perceptron(alpha=-1.0)` → **构造期** `InvalidParameterError`；
   - `Perceptron(penalty="l1")` → 正常 fit（会正则化，偏离教科书）。
9. **多分类**：digits 上 `coef_.shape=(10, 64)`、`intercept_.shape=(10,)`、`n_iter_=22`；`_fit_multiclass`（L757–810）用 joblib `Parallel(n_jobs=..., require="sharedmem")`，`n_iter_` 取各二分类的**最大值**（L794–801）。
10. **没有 `predict_proba`**：`Perceptron` 实例只有 `decision_function` / `predict` / `score`；`hasattr(p, "predict_proba")` 为 False。

---

## Proposed Changes

### 变更 1：新建方向清单文档（唯一改动）

**文件**：`.trae/documents/perceptron_code_analysis_plan.md`

**做什么**：按 hw3 / hw4 方向清单的版式写，中文，含以下固定章节：

1. `# 感知机源码解析 —— 「值得分析的硬核差异点」方向清单`
2. `## Summary` —— 用户提问、澄清结论（只出清单 / 只讲 Perceptron）、核心叙事。
3. `## Current State Analysis` —— 工作区现状、环境、源码地图、可分析性边界（`.pyx` 缺失、`.pxd` 不暴露 Hinge）。
4. `## 「值得分析的硬核差异点」` —— 按下列 **P1–P10** 逐条写，每条统一格式：
   `**P#｜标题**` + `- 位置：` + `- 做法/现象：` + `- 为什么不一样：` + `- 实测点：`（贴本轮已验证的数值）。
5. `## 与 hw4（线性回归 / SGD）的重叠与去重` —— 明确哪些点 hw4 已讲、本条只取感知机独有结论。
6. `## 取舍` —— 主线 / 扩展 / 选读 / 明确不做。
7. `## 待确认：若后续做 notebook 的结构` —— 标注「未获确认不执行」，给 9–10 节表格。
8. `## Assumptions & Decisions`。
9. `## Verification`（文档自身的核对清单）。
10. `## Risks`。

### P1–P10 内容规格（executor 按此写，不需再决策）

#### 主线（必做，P1–P5）

| 编号 | 方向 | 源码位置 | 必须写清的「不一样」 | 实测点 |
|---|---|---|---|---|
| **P1** | **221 行里 158 行是 docstring：`Perceptron` 是「减法 API」的预设档位** | `_perceptron.py` L10–167（docstring）、L169–179（约束字典 `pop("loss")` / `pop("average")`）、L181–221（__init__ 硬编码 `loss="perceptron"`、`learning_rate="constant"`、`eta0=1.0`、`power_t=0.5`） | 它不是新算法：把基类的 `loss` / `average` 两个公开参数**从约束表里删掉**，再把 4 个参数写死；`penalty=None` 才是「教科书感知机」的前提 | `Perceptron(average=True)`、`Perceptron(loss="hinge")` 均构造期 `TypeError`；`Perceptron(penalty="l1")` 可 fit 但已不是教科书更新 |
| **P2** | **「感知机损失」其实是 `Hinge(θ=0)`：和线性 SVM 只差一个阈值** | `_stochastic_gradient.py` L499–509，尤其 L500 `hinge→(Hinge,1.0)` 与 L502 `perceptron→(Hinge,0.0)`；`_get_loss_function` L165–171 | 课堂上「感知机 = 误分类驱动」「SVM = 间隔驱动」被实现成**同一个 Cython 类 + 一个 double 参数**；`loss_function_` 是 `_sgd_fast.Hinge` 对象，`cdef` 方法在 Python 侧不可调用（无 `.loss` 属性），所以只能用行为反推 | 三点探针（m=0.5 / 2.0 / −0.5）结果如「关键事实 3」；对照 hinge 时必须显式 `learning_rate="constant", eta0=1, penalty=None`，否则默认 optimal 步长 10.0 会污染结论 |
| **P3** | **教科书循环可逐元素复现：更新式、遍历顺序、停止时机** | 行为层面对 `_plain_sgd` 黑盒；外层 `fit_binary` L359–491、`_prepare_fit_binary` L327–356 | 只有在 `shuffle=False`、`tol=None`、稠密、无 penalty、无 early_stopping 时，才能退化成「按行顺序、误分类即更新、跑满 max_iter」；默认 `tol=1e-3` 会在 loss 不再下降时提前停，和「零错误才停」不是一回事 | 200 样本非可分数据：手工循环与 sklearn 完全一致（14526 次更新，`n_iter_=1000`, `t_=200001`）；60 样本默认 tol 只跑 15 轮。**精确边界（决策值恰好为 0 是否更新）留作 notebook 单样本构造实验**，正文不要断言 |
| **P4** | **`t_` 的 docstring 在撒谎：它数的是样本，不是权重更新次数** | `_perceptron.py` L137–139；`_stochastic_gradient.py` L692（fit 重置 `t_=1.0`）、L740 / L800（`t_ += n_iter_ * n_samples`）、L623–624（partial_fit 首次置 1.0） | docstring 写「Number of weight updates… Same as `n_iter_ * n_samples + 1`」，而后者本身就是样本数公式；实测在非可分数据上**更新次数 14526 ≠ t_ 200001**，说明 `t_` 不是更新次数 | 901 / 39535 / 601 / 101→201 四组数值；`partial_fit` 的 `n_iter_` 恒为 1（`max_iter=1`），`t_` 跨调用累积，`fit` 每次重置 |
| **P5** | **多分类 = OvA + joblib：`n_jobs` 只在多分类生效，`n_iter_` 取最大值** | `_stochastic_gradient.py` L627–646（>2 类分派）、L757–810（`_fit_multiclass`）、L773–792（`Parallel(require="sharedmem")`）、L794–801 | 感知机没有 softmax / 多类联合更新；每个类各训一个「本类 vs 其余」的二分类器，**每个二分类器独立早停**，最后 `n_iter_` 是 max 而非 sum；二分类路径完全不碰 joblib | digits：`coef_(10,64)`、`intercept_(10,)`、`n_iter_=22`、`t_=39535`；`n_jobs` 在二分类上是无效参数 |

#### 扩展（选做，P6–P8）

| 编号 | 方向 | 源码位置 | 要点 | 实测点 |
|---|---|---|---|---|
| **P6** | **`tol` 比较的是「本 epoch 平均 loss」而非权重变化；`tol=None` 静默跑满 `max_iter`** | docstring `_perceptron.py` L42–46；`_stochastic_gradient.py` L450（`tol=None → -inf`）、L708–720（仅 `tol` 非空且 `n_iter_==max_iter` 才警告） | 课堂上「收敛」通常指「不再误分类 / 权重不再变」；这里是**训练损失的下降幅度 < tol**，且非可分数据上它可能很晚才触发；`tol=None` 时既不会早停也不会警告 | `tol=None, max_iter=3` 无警告；`tol=1e-3, max_iter=3` 报 `ConvergenceWarning` |
| **P7** | **early_stopping：在原始多分类标签上做分层切分，所有 OvA 共享同一掩码** | `_stochastic_gradient.py` L257–310（`StratifiedShuffleSplit`）、L312–324（回调）、L763–765（多分类先按原标签切）、L147–148（partial_fit 禁用） | 早停判据是**验证集 score 连续 `n_iter_no_change` 轮不提升 ≥ tol**，与 P6 的训练 loss 判据是两套机制；切分发生在 OvA 之前，所以每个二分类器看到同一批验证样本；`partial_fit` 直接 ValueError | `early_stopping=True` → `n_iter_=6, t_=601`；`partial_fit + early_stopping` → `ValueError: early_stopping should be False with partial_fit` |
| **P8** | **稀疏输入的截距每步只走 1%：`SPARSE_INTERCEPT_DECAY=0.01`** | `_base.py` L52–54、L136–186（L178–184 分支）；`fit_binary` L432–434 | 同一份数据、同一超参，`csr_matrix` 与稠密数组会得到**不同模型**，不是数值误差；源码注释给的理由是「避免截距振荡」。感知机不加正则，这个衰减直接改变决策边界 | 稠密 `b=3.0` vs 稀疏 `b=0.14`；`coef_` 也随之不同（`[-9.90,-7.04]` vs `[-9.38,-8.87]`）；`t_` 均为 3001 |

> P8 与 hw4 的 R13 同源（同一个 `make_dataset` 分支）。文档中必须注明「机制已在 hw4 讲过，本条只补**感知机上的后果**：无正则时截距与系数同时漂移」。

#### 选读（P9–P10）

| 编号 | 方向 | 源码位置 | 要点 | 实测点 |
|---|---|---|---|---|
| **P9** | **`class_weight` 不是重采样，而是把权重乘进 C 循环** | `_stochastic_gradient.py` L603–606（`compute_class_weight`）、L432–434 与 L474–475（`pos_weight` / `neg_weight` 入 C）；`sample_weight` 与之相乘 | `class_weight="balanced"` 的公式是 `n_samples / (n_classes * bincount)`，但实现上不复制样本、不改数据，只把两个标量权重交给内循环；与 `sample_weight` 是**乘性叠加** | notebook 中可用「复制少数类样本」与 `class_weight="balanced"` 对照，说明「近似等价但非恒等」；`partial_fit` 下 `balanced` 被禁（在 `SGDClassifier` 中实现，需实测确认报错文本） |
| **P10** | **没有 `predict_proba`：只有 `decision_function` 的符号 / argmax** | `LinearClassifierMixin`（`_base.py`）；`Perceptron` 未覆盖 | 感知机输出的是**未校准的分数**，不是概率；二分类 `decision_function` 是 1D，多分类是 2D，`predict` 走 `argmax`；这也是它与 logistic / `modified_huber`（唯一能给概率的 SGD 损失）在 API 层的分界线 | `hasattr(p,"predict_proba")` 为 False；digits 上 `decision_function(X).shape == (n_samples, 10)` |

#### 明确不做（写进「取舍」）

- 不展开 PassiveAggressive（PA-I/PA-II）与 `pa1`/`pa2` 学习率——虽然同基类，但属于另一估计器，用户已确认只讲 Perceptron。
- 不重复 hw4 已写的：`_sgd_fast` 32/64 dtype 分派、ASGD `average`、`make_dataset` 的 Dataset 抽象、SGD 的 `optimal/invscaling` 学习率表。
- 不做感知机收敛性证明、不做线性可分性定理。
- 不读/不引用 Cython 内循环源码（本地没有 `.pyx`）；所有内循环结论只用行为实验。
- 不新建 notebook / HTML，不改 README，不 commit。

### 版式要求（与 hw3/hw4 对齐）

- 开头用 `> 说明` 块写清适用范围与版本。
- 差异点用 **P1–P10 编号**，列表 + 表格混排；每条都要有「位置 / 为什么不一样 / 怎么实测」。
- 行号只用于本文档定位；明确注明「日后做 notebook 一律用 `inspect.getsource`，不写死行号」。
- 所有实测数值保留本轮的具体数字（样本数、轮数、`t_`、`coef_`、`intercept_`），便于复核。
- 语气保持 hw3/hw4 的「工程差异点」风格，不写成教程。

---

## Assumptions & Decisions

1. **交付形式**（用户已选）：只出方向清单，不产出 notebook / HTML。
2. **覆盖范围**（用户已选）：只讲 `Perceptron` 本身；PA、hinge-SVM 只作为 P2 的对照出现，不单列主线。
3. **版本**：vmunet 为 sklearn **1.3.2**，与 hw1–hw4 一致；本轮所有行号与结论均在该版本下验证。
4. **源码不可见部分**：`_sgd_fast` 本地只有 `.so`；`.pxd` 仅 26 行且不含 `Hinge` 声明。因此 P2/P3/P4/P6 的内循环结论**以行为实测为准**，文档中不得出现「源码第 N 行写了更新式」这类表述。
5. **与 hw4 的关系**：P8 与 hw4 R13 同源、P4 与 hw4 R10 部分重叠；文档必须显式写「去重说明」，避免同一结论讲两遍。
6. **不做代码改动**：本次唯一写入是方向清单 markdown。

---

## Verification

### 文档完成后逐项核对

1. 文件存在于 `.trae/documents/perceptron_code_analysis_plan.md`，且包含 Summary / Current State Analysis / P1–P10 / 去重 / 取舍 / 待确认 / Assumptions / Verification / Risks。
2. 每条 P# 都有：源码位置、与课堂/教科书的差异、可执行的实测方法；无「待补充」占位。
3. 文档中的数字与下列实测一致（可重跑复核）：

```bash
# 等价性 + t_ + 多分类
wsl -d Ubuntu-22.04 -- /root/MSA/ENTER/envs/vmunet/bin/python - <<'PY'
import numpy as np
from sklearn.linear_model import Perceptron, SGDClassifier
from sklearn.datasets import make_blobs, load_digits
X,y = make_blobs(n_samples=60, centers=2, random_state=0)
p = Perceptron(random_state=0).fit(X,y)
s = SGDClassifier(loss="perceptron", penalty=None, eta0=1.0,
                  learning_rate="constant", random_state=0).fit(X,y)
assert np.array_equal(p.coef_, s.coef_) and np.array_equal(p.intercept_, s.intercept_)
assert p.t_ == p.n_iter_ * X.shape[0] + 1 == 901
Xd, yd = load_digits(return_X_y=True)
m = Perceptron(random_state=0, max_iter=50).fit(Xd, yd)
assert m.coef_.shape == (10,64) and m.t_ == m.n_iter_*Xd.shape[0]+1
print("ok")
PY

# 教科书循环复现（shuffle=False, tol=None）
wsl -d Ubuntu-22.04 -- /root/MSA/ENTER/envs/vmunet/bin/python - <<'PY'
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.datasets import make_blobs
X,y = make_blobs(n_samples=200, centers=2, random_state=7, cluster_std=2.0)
y = np.where(y==0,-1,1)
w = np.zeros(X.shape[1]); b = 0.0
for _ in range(1000):
    for i in range(len(y)):
        if y[i]*(X[i]@w + b) <= 0:
            w += y[i]*X[i]; b += y[i]
sk = Perceptron(tol=None, shuffle=False, random_state=0, max_iter=1000).fit(X,y)
assert np.allclose(w, sk.coef_[0]) and np.allclose(b, sk.intercept_[0])
print("ok")
PY

# 稀疏截距衰减 + 警告行为 + 参数校验时机
wsl -d Ubuntu-22.04 -- /root/MSA/ENTER/envs/vmunet/bin/python - <<'PY'
import numpy as np, scipy.sparse as sp, warnings
from sklearn.linear_model import Perceptron
from sklearn.datasets import make_blobs
X,y = make_blobs(n_samples=100, centers=2, random_state=3, cluster_std=1.5)
kw = dict(shuffle=False, random_state=0, max_iter=30, tol=None)
d = Perceptron(**kw).fit(X,y); s = Perceptron(**kw).fit(sp.csr_matrix(X),y)
assert d.intercept_[0] == 3.0 and abs(s.intercept_[0]-0.14) < 1e-9
with warnings.catch_warnings(record=True) as w1:
    warnings.simplefilter("always")
    Perceptron(max_iter=3, tol=None, random_state=0).fit(X,y)
    assert not w1
with warnings.catch_warnings(record=True) as w2:
    warnings.simplefilter("always")
    Perceptron(max_iter=3, tol=1e-3, random_state=0).fit(X,y)
    assert any(issubclass(x.category, UserWarning) for x in w2)
for kw_, exc in [({"average":True}, TypeError), ({"loss":"hinge"}, TypeError), ({"alpha":-1.0}, Exception)]:
    try:
        Perceptron(**kw_); raise AssertionError("should raise")
    except exc:
        pass
print("ok")
PY
```

4. 文档中不出现「本地可读 `_sgd_fast.pyx`」之类的错误表述；`_sgd_fast.pxd` 的 26 行、不含 `Hinge` 的事实不能写反。
5. 明确标注「待确认的 notebook 结构」不执行；`README.md` 未改动；无新增 `ipynb` / `html`。

---

## Risks

- **版本/行号漂移**：行号只作定位提示；若 `vmunet` 升级，文档需重核 `_perceptron.py` 的 docstring 长度与 `_stochastic_gradient.py` 的 `loss_functions` 行号。
- **Cython 黑盒**：P2 的阈值语义、P6 的 loss 判定都靠实验反推；文档需明确「这是行为证据，不是源码引用」，避免过度断言（尤其决策值恰为 0 的边界）。
- **P3 的可复现条件很窄**：`shuffle=False` + `tol=None` + 稠密 + 无 penalty + 无 early_stopping；文档必须列出全部前置条件，否则读者换数据会得到不一致的结果。
- **P8 与 hw4 重复风险**：只保留「感知机上的后果」，机制一句带过并指向 hw4。
- **P9 的 `balanced` + `partial_fit` 报错文本未实测**：文档只写「需在 notebook 中实测确认」，不写死异常类型。
