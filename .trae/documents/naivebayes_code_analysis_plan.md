# 朴素贝叶斯源码解析 —— 「值得分享的硬核差异点」方向清单

## Summary

用户提问：**「朴素贝叶斯的代码有什么值得分享的」**。澄清结果：
- **交付形式**：先不要产出文件（notebook / HTML 以后再定）。
- **覆盖范围**：只挑硬核差异点（不铺开逐个变体、不写课堂上讲过的贝叶斯公式推导）。

因此本文件是**方向清单（本问题的直接回答）**，不是完整的 notebook 实施计划；notebook 结构放在末尾「待确认」小节。

产出对象：`/root/MSA/ENTER/envs/vmunet/lib/python3.8/site-packages/sklearn/naive_bayes.py`（sklearn 1.3.2，纯 Python，1530 行，无 `.so`，全部可读可实测）。

---

## Current State Analysis

### 工作区现状（`\\wsl.localhost\Ubuntu-22.04\root\Machine Learning`）
- 已有 `homework1-kmeans.ipynb/.html`、`homework2-KNN.ipynb/.html`、`fix_pagebreak.py`、`README.md`。
- 已完成两份分析计划：`.trae/documents/kmeans_code_match_plan.md`、`.trae/documents/knn_code_analysis_plan.md`。
- **作业 3（朴素贝叶斯）尚无任何文件**，也无相关记忆命中。

### 环境（已实测确认）
- 项目环境 `/root/MSA/ENTER/envs/vmunet`，`sklearn 1.3.2`，Python 3.8。
- `from sklearn import naive_bayes` 的源文件即上路径，**纯 Python**。
- 用户规则：一律用 vmunet 环境，不得用 base。
- 调用方式（Windows 侧需经 wsl）：
  `wsl -d Ubuntu-22.04 -- /root/MSA/ENTER/envs/vmunet/bin/python <script.py>`

### 可分析性
`naive_bayes.py` 全部是 Python，无编译模块；且 5 个分类器共享同一基类骨架，**可读性极好、实测成本低**。这与作业 2（KNN 大量 `.so` 只能黑盒）形成对比——本次可以真正 dump 到每一行。

### 源码骨架（行号基于 1.3.2）
```
_BaseNB(ABC)                          L39   —— 抽象 _joint_log_likelihood / _check_X
├── GaussianNB                        L147  —— 连续，直接挂 _BaseNB
└── _BaseDiscreteNB(ABC)              L518  —— 抽象 _count / _update_feature_log_prob
    ├── MultinomialNB                 L786
    ├── ComplementNB                  L912
    ├── BernoulliNB                   L1065
    └── CategoricalNB                 L1235
```

---

## 「值得分享的硬核差异点」（本问题的直接回答）

筛选标准：**课堂上不会讲、且是 sklearn 工程实现的真实决策**。按硬核程度排序，H1–H8。

### H1｜GaussianNB 的在线均值/方差合并——Chan-Golub-LeVeque 算法（最硬核）
- **位置**：`_update_mean_variance`（L271–343，`@staticmethod`）。
- **做法**：不保存任何历史样本，直接把「旧统计量 + 新一批数据」合并：
  `total_mu = (n_new·new_mu + n_past·mu) / n_total`（L333）、
  `total_ssd = old_ssd + new_ssd + (n_new·n_past/n_total)·(mu-new_mu)²`（L340）、
  `total_var = total_ssd / n_total`（L341）。
- **为什么值得说**：课堂上方差永远是「Σ(x−μ)²/n，一次性算完」；这里为了支持 `partial_fit`（out-of-core / 在线学习），用的是并行合并公式，第三项是关键修正项。还顺带处理了 `n_past==0`（首次）、`X.shape[0]==0`、`sample_weight.sum()≈0` 三个边界（L311–327）。
- **实测点**：分批 `partial_fit` 的 `theta_`/`var_` 与一次性 `fit` 完全一致（`np.allclose`）。

### H2｜`epsilon_` 的「先减后加」——partial_fit 里的平滑量防重复累加
- **位置**：`epsilon_` 定义于 L431；非首次调用入口 `self.var_[:, :] -= self.epsilon_`（L465）；循环结束 `self.var_[:, :] += self.epsilon_`（L497）。
- **做法**：`var_` 里存的是「已加过 epsilon」的方差；再次 `partial_fit` 时**先把它扣掉**，用干净的方差做 H1 的在线合并，合并完**再加回来**。
- **为什么值得说**：这是个极隐蔽的 bug 规避——若不先减，epsilon 会被卷进 CGL 组合公式，随每次 `partial_fit` 越滚越大。教科书（一次性算完）根本不会遇到这个问题。
- **实测点**：连续 `partial_fit` 3 次后，`var_` 不超过「一次性 fit + 单次 epsilon」的量级。

### H3｜BernoulliNB 的代数化简——消掉 `(1-X)` 稠密矩阵
- **位置**：`_joint_log_likelihood`（L1216–1232）。`neg_prob = log(1 - exp(feature_log_prob_))`（L1227）；
  `jll = safe_sparse_dot(X, (feature_log_prob_ - neg_prob).T); jll += class_log_prior_ + neg_prob.sum(axis=1)`（L1229–1230）。
- **做法**：原始公式 `Σ[x_i·log p_i + (1−x_i)·log(1−p_i)]` 直接写需要构造 `(1-X)`；用恒等式
  `neg_prob·(1−X) = Σneg_prob − X·neg_prob` 化简成**一次稀疏点积 + 一个常数项**。
- **为什么值得说**：一个纯代数恒等式换来「支持 CSR 稀疏输入」且「不物化 (1−X)」。这是把数学化简直接变成内存收益。
- **实测点**：手工公式 vs sklearn 的 `jll` 数值一致；稀疏 X 输入结果与稠密一致。

### H4｜ComplementNB 用一个负号把 argmin 折进基类的 argmax
- **位置**：`_update_feature_log_prob`（L1045–1055）与 `_joint_log_likelihood`（L1057–1062）；
  **源码注释直接点破**：`# _BaseNB.predict uses argmax, but ComplementNB operates with argmin.`（L1049）。
- **做法**：互补计数 `comp_count = feature_all_ + alpha − feature_count_`（L1047）；默认路径输出 `feature_log_prob = -logged`（L1054），于是 `argmax(-s) ≡ argmin(s)`，**复用基类写死的 `predict = classes_[argmax(jll)]`（L103）**。
- **为什么值得说**：算法上 Complement NB 是「最小化补集权重」，工程上靠一个负号就对上了基类的 argmax；`norm=True` 才符合 Rennie et al. 2003 论文 Table 9，默认 `norm=False` 是 Mahout/Weka 的口味（docstring L944–948 明说）。
- **实测点**：`norm` 开关对预测的影响；`ComplementNB` 在类别不平衡数据上的表现。

### H5｜模板方法：只实现一个 `_joint_log_likelihood`，白拿四个公开方法
- **位置**：`_BaseNB.predict_joint_log_proba / predict / predict_log_proba / predict_proba`（L62–144），唯一计算入口是抽象的 `_joint_log_likelihood`（L42–53）。
- **做法**：
  - `predict` = `classes_[argmax(jll, axis=1)]`（L103）——**归一化都不用做**，argmax 与常数无关；
  - `predict_log_proba` = `jll − logsumexp(jll, axis=1)`（L125–126）——用 **log-sum-exp** 在 log 域归一化，避免概率连乘下溢；
  - `predict_proba` = `exp(predict_log_proba)`（L144）；
  - `predict_joint_log_proba` 直接暴露未归一化的 jll（L62–84）。
- **为什么值得说**：朴素贝叶斯整族「算概率」被压缩成两个钩子函数（`_joint_log_likelihood` + `_check_X`）；三个公开方法的差别仅仅是「要不要归一化 / 在哪个域归一化」。logsumexp 是数值稳定性的标准答案。
- **实测点**：jll 很小（如 −1e3）时，`predict_proba` 仍不出现 0/0；`predict_proba` 行和为 1。

### H6｜sample_weight 被折叠进 one-hot 标签矩阵——`_count` 对权重完全无感
- **位置**：`fit` 的 `Y *= sample_weight.T`（L764）、`partial_fit` 的 L707，且为此把 Y 从 int64 转 float64（L757–761 / L701–703）。
- **做法**：把 `(n_samples, n_classes)` 的 one-hot 矩阵按样本乘上权重，之后 `_count` 里
  `feature_count_ += Y.T @ X`、`class_count_ += Y.sum(axis=0)` 自动得到**加权计数**，`_count` 内部一行权重判断都没有。
- **为什么值得说**：权重处理被提前折叠，5 个变体的 `_count` 得以保持干净。副作用也解释了两处看似奇怪的地方：① 为什么要转 float64；② `CategoricalNB._count` 里为何要判断 `Y.dtype.type == np.int64`（无权重才走 `np.bincount` 无 weights 的路径，L1492–1496）。
- 同一节的退化处理也值得说：标签二值化后若只有 1 列，两类时 `Y = concat(1−Y, Y)`，单类时 `Y = ones_like(Y)`（L751–755 / L690–695）。
- **实测点**：等价权重（如整数重复样本）与 sample_weight 结果一致；CategoricalNB 加权计数与整数化重复一致。

### H7｜CategoricalNB 的异构数据结构 + `np.pad` 动态增列
- **位置**：`_init_counters`（L1455–1457）`category_count_ = [np.zeros((n_classes, 0)) for _ in range(n_features)]`；
  动态增列 `_update_cat_count_dims` 用 `np.pad(cat_count, [(0,0),(0,diff)], "constant")`（L1482–1487）；
  按类掩码加权直方图 `np.bincount(X_feature[mask], weights=weights)`（L1496）；
  `_joint_log_likelihood` 用花式索引 `self.feature_log_prob_[i][:, indices].T`（L1526）。
- **为什么值得说**：前四个变体的 `feature_log_prob_` 都是规整的 `(n_classes, n_features)` 二维矩阵；**只有 CategoricalNB 是 list-of-arrays，且每张表的列数会随 `partial_fit` 见到的新类别动态增长**。这是「类别数事先未知、还要支持增量学习」的代价。
- **实测点**：`feature_log_prob_` 是 `list`、每个元素形状不同；分两次 `partial_fit`，第二次出现新类别后对应表列数变大。

### H8｜`_check_alpha` 的三层数值保护 + `force_alpha` 弃用迁移
- **位置**：`_check_alpha`（L604–639）。
- **做法**：① 支持 array alpha（按特征，需长度 == `n_features_in_`，L609–617）；② alpha 下限 `1e-10`，低于则 FutureWarning 并 clamp；③ `force_alpha` 默认 `"warn"` 这一过渡态，在 1.4 会变 `True`。
- **为什么值得说**：一个超参上叠了「数值保护 + 逐特征配置 + 弃用迁移」三层逻辑。`alpha=0`（不平滑）必须显式 `force_alpha=True` 才允许。
- **附带一处文档/代码不一致**：docstring 写 `force_alpha : bool, default=False`（如 L803），但签名默认值是 `"warn"`（如 L880）——可在 notebook 里作为「读源码要信签名而非 docstring」的例子。
- **实测点**：`alpha=0` 触发告警并 clamp；`force_alpha=True` 保持 0。

### 补充框架点（作 H1–H8 的骨架，不单独展开）
- 三级继承：`_BaseNB`(ABC) → `_BaseDiscreteNB`(ABC) → 四个离散变体；`GaussianNB` 直接挂 `_BaseNB`。
- 离散基类把 `fit`/`partial_fit` 的流水线「计数 → 平滑 → 先验」只写一遍：`_count` → `_check_alpha` → `_update_feature_log_prob` → `_update_class_log_prior`（L713–721 / L772–775）。
- `fit_prior=False` 时先验退化为均匀分布的 log 值 `-log(n_classes)`（L602）；某类未被观测时 `log(0)` 的 RuntimeWarning 被显式静默（L593–597）。
- `_more_tags`：`poor_score`（L782–783）、`requires_positive_X`（L890 / L1035 / L1437）。

### 明确「不做」
- 贝叶斯公式推导、条件独立假设、拉普拉斯平滑的教科书推导（课堂已讲）。
- 逐变体全量 dump 源码（稀释深度）；只 dump H1–H8 涉及的关键函数。
- 与作业 2 重叠的内容（`_parameter_constraints` 声明式校验、ABC 写法）只一句带过。

---

## 待确认：若后续要做成 `homework3-NaiveBayes.ipynb`

> 用户当前选择「先不要产出文件」，本节仅为后续备选，未获确认前不执行。

建议 10 节结构（markdown 提炼 → code 用 `inspect.getsource` 取源码 + 实测，不写死行号）：

| # | 节标题 | 覆盖 | code cell 做什么 |
|---|---|---|---|
| 1 | 朴素贝叶斯的两种「面孔」：课上 vs sklearn | — | 点明只需看「实现里不一样的地方」 |
| 2 | 骨架：三级继承 + 模板方法 | H5/框架 | 打印 `GaussianNB.__mro__`、`_BaseNB.__abstractmethods__`、`inspect.getsource(_BaseNB)` |
| 3 | 一个 `_joint_log_likelihood` 换四个公开方法 | H5 | dump 四个 predict 方法；实测 jll 极小值下 predict_proba 仍稳定、行和为 1 |
| 4 | 离散基类的流水线：计数→平滑→先验 | 框架 | dump `_BaseDiscreteNB.fit`/`_update_class_log_prior`；实测 `fit_prior=False` 的均匀先验 |
| 5 | 权重去哪儿了：折叠进 one-hot 矩阵 | H6 | dump L757–764；实测 sample_weight 与整数重复样本等价 |
| 6 | GaussianNB 的在线合并（CGL） | H1 | dump `_update_mean_variance`；实测分批 partial_fit == 一次性 fit |
| 7 | epsilon 的先减后加 | H2 | dump `_partial_fit` 关键行；实测多次 partial_fit 的 var_ 量级 |
| 8 | Bernoulli 的代数化简 | H3 | dump `_joint_log_likelihood`；手工公式对照 + 稀疏输入一致 |
| 9 | Complement 的负号把 argmin 折进 argmax | H4 | dump `_update_feature_log_prob`；实测 norm 开关、不平衡数据 |
| 10 | CategoricalNB 的异构结构与动态增列 | H7/H8 | dump `_init_counters`/`_count`；实测 list-of-arrays、`np.pad` 增列、`_check_alpha` 告警 |

后续若确认产出，沿用 README 工作流：
`nbconvert --to notebook --execute --inplace` → `nbconvert --to html --execute` → `fix_pagebreak.py`。

---

## Assumptions & Decisions

1. **筛选标准**：优先「课堂没讲透 / 没讲」的实现决策；贝叶斯公式本体不展开。
2. **交付形式**（用户已定）：**暂不产出文件**；本文档 = 方向清单。
3. **覆盖范围**（用户已定）：**只挑硬核差异点**，即 H1–H8，不逐个变体铺开。
4. **行号基于 sklearn 1.3.2**：行号只用于本文档定位；若日后做 notebook，一律用 `inspect.getsource(obj)` 取源码，避免版本漂移。
5. **Python 环境**：只用 `/root/MSA/ENTER/envs/vmunet`（用户规则）。
6. 编译模块问题本次不存在（`naive_bayes.py` 纯 Python），全部可实测。

---

## Verification

1. 逐一核对 H1–H8 引用的函数/行号与 1.3.2 源码一致（本文档已逐条对照）。
2. 若日后做 notebook：`nbconvert --execute` 全量 0 error；逐节核对「markdown 引用」与「code dump」一致（作业 1 暴露过的主要缺陷）。
3. 关键实测结论：H1（分批 == 一次性）、H2（var_ 不越界）、H3（手工公式一致 + 稀疏一致）、H4（负号等价 argmin）、H5（行和为 1）、H6（权重等价）、H7（list 结构 + 动态增列）、H8（alpha 告警）。

## Risks

- **H4/H7 的实测构造成本**：H4 需类别不平衡数据才看得出来，H7 需精心设计「第二次 partial_fit 出现新类别」的数据；若构造困难，退化为「代码精读 + 注释」。
- **H2 的量级断言易脆**：不要写死倍数，只做定性说明并固定 `random_state`。
- **篇幅**：H1–H8 已足够，避免为凑数把 H8 附带项（docstring 与签名不一致）当成独立大点。