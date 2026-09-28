# 第二次作业（KNN）—— sklearn 里「不太一样」的 KNN 实现方向 + notebook 实施计划

## Summary

用户反馈：`uniform` 多数投票 / `distance` 加权投票**课上已讲**，不是本作业的重点。因此本计划改以「**sklearn 实现里与教科书/课堂不同的地方**」为筛选标准：

- 课上内容（投票、加权投票、KNN 回归）→ notebook 里只用 1 个对照实验呼应，不展开推导。
- 重点放在**数学假设、数据结构、工程架构**三类「不一样」的实现决策（下节 D1–D9 全部经过源码核实）。

产出物（默认沿用作业 1 惯例，因提问被跳过）：
1. `homework2-KNN.ipynb`（新建，13 节）
2. `homework2-KNN.html`（nbconvert 导出，复用 `fix_pagebreak.py` 的分页 CSS）

---

## Current State Analysis

### 工作区现状（`\\wsl.localhost\Ubuntu-22.04\root\Machine Learning`）
- 已有 `homework1-kmeans.ipynb`（28 cells / 13 节）与 `homework1-kmeans.html`。
- 已有 `fix_pagebreak.py`：向 HTML `</head>` 前注入打印 CSS，把 `break-inside`/`page-break-inside` 覆盖为 `auto !important`。**作业 2 直接复用。**
- 作业 1 计划（`.trae/documents/kmeans_code_match_plan.md`）确立了格式：**markdown「核心提炼」→ code「源码/实测」交替，markdown 引用具体行号**。

### 运行环境（已确认）
- 项目环境 `/root/MSA/ENTER/envs/vmunet`（Python 3.8），`sklearn 1.3.2`。
- sklearn 路径：`/root/MSA/ENTER/envs/vmunet/lib/python3.8/site-packages/sklearn/`
- **必须用 vmunet 环境，不得用 base/系统 python3**（用户规则）。

### 可分析性约束（关键）
`neighbors/` 下**纯 Python 可读**：`_base.py`(1359)、`_classification.py`(788)、`_regression.py`(505)、`_graph.py`(678)、`_unsupervised.py`(175)、`_lof.py`(516)、`_nca.py`(525)、`_kde.py`(365)、`_nearest_centroid.py`(261)。
**只有 `.so`**：`_dist_metrics`、`_kd_tree`、`_ball_tree`、`_partition_nodes`、`_quad_tree`（且 `_dist_metrics.pyx` 未随包发布）。→ 编译模块一律走「接口 / `valid_metrics` / docstring / 调用点」黑盒分析。

---

## 「不太一样的」方向分析（本节是用户问题的直接回答）

按「与课堂重复度低 + 源码里确有其事 + 可实测」筛选，分三类。

### A. 数学层面「不太一样」

| 编号 | 方向 | 源码位置（`_base.py` 等） | 为什么不一样 |
|---|---|---|---|
| **D1** | **平方欧氏 + 延迟开方** | `kneighbors` brute 路径传 `squared=True`（L854–855）；`_kneighbors_reduce_func` 仅当 `effective_metric_=="euclidean"` 才 `sqrt`（L722–723） | 排序只用距离的**单调变换**，全程不产生真实距离；半径版更要 `radius *= radius` 才能与平方距离比较（L1186–1189）。课堂只算距离，不提这套等价变换 |
| **D2** | **`effective_metric_` 特化坍缩** | `_fit`（L512–525） | `minkowski` 且 `p=1/2/∞` 且无权重时，**不调用通用 minkowski**，而是坍缩成 `manhattan`/`euclidean`/`chebyshev` 走更快实现；有 `w` 才保留通用分支 |
| **D3** | **`0<p<1` 是半度量** | `_fit`（L621–641） | 不满足三角不等式 ⇒ KDTree/BallTree 数学上不可用，**强制 brute**（`p<1` 且指定树就报错）。把「度量公理」直接写进工程约束 |
| **D4** | **`weights='distance'` 的零距离语义** | `_get_weights`（L103–122） | 不是 `1/(d+ε)`，而是**同一位置的点权重置 1、其余置 0**；另有一条 `object dtype` 分支专门接半径分类器的 outlier 哨兵值 |
| **D5** | **半径结果用稳定排序** | `radius_neighbors`（L1220） | `np.argsort(kind="mergesort")` 保证等距邻居顺序确定，固定 k 版则不做此保证 |

### B. 数据结构 / 算法层面「不太一样」

| 编号 | 方向 | 源码位置 | 为什么不一样 |
|---|---|---|---|
| **D6** | **Top-k 用 `argpartition` 而非排序** | `_kneighbors_reduce_func`（L716–720） | O(n) 选出第 k 小、只对 k 个做小排序；`argpartition` 不保证有序，**必须再 argsort 一次**，否则 bug |
| **D7** | **`algorithm="auto"` 的硬编码启发式** | `_fit`（L586–619）、`_check_algorithm_metric`（L413–452） | 规则是写死的：`n_features>15` ⇒ brute；`n_neighbors ≥ n_samples//2` ⇒ brute；带权 minkowski ⇒ **ball_tree 而非 kd_tree**；metric 不在 `KDTree.valid_metrics` 则降级 |
| **D8** | **稀疏/预计算图为输入（KNN 可以「没有坐标」）** | `_base.py` `_check_precomputed`(L155)、`_is_sorted_by_data`(L128)、`sort_graph_by_row_values`(L201)、`_kneighbors_from_graph`(L278)；`_graph.py` L38/L130 | 把稀疏距离矩阵当图，**只有非零元才是候选邻居**；`extract` 用 `np.take(..., mode="clip")` 把 ragged CSR 压成规则 `(n,k)` 块（L315–322），是课堂不会讲的黑魔法 |
| **D9** | **变长邻居 = object ndarray** | `_radius_neighbors_from_graph`（L369–370）、`radius_neighbors`（L1211–1216）、`_radius_neighbors_reduce_func`（L1046） | 半径版邻居数不固定，只能 `np.split` + `_to_object_array` 造 ragged 数组，和固定 k 的二维块是两套实现 |

### C. 架构 / 工程层面「不太一样」

| 编号 | 方向 | 源码位置 | 为什么不一样 |
|---|---|---|---|
| **D10** | **惰性学习的架构后果：索引可复用** | `_fit`（L527–546） | `fit` 不训练，`_fit_X` 就是模型；`X` 若已是 `NeighborsBase`/`KDTree`/`BallTree`，**直接复用其 `_tree`**，等于「预建索引 + 多模型共享」 |
| **D11** | **`NeighborsBase` 是 ABC，且 `__init__` 被标 `@abstractmethod`** | L378、L392 | 抽象构造器这种写法很少见，强制子类自定义 `__init__`；参数校验走声明式 `_parameter_constraints`（`Interval`/`StrOptions`/`callable`，L381–390） |
| **D12** | **四条查询路径 + `PairwiseDistancesReductions`** | `kneighbors`（L815–882） | ①`ArgKmin` 快速路径 ②稀疏预计算图 ③`pairwise_distances_chunked`+joblib ④树的线程分片。快速路径的意义是**不物化 n×m 距离矩阵**，冲突点：`strategy="parallel_on_X"` 是实测经验硬编码（`_classification.py` L341–349） |
| **D13** | **并行策略因后端而异** | `kneighbors`（L870–882）、`radius_neighbors`（L1225–1245） | 树查询用 `Parallel(prefer="threads")`（吃 Cython 释放 GIL 的红利），brute 用 joblib 进程；同一参数 `n_jobs` 语义完全不同 |
| **D14** | **query==train 的自身排除 + 重复点边界** | `kneighbors`（L893–921） | 查询即训练时先多取 1 个再掩掉自己；但**当重复点比 k 还多**时首个近邻不是自己而是重复点，代码专门 mask 掉第一个重复点。这是纯工程 corner case |
| **D15** | **outlier 哨兵与 -1 标签** | `_classification.py` `RadiusNeighborsClassifier.predict`（L689–692）、`predict_proba`（L718–722） | 用「概率全为 0」判定无邻居，赋 `outlier_label_`；与 D4 的 object 分支、`radius_neighbors_graph` 前的 `1e-6` 哨兵拧在一起 |
| **D16** | **`KNeighborsTransformer` 的 ±1 邻居** | `_graph.py` `transform`（L415–418） | `add_one = (mode=="distance")`：distance 模式要多取 1 个邻居，因为对角线（自身，距离 0）总被显式写回 |

### 延伸方向（第三梯队，选做 1 个）
- **LOF**（`_lof.py` L20/L231/L439/L486）：k-距离 / 可达距离 / 局部可达密度——KNN 邻域思想的完整演绎，可视化最直观。**默认选它。**
- **NCA**（`_nca.py` L34/L226/L450）：让 KNN「学一个距离」，含 softmax 近邻概率与 L-BFGS 梯度，理论最深。
- **KDE**（`_kde.py` L35/L191/L245）：核密度，与 KNN 同源但连续化。

### 取舍
**做** D1–D16（即第一/二/三梯队的 A/B/C 三类），**选做** LOF。**不做**：`.pyx` 源码 dump（不可行）、neighbors 包全景铺开（稀释深度）、投票/加权投票的公式推导（课上已讲）。

---

## Proposed Changes

### 新建文件 1：`homework2-KNN.ipynb`（13 节，对齐作业 1 体量）

| # | 节标题 | 覆盖方向 | markdown 核心提炼 | code cell 做什么 |
|---|---|---|---|---|
| 1 | KNN 的两种「面孔」：课上 vs sklearn | — | 半页点明：投票/加权课上已讲；本作业只看实现里不一样的地方 | — |
| 2 | 定位源码：neighbors 包的可分析边界 | — | `.py` 可读 / `.so` 不可读 的清单与原因 | 打印 `sklearn.__file__`；列出 `neighbors/` 目录并分类 `.py`/`.so` |
| 3 | 惰性学习与索引复用 | D10 | `fit` 只校验+建索引；模型即训练集；索引可跨模型复用 | `inspect.getsource(NeighborsBase._fit)`；实测把已 fit 的 `NearestNeighbors` 直接当 `X` 传给另一个模型，验证 `_tree` 被复用 |
| 4 | 抽象基类与 Mixin 组合 | D11 | ABC + `@abstractmethod __init__`；声明式参数校验 | 打印 `KNeighborsClassifier.__mro__`；`print(NeighborsBase.__abstractmethods__)`；`print(KNeighborsClassifier._parameter_constraints)` |
| 5 | 算法选择启发式 | D7 | auto 的硬编码规则 + `KDTree`/`BallTree` 的 `valid_metrics` 差异 | `inspect.getsource(NeighborsBase._fit)`（切片只留 L582–641）；`print(VALID_METRICS['kd_tree'])` vs `['ball_tree']`；实测改维度/`n_neighbors` 看 `_fit_method` 翻转 |
| 6 | 距离的等价变换与度量特化 | D1/D2/D3 | 平方欧氏+延迟开方；`effective_metric_` 坍缩；`p<1` 半度量 | `print(VALID_METRICS...)`；实测 `p=1/2/3/0.5` 时 `effective_metric_` 的值；复现 `p=0.5` + `algorithm='kd_tree'` 的报错 |
| 7 | `kneighbors` 内核：四路径与 Top-k | D6/D12/D13/D14 | 四条执行路径；argpartition 的「无序再排序」；自身排除/重复点边界 | `inspect.getsource(KNeighborsMixin.kneighbors)`（切片留函数体）；`inspect.getsource(_kneighbors_reduce_func)`；构造重复点数据实测 `query==train` 的排除行为 |
| 8 | 为什么不物化 n×m 距离矩阵 | D12 | `PairwiseDistancesReductions`/`ArgKmin` 的内存动机（黑盒）；`strategy` 经验选择 | 计时/内存对比：brute 大样本下 `ArgKmin` 路径 vs 强制 callable metric 的 joblib 路径，打印耗时 |
| 9 | 权重语义与半径版的变长邻居 | D4/D5/D9 | 零距离权重语义；object 哨兵分支；ragged 数组；稳定排序 | `inspect.getsource(_get_weights)`；`inspect.getsource(RadiusNeighborsMixin.radius_neighbors)`；实测含重复点数据下 `uniform` vs `distance` 的预测差异；实测 `radius_neighbors` 返回 `dtype=object` |
| 10 | KNN 可以「没有坐标」：预计算稀疏图 | D8/D16 | 稀疏距离矩阵即图，非零元=候选邻居；ragged→规则块的压平技巧；Transformer 的 ±1 | `inspect.getsource(_kneighbors_from_graph)`；构造稀疏图喂给 `metric='precomputed'` 并验证结果；`kneighbors_graph` 输出 `toarray()` |
| 11 | 分类/回归聚合：仅一个呼应实验 | D15(+课上) | 一句话承认课上已讲；只说 outlier 的 `-1` 与零概率哨兵 | `RadiusNeighborsClassifier` 对孤立点返回 `-1` 的实测；`predict_proba` 行和为 1 |
| 12 | 「不太一样」实验集 + LOF 延伸 | D1–D16 + LOF | 汇总对比 | ①三后端预测一致性；②后端耗时随维度翻转；③含离群点数据上 LOF `fit_predict` 可视化 |
| 13 | 小结 | — | 把「不一样」串成一条线：等价变换→启发式→路径分派→哨兵/边界 | — |

代码要求：
- 一律用 `inspect.getsource(...)` 取源码（**不写死行号**），并在 print 前加注释标出 markdown 引用的关键行段。
- 长函数（`kneighbors` ~190 行含 docstring、`_fit` ~220 行）**切片只打印函数体**或按逻辑分段打印。
- 图表标题/图例用纯 ASCII（避免 CJK glyph missing），沿用作业 1 经验。
- 不引入新依赖，全部在 `vmunet` 下可执行。

### 新建文件 2：`homework2-KNN.html`
- 在 vmunet 环境执行 `jupyter nbconvert --to html --execute homework2-KNN.ipynb`，再运行 `fix_pagebreak.py`（幂等）。

---

## Assumptions & Decisions

1. **筛选标准**（本次核心修订）：优先「课堂没讲透/没讲」的实现细节；投票与加权投票**降级为第 11 节的一个对照实验**，不写公式推导。
2. **交付形式**：沿用作业 1（ipynb + HTML + 分页 CSS）；提问被跳过后按此默认执行，若只需 notebook 则跳过文件 2。
3. **范围**：D1–D16 全覆盖，延伸算法只放 LOF（D10 之外的 `_lof.py`）。
4. **编译模块**：`_dist_metrics`/`_kd_tree`/`_ball_tree` 等只有 `.so`，一律黑盒（`valid_metrics`、docstring、调用点）。
5. **Python 环境**：只用 `/root/MSA/ENTER/envs/vmunet`（用户规则）。
6. **行号基于 sklearn 1.3.2**：行号只写在 markdown 说明里，code 用对象取源码，避免版本漂移失效。

---

## Verification

1. notebook JSON 合法，`nbconvert --execute` 全量跑通，无 Error/Traceback。
2. **逐节核对「markdown 引用的函数/行号」与「code 实际 dump 的内容」一致**（作业 1 暴露过的主要缺陷，重点自查）。
3. 关键实测输出存在且正确：
   - 第 3 节：索引复用后 `_tree is` 原对象为 True；
   - 第 5 节：维度 → `_fit_method` 翻转符合 L586–619 规则；
   - 第 6 节：`p=0.5` + 树 报 ValueError；`effective_metric_` 坍缩到 `manhattan/euclidean/chebyshev`；
   - 第 7 节：重复点数据下 `query==train` 成功排除自身、输出 shape 正确；
   - 第 9 节：`radius_neighbors` 返回 `dtype=object`；
   - 第 10 节：稀疏图 `precomputed` 结果与直接 `kneighbors` 一致；
   - 第 12 节：三后端预测完全一致。
4. 导出 HTML 后确认 `break-inside: auto !important` 仍在，打印预览长代码自然截断。

## Risks

- **第 8 节的耗时对比可能不稳定**（机器负载、Cython 快速路径在小数据下不一定更快）。对策：用足够大的样本量并固定 `random_state`，结果只做定性说明，不写死倍数。
- **`ArgKmin`/`RadiusNeighbors`/`ArgKminClassMode` 在编译模块**：只在 markdown 讲角色与启用条件，code 仅展示「何时进入该分支」，不 dump 源码。
- **D14 的重复点边界**（L908–913）逻辑绕，实测要精心构造：`n_neighbors` 小于重复点数时才会触发。若构造困难，退化为「代码精读 + 注释」。
- **篇幅可能超作业 1**：D1–D16 共 16 个点，若过长，优先砍 C 类里最弱的 D16（Transformer ±1）与第 8 节实验，保留 A/B 两类的数学与数据结构硬货。