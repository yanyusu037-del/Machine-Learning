# KMeans Notebook 代码与核心提炼对齐 修改计划

## Repository Research

当前 `homework1-kmeans.ipynb` 共 28 个 cell，结构为「markdown 核心提炼 → code 展示源码/示例」交替。逐节比对后发现 4 处明显的「代码展示」与「核心提炼」不匹配：

### 1. Section 3（参数与属性）—— code 不验证属性
- **markdown** 列出 `cluster_centers_` / `labels_` / `inertia_` / `n_iter_` 等关键属性及含义。
- **code（cell 5）** 只 `print(inspect.signature(...))` 和 `_parameter_constraints`，**没有 fit 后打印这些属性**，读者看不到属性真实存在与形状。

### 2. Section 4（fit 主流程）—— code 截断严重
- **markdown** 详细拆解 fit 的 5 阶段，引用行号 L1446–L1573（共 129 行 / 4650 字符）。
- **code（cell 7）** 用 `print(src[:1000])` 只显示前 1000 字符（约 25 行），**完全覆盖不到后面 4 个阶段**。读者无法对照 markdown 看到第 3~5 阶段的真实代码。

### 3. Section 9（predict/transform/score）—— code 只展示了 predict 的底层 helper
- **markdown** 分别讲了 `predict`（50 行）、`transform`（22 行）、`score`（30 行）三个方法。
- **code（cell 18）** 只 `print(inspect.getsource(_labels_inertia))`（69 行的 E 步 helper），**transform 和 score 的源码完全没展示**，predict 本身也没展示，只展示了其内部调用的 `_labels_inertia`。

### 4. Section 11（端到端例子）—— 3 个 code cell 只有 1 行引导
- **markdown（cell 21）** 只有一句话：「对比 lloyd 与 elkan 两种算法的输出一致性，并打印收敛信息。」
- **实际有 3 个 code cell**：cell 22（对比 inertia/V-measure）、cell 23（可视化聚类结果）、cell 24（质心轨迹图）。后两个尤其是轨迹图**没有任何 markdown 说明其目的、做法、读图方式**，属于「代码孤儿」。

### 5. 共性问题
- 多个 code cell 用 `inspect.getsource(...)` 直接 dump 原始源码，未在代码内用注释标注 markdown 所引用的关键行/关键变量（如 `best_inertia` 的择优条件、`_tolerance` 的量纲处理），读者需自行在 dump 里找。

## Files and Modules

- `homework1-kmeans.ipynb`：修改 4 个节的 code/markdown cell，不新增 cell、不改章节号、不动其他节。

## Implementation Steps

### Step 1：Section 3（cell 5）—— 补属性验证
在 cell 5 现有代码后追加：实例化并 `fit` 一个小 KMeans，打印 `cluster_centers_.shape`、`labels_.shape`、`inertia_`、`n_iter_`、`n_features_in_`，与 markdown 列的属性一一对应。

### Step 2：Section 4（cell 7）—— 展示完整 fit 源码
把 `print(src[:1000])` 改为 `print(src)`（129 行可完整显示），并在 cell 顶部加注释标出 5 个阶段对应的行段（与 markdown 引用的 L1446–L1573 对应），方便对照。

### Step 3：Section 9（cell 18）—— 补齐 predict/transform/score
把 cell 18 改为分别 `print(inspect.getsource(KMeans.predict))`、`...transform`、`...score`，并在末尾加一个小 demo：对 fit 好的模型调用 `predict(X[:5])`、`transform(X[:3]).shape`、`score(X[:10])`，输出对应结果，与 markdown 三方法一一对应。（`_labels_inertia` 作为 predict 的底层 helper 保留一行引用说明即可。）

### Step 4：Section 11（cell 21 markdown）—— 补全引导说明
把 cell 21 的单行 markdown 扩写为 3 小段，分别说明：
1. cell 22：Lloyd vs Elkan 的 inertia / n_iter / V-measure 一致性对比；
2. cell 23：聚类结果可视化（散点 + 质心）；
3. cell 24：质心轨迹图（逐轮 `max_iter=1` fit 记录质心迁移，对比两算法轨迹是否一致）。

### Step 5：微调共性（注释标注）
在 cell 12（Lloyd）、cell 16（择优）的源码打印前加简短注释，点出 markdown 提到的关键行（如 `center_shift_tot`、`_is_same_clustering` 条件），帮助读者在 dump 中定位。不改源码本身。

## Dependencies and Considerations

- 不改变章节序号、不新增/删除 cell（只改现有 cell 内容），保持 28 格结构。
- 所有新增代码必须在 `vmunet`（sklearn 1.3.2）下可执行，不引入新依赖。
- matplotlib 图标题/图例继续用纯 ASCII（避免 CJK glyph missing）。
- 修改后需 `nbconvert --execute` 全量跑通，并重新生成 `homework1-kmeans.html`（保留之前的 page-break override CSS）。

## Validation

1. 用脚本读取 notebook，确认 4 个目标 cell 的源码已更新、JSON 合法、cell 数仍为 28。
2. `jupyter nbconvert --to html --execute` 全量执行成功，无 Error/Exception。
3. 人工核对：cell 5 输出含属性；cell 7 输出为完整 129 行 fit；cell 18 含 predict/transform/score 三段源码 + demo 输出；cell 21 markdown 含三段引导。
4. 重新生成 HTML 并确认 `break-inside: auto !important` 覆盖仍在。

## Risks

- **fit 源码 129 行直接 print 可能偏长**：但 markdown 已引用具体行号，完整展示更利于对照；接受此长度。若过长可改为分段 print（按 5 阶段切片），计划中先按完整展示执行，必要时回退为分段。
- **cell 18 增加 3 段源码 + demo 后输出变长**：可接受，因为 markdown 明确讨论了三方法。
- **轨迹图 cell 无 markdown 引导的根因**是该 cell 是后加的，Step 4 仅补引导、不移动 cell 位置，保持原顺序。

# ========== 追加到文件末尾 ==========
c = get_config()

# 使用webpdf（浏览器渲染PDF，推荐，比latex稳定）
c.WebPDFExporter.enabled = True

# 注入全局CSS，输出框超出高度自动截断，不撑出空白
c.WebPDFExporter.extra_template_based_variables = {
    "extra_css": """
    /* 输出单元格：超出页面高度直接截断，禁止无限撑开 */
    .jp-OutputArea-output pre,
    .jp-OutputArea-output {
        max-height: 22cm;   /* 控制最大高度，A4一页高度，按需调小 18~24cm */
        overflow: hidden !important;  /* 直接截断，不会滚动、不会留下大片空白 */
        page-break-inside: avoid;
    }
    /* 代码输入块尽量不分页（你上一轮需求） */
    .jp-Cell-inputWrapper {
        page-break-inside: avoid !important;
    }
    /* 清除多余margin空白 */
    .jp-Cell {
        margin-bottom: 0.4rem;
    }
    """
}