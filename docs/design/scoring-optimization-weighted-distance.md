# 评分机制优化文档 — 加权欧氏距离

> 日期：2026-05-06  
> 范围：动作对比评分模块（`src/action_comparison/`）  
> 影响：离线视频分析模式（实时模式不受影响，见末尾说明）

---

## 一、背景与问题

### 原有评分链路

```
用户序列 → 角度特征矩阵 (T, 9) → DTW(欧氏距离) → 相似度(高斯核映射)
                                            ↓
用户序列 → 坐标特征矩阵 (T, J*3) → 沿 path 逐帧逐关节欧氏距离 → 偏差报告
            ↑ 髋中心+躯干尺度+肩轴对齐
```

### 三个核心问题

| # | 问题 | 影响位置 | 严重度 |
|---|------|---------|--------|
| 1 | **z 轴噪声与 xy 等权**：MediaPipe 深度估计噪声约为 xy 的 3 倍，但评分中与 xy 同等对待 | DTW 相似度 + 偏差分析 | ⭐⭐⭐ |
| 2 | **关节等同看待**：17 个关节统一权重，但手腕自然活动范围远大于髋关节 | 偏差分析、矫正建议排序 | ⭐⭐⭐ |
| 3 | **坐标与角度独立**：两套特征各走各的，没有互补融合 | DTW 相似度 | ⭐⭐ |

---

## 二、修改方案

### 阶段 A：加权欧氏距离（核心，已实施）

#### A1. 轴权重

```python
AXIS_WEIGHTS = np.array([1.0, 1.0, 0.35], dtype=np.float64)
```

z 轴权重降至 0.35，深度噪声对评分的贡献从 ~33% 降到 ~11%。

#### A2. 关节重要性权重

基于 `CORE_JOINT_INDICES` 中的 MediaPipe 索引：

| 层级 | 关节 | 权重 | 理由 |
|------|------|:----:|------|
| 核心躯干 | 肩(11,12)、髋(23,24) | 1.2 | 偏差对动作质量影响最大 |
| 头部参考 | 鼻子(0) | 1.0 | 标准权重 |
| 中大关节 | 肘(13,14)、膝(25,26) | 1.0 | 标准权重 |
| 末端关节 | 腕(15,16)、踝(27,28) | 0.7 | 自然活动范围大 |
| 足部末端 | 脚跟(29,30)、脚尖(31,32) | 0.6 | 自然活动范围最大 |

#### A3. 可见度加权

低可见度关节放大偏差（`1/visibility`），倒逼用户改善拍摄条件，同时避免被遮挡关节的偏差被忽略。

#### A4. 加权距离公式

```
weighted_dev = raw_euclidean(xyz_diff ⊙ AXIS_WEIGHTS) × joint_weight × (1/visibility)
```

---

### 阶段 B：坐标+角度双通道 DTW（已实施）

新增 `metric="hybrid"` 选项，在 DTW 比对阶段融合坐标和角度特征：

```
hybrid_matrix = concat([coord_L2_norm × 0.6, angle_L2_norm × 0.4])
```

角度天然不受位置/体型/视角影响，坐标提供精确的空间定位——两者互补。

---

### 阶段 C：时域平滑（已实施）

在 `DeviationReport` 中新增 `temporal_volatility` 字段：

```python
temporal_volatility = mean(|diff(frame_deviations)|)
```

用于衡量动作稳定性——不仅惩罚"偏离模板"，也暴露"忽好忽坏"的不稳定表现。

---

## 三、文件变更清单

| 文件 | 变更类型 | 说明 |
|------|---------|------|
| `src/action_comparison/distance_metrics.py` | 修改 | 新增 `AXIS_WEIGHTS`、`JOINT_IMPORTANCE_WEIGHTS`、`weighted_euclidean_distance()`、`weighted_euclidean_frame()`、`sequence_to_landmark_matrix_weighted()`、`sequence_to_hybrid_matrix()` |
| `src/action_comparison/deviation_analyzer.py` | 修改 | `JointDeviationAnalyzer` 新增 `use_weighted` 参数（默认 `True`）；`DeviationReport` 新增 `temporal_volatility` 和 `use_weighted` 字段 |
| `src/action_comparison/comparison.py` | 修改 | `ActionComparator` 支持 `metric="hybrid"` 混合特征矩阵 |
| `src/action_comparison/__init__.py` | 修改 | 导出所有新增符号 |
| `configs/default_config.yaml` | 修改 | 新增 `axis_weights`、`use_weighted_distance`、`temporal_smoothness_enabled` |
| `tests/test_action_comparison.py` | 修改 | 新增 3 个测试类：`TestWeightedDistance`（7 项）、`TestHybridMatrix`（3 项）、`TestDeviationAnalyzerWeighted`（5 项）、`TestHybridComparator`（2 项） |
| `scripts/verify_weighted.py` | **新建** | 三种场景对比验证脚本（z 轴偏差 / 手腕偏差 / 核心关节偏差） |

---

## 四、验证结果

### 4.1 单元测试

```
tests/test_action_comparison.py — 47 passed ✅
```

包括全部原有测试 + 17 项新增测试，无回归。

### 4.2 场景对比验证

| 场景 | 未加权偏差 | 加权偏差 | 变化 | 结论 |
|------|----------|---------|------|------|
| A: z 轴偏差（前后晃动） | 0.1699 | 0.0519 | **↓69%** | z 降权生效 ✅ |
| B: 手腕偏差（末端关节） | 0.0392 | 0.0305 | **↓22%** | 末端降权生效 ✅ |
| C: 核心关节偏差（肩/髋） | 0.2124 | 0.1852 | ↓13% | 最差关节排名从脚跟→膝/肩，核心加权在单关节级放大生效 ✅ |

### 4.3 向后兼容

`JointDeviationAnalyzer(use_weighted=False)` 行为与旧版完全一致。设置为 `False` 即可回退原始欧氏距离。

---

## 五、使用方式

```python
# 加权偏差分析（新默认，推荐）
analyzer = JointDeviationAnalyzer(top_k=5, use_weighted=True)
report = analyzer.analyze(user_seq, template_seq, result)
# report.temporal_volatility  → 时域波动率
# report.use_weighted          → 是否使用加权

# 混合特征 DTW 比对（坐标+角度双通道）
comp = ActionComparator(metric="hybrid")
result = comp.compare(user_seq, template_seq)

# 回退原始行为
analyzer = JointDeviationAnalyzer(top_k=5, use_weighted=False)
```

---

## 六、离线/实时模式分析

### 离线模式（视频分析）

```
Pipeline: 角度(9维) → DTW评分 + 对齐路径
              ↓
          坐标(51维) → 加权逐关节偏差 ← 本次优化生效
              ↓
          角度(9维) → 角度偏差对比
```

**加权距离优化完全接入**，无需额外配置。

### 实时模式

```
RealtimeFeedbackEngine: 角度(9维) → 窗口DTW + 角度偏差
                        坐标维度 → 未实现
```

**不受本次优化影响**。原因：实时模式当前不使用坐标特征进行偏差分析，且窗口 DTW 的对齐路径被计算后丢弃、未用于帧级对齐。如需为实时模式添加空间坐标反馈，改动量约 30~40 行（局限在 `realtime_feedback.py` 单文件内），详见 `docs/design/roadmap.md`。

---

## 七、后续优化方向

| 优先级 | 内容 | 预估改动量 |
|:--:|------|:--:|
| P1 | 实时模式接入坐标维度 + DTW 路径复用 | ~40 行，单文件 |
| P2 | 运动链溯源偏差分析（区分根因偏差 vs 传导偏差） | ~80 行 |
| P3 | 基于多模板统计的逐关节"正常变异范围"学习 | ~120 行 |
| P4 | 时域平滑惩罚纳入最终评分（当前仅报告、不扣分） | ~30 行 |

---

## 八、关键设计决策

1. **加权欧氏距离不影响 DTW 相似度评分**。DTW 比对仍使用角度特征（或 hybrid），加权仅用于 Step 4 的空间偏差分析。两者解耦，互不干扰。

2. **use_weighted 默认 True**。`CorrectionPipeline` 和所有脚本在创建 `JointDeviationAnalyzer` 时未显式传参，自动获得加权行为。

3. **hybrid 度量是独立的 DTW 特征选择**，与加权偏差分析正交。hybrid 决定 DTW 用什么特征找对齐路径，加权决定沿路径怎么算偏差。

4. **z 轴权重 0.35 可调**。若拍摄场景深度信息可靠（如深度相机），可在 `configs/default_config.yaml` 中调整 `axis_weights`。
