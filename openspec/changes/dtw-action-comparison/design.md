## Context

Phase 1 和 Phase 2 已完成，系统能够：
- 通过 MediaPipe 提取 33 关键点序列（`PoseSequence`）
- 预处理关键点序列（插值、平滑、重采样到固定帧数）
- 管理标准动作模板库（`TemplateLibrary`）

现在需要实现核心对比逻辑：将用户动作序列与标准模板进行时序对齐和相似度计算。

### 数据流

```
用户动作 PoseSequence          标准模板 PoseSequence
         │                              │
         ▼                              ▼
   preprocess_pipeline()         preprocess_pipeline()
         │                              │
         ▼                              ▼
   to_feature_matrix()           to_feature_matrix()
   (T, 33*3) ndarray            (T', 33*3) ndarray
         │                              │
         └──────────┬───────────────────┘
                    ▼
          DTW 算法 (经典/FastDTW/DDTW)
                    │
                    ▼
         ComparisonResult
         ├── distance: float        (DTW 距离)
         ├── similarity: float      (归一化相似度 0~1)
         ├── path: List[(i,j)]      (最优对齐路径)
         └── cost_matrix: ndarray   (累积代价矩阵)
                    │
                    ▼
         逐关节偏差分析 (JointDeviationAnalyzer)
                    │
                    ▼
         DeviationReport
         ├── joint_deviations: Dict[str, float]  (各关节平均偏差)
         ├── worst_joints: List[str]              (偏差最大关节)
         ├── frame_deviations: ndarray            (逐帧偏差曲线)
         └── severity: str                        (轻微/中等/严重)
```

### 现有配置（`configs/default.yaml`）

```yaml
action_comparison:
  algorithm: "dtw"              # dtw / fastdtw / ddtw
  distance_metric: "euclidean"  # euclidean / cosine / manhattan
  window_size: null             # Sakoe-Chiba 带宽约束
  threshold: 0.3                # 动作相似度阈值
```

## Goals / Non-Goals

**Goals:**
- 实现经典 DTW、FastDTW、Derivative DTW 三种算法，统一接口
- 支持欧氏距离、余弦距离、曼哈顿距离三种度量
- 支持 Sakoe-Chiba 带宽约束以控制 DTW 搜索范围
- 输出结构化结果（距离、相似度、对齐路径、代价矩阵）
- 实现逐关节偏差分析——定位偏差最大的关节
- 提供可视化工具（对齐路径图、距离热力图、偏差雷达图）
- 提供端到端对比脚本

**Non-Goals:**
- 不实现在线实时流式 DTW（Phase 7 集成时考虑）
- 不实现 Weighted DTW / Segmented DTW（论文提及但对毕设非核心）
- 不做 GPU 加速（序列长度 ≤ 300 帧，CPU 即可满足实时性）

## Decisions

### 1. DTW 实现策略：混合方案

**决策**：经典 DTW 自实现 + FastDTW 用 `fastdtw` 库 + DDTW 自实现

**理由**：
- 经典 DTW 算法简单且核心（~50 行），自实现便于调试、理解和论文阐述
- FastDTW 库 `fastdtw` 已成熟且经过验证，无需重复实现
- DDTW 只是在经典 DTW 基础上对序列求一阶差分后再比较，在自实现 DTW 上简单扩展
- 备选方案 `dtw-python` 功能更全，但 API 偏重科研，接口较重；`fastdtw` 更轻量

### 2. 特征表示：关键点坐标向量

**决策**：将 PoseSequence 转换为 `(T, 33*3)` 的二维矩阵（33 关键点 × xyz 坐标）

**理由**：
- DTW 比较的是两个时间序列的**每一帧**之间的距离
- 使用原始 xyz 坐标而非角度/速度特征，因为偏差分析需要定位到具体关节的空间位置
- 角度/速度特征适合分类（Phase 4），但对"关节偏差多少度"的反馈更直观
- 备选方案：仅用角度特征 → 丢失空间位置信息，不利于矫正建议

### 3. 相似度归一化

**决策**：`similarity = 1.0 / (1.0 + normalized_distance)`，其中 `normalized_distance = dtw_distance / path_length`

**理由**：
- DTW 原始距离与序列长度相关，除以路径长度得到 per-step 平均距离
- Sigmoid 风格映射保证输出在 (0, 1] 范围，1 表示完全匹配
- 备选方案：指数衰减 `exp(-d)` → 对大偏差不够灵敏

### 4. 逐关节偏差分析

**决策**：沿对齐路径计算每个关节的平均欧氏偏差

**理由**：
- 对齐路径已解决时间错位问题，沿路径计算才有意义
- 对 33 个关节分别计算偏差，输出字典 `{joint_name: deviation}`
- 取 top-K 偏差最大关节（默认 K=5），作为矫正反馈的输入

### 5. 模块文件结构

```
src/action_comparison/
├── __init__.py              # 公共 API 导出
├── dtw_algorithms.py        # 三种 DTW 算法实现
├── distance_metrics.py      # 距离度量函数
├── comparison.py            # ActionComparator 对比器 + ComparisonResult
├── deviation_analyzer.py    # 逐关节偏差分析
└── visualizer.py            # 对比结果可视化
```

## Risks / Trade-offs

- **[性能] 经典 DTW O(n²)** → 对于 60 帧归一化序列，矩阵 60×60=3600 项，计算时间 < 50ms，可接受。长序列（>300 帧）建议用 FastDTW。
- **[精度] FastDTW 是近似算法** → 对于动作矫正场景，精度损失可忽略；论文中已证明线性复杂度下精度损失 < 5%。
- **[坐标系] MediaPipe 归一化坐标可能受体型影响** → 后续 Phase 5 可考虑身体比例归一化；当前阶段先用原始归一化坐标。
- **[DDTW 边界帧] 一阶差分会丢失首帧** → 用边界填充（首帧复制）保持序列长度不变。
