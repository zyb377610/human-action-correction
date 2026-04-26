## Why

Phase 1 完成了姿态估计（33 关键点提取 + 特征工程），Phase 2 完成了标准动作数据层（预处理 + 模板库 + 增强）。目前系统可以采集和存储关键点序列，但缺少**将用户动作与标准模板进行比较**的核心能力。DTW 动作对比模块是矫正反馈（Phase 5）的前置依赖——只有知道"哪里偏差了、偏差多大"，才能生成有意义的矫正建议。

本阶段基于 DTW 综述论文实现经典 DTW / FastDTW / Derivative DTW 三种算法，支持 Sakoe-Chiba 带宽约束，输出相似度评分、最优对齐路径和逐关节偏差分析。预计工时：10-14 小时。

## What Changes

- 新增 `src/action_comparison/` 模块，包含 DTW 核心算法和对比流水线
- 实现三种 DTW 变体：经典 DTW (O(n²))、FastDTW（线性近似）、Derivative DTW（基于一阶导数的形状匹配）
- 支持三种距离度量：欧氏距离、余弦距离、曼哈顿距离
- 支持 Sakoe-Chiba 带宽约束以提升效率
- 输出结构化对比结果：相似度评分、最优对齐路径、逐帧距离矩阵
- 实现逐关节偏差分析——定位偏差最大的关节及其偏差程度
- 提供可视化工具：对齐路径图、逐帧距离热力图、偏差雷达图
- 提供端到端对比流水线：模板加载 → 预处理 → DTW 对比 → 偏差分析 → 输出报告
- 新增 `scripts/compare_action.py` 命令行对比脚本
- 新增 `tests/test_action_comparison.py` 单元测试

## Capabilities

### New Capabilities
- `dtw-comparison`: DTW 算法实现（经典/FastDTW/DDTW）、距离度量、带宽约束、相似度评分
- `joint-deviation-analysis`: 逐关节偏差检测与定位，识别偏差最大的关节并量化偏差程度
- `comparison-visualization`: 对比结果可视化（对齐路径图、距离热力图、偏差雷达图）

### Modified Capabilities
（无已有能力需修改）

## Impact

- **新增代码**: `src/action_comparison/` 整个模块（~6 个文件）
- **依赖新增**: `dtw-python`、`fastdtw` 库
- **上游依赖**: `src/pose_estimation/data_types.py`（PoseSequence）、`src/data/`（TemplateLibrary、预处理）
- **下游影响**: Phase 5（矫正反馈）将直接消费逐关节偏差分析结果
- **配置变更**: `configs/default.yaml` 中 `action_comparison` 段已预留，需要扩展
