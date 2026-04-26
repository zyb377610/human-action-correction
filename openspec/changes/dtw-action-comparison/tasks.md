## 1. 距离度量与基础设施

- [x] 1.1 创建 `src/action_comparison/__init__.py` 和 `src/action_comparison/distance_metrics.py` — 实现欧氏距离、余弦距离、曼哈顿距离函数，统一接口 `get_distance_func(name) -> Callable`
- [x] 1.2 实现 `sequence_to_feature_matrix()` — 将 PoseSequence 转换为 (T, 33*3) ndarray 特征矩阵

## 2. DTW 核心算法

- [x] 2.1 实现 `src/action_comparison/dtw_algorithms.py` — 经典 DTW 算法（累积代价矩阵 + 回溯最优路径），支持 Sakoe-Chiba 带宽约束
- [x] 2.2 实现 FastDTW — 基于 `fastdtw` 库的封装，统一输入输出接口
- [x] 2.3 实现 Derivative DTW — 对序列求一阶差分后调用经典 DTW，边界帧复制填充
- [x] 2.4 实现统一调度函数 `compute_dtw(query, template, algorithm, metric, window_size)` — 返回 `(distance, path, cost_matrix)`

## 3. 对比器与结果结构

- [x] 3.1 实现 `src/action_comparison/comparison.py` — 定义 `ComparisonResult` 数据类（distance, similarity, path, cost_matrix）
- [x] 3.2 实现 `ActionComparator` 类 — 封装完整对比流程：特征转换 → DTW 计算 → 相似度归一化 → 返回 ComparisonResult
- [x] 3.3 实现批量对比 `compare_with_templates()` — 将用户序列与模板库中所有模板对比，返回排序结果

## 4. 逐关节偏差分析

- [x] 4.1 实现 `src/action_comparison/deviation_analyzer.py` — 沿对齐路径计算每个关节的平均欧氏偏差
- [x] 4.2 实现 `DeviationReport` 数据类 — 包含 joint_deviations、worst_joints (top-K)、frame_deviations、severity 字段
- [x] 4.3 实现偏差严重程度分级逻辑 — 基于阈值划分轻微/中等/严重

## 5. 可视化与工具

- [x] 5.1 实现 `src/action_comparison/visualizer.py` — 对齐路径图（DTW warping path 叠加代价矩阵）
- [x] 5.2 实现距离热力图 — 逐帧代价矩阵的颜色编码热力图
- [x] 5.3 实现偏差雷达图 — 33 关节偏差的雷达/极坐标图，突出 top-K 偏差关节

## 6. 脚本、集成与测试

- [x] 6.1 编写 `scripts/compare_action.py` — 命令行对比脚本（输入用户序列文件 + 模板类别，输出评分和偏差报告）
- [x] 6.2 更新 `src/action_comparison/__init__.py` — 导出所有公共 API
- [x] 6.3 编写 `tests/test_action_comparison.py` — 单元测试覆盖三种 DTW 算法、距离度量、偏差分析、对比器
- [x] 6.4 安装新依赖 `fastdtw` 并更新 `requirements.txt`