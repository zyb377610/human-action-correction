## Why

姿态估计是整个人体动作矫正系统的基础模块和数据入口。系统需要从视频流（摄像头实时画面或视频文件）中提取人体骨骼关键点坐标序列，作为后续动作对比（DTW）、动作分类（深度学习模型）和矫正反馈的输入数据源。当前项目骨架已搭建完成，但 `src/pose_estimation/` 模块尚无实际代码实现。姿态估计模块是系统流水线的第一个环节，必须优先开发，后续所有模块均依赖其输出。

基于 MediaPipe 论文所述，MediaPipe 提供跨平台、高效的感知管道框架，其 Pose Landmark 模型可检测 33 个人体关键点，具有高实时性和无需额外硬件的优势（参见 ASD 运动分析综述中对 MediaPipe 的评价），是本项目的最优选择。

**预计开发时间：8-12 小时**

## What Changes

- **新增 `PoseEstimator` 核心类**：封装 MediaPipe Pose 模型，提供统一的姿态估计接口，支持单帧图像和视频流两种输入模式
- **新增关键点数据结构**：定义标准化的骨骼关键点数据格式（33 个关键点的坐标、置信度、可见度），支持序列化存储
- **新增特征工程模块**：基于原始关键点计算关节角度、骨骼长度比例、运动速度等衍生特征
- **新增可视化工具**：在视频帧上绘制骨骼连接图和关键点标注
- **新增视频输入管理**：支持摄像头实时流和视频文件两种数据源，统一接口

## Capabilities

### New Capabilities
- `pose-landmark-extraction`: MediaPipe 姿态估计核心能力——从图像/视频中提取 33 个人体骨骼关键点坐标与置信度
- `pose-feature-engineering`: 基于原始关键点的特征工程——计算关节角度、骨骼长度比、运动速度/加速度等衍生特征
- `pose-visualization`: 骨骼关键点可视化——在视频帧上绘制骨骼图、关键点标注、运动轨迹
- `video-input-manager`: 视频输入管理——统一摄像头实时流和视频文件的读取接口

### Modified Capabilities
<!-- 无已有 capability 需要修改 -->

## Impact

- **新增代码**：`src/pose_estimation/` 目录下新增多个 Python 模块
- **新增工具函数**：`src/utils/` 中新增配置加载和数据格式转换工具
- **依赖确认**：`mediapipe>=0.10.0`、`opencv-python>=4.8.0`、`numpy>=1.24.0`（均已在 requirements.txt 中）
- **配置扩展**：`configs/default.yaml` 中 `pose_estimation` 和 `camera` 配置段将被实际使用
- **数据格式**：定义标准化的关键点数据格式（JSON/NumPy），作为下游 DTW 对比模块和深度学习模块的输入契约
- **下游影响**：本模块的输出接口将被 `src/action_comparison/`（DTW 动作对比）和 `src/models/`（深度学习分类）直接消费
