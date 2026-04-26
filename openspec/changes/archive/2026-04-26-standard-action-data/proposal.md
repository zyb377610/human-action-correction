## Why

Phase 1 姿态估计模块已完成，能从视频中提取 33 个骨骼关键点和衍生特征。但后续的 DTW 动作对比（Phase 3）和深度学习分类（Phase 4）都需要标准动作数据作为参考基准和训练样本。没有标准数据，整个矫正系统无法运作。

本模块负责：录制标准动作模板、建立数据管理规范、实现预处理流水线。

**预计开发时间：6-8 小时**

## What Changes

- **新增标准动作录制工具**：基于 Phase 1 的 PoseEstimator 录制标准动作关键点序列
- **新增数据预处理流水线**：平滑滤波、缺失值插值、序列归一化
- **新增标准动作模板库管理**：动作分类定义、模板存储与加载
- **新增数据增强工具**：时间拉伸、噪声添加、镜像翻转

## Capabilities

### New Capabilities
- `action-recording`: 标准动作录制——通过摄像头录制标准动作并保存为 PoseSequence
- `data-preprocessing`: 数据预处理——平滑滤波、缺失值插值、序列长度归一化
- `template-library`: 标准模板库——动作类别管理、模板 CRUD、基于文件系统的存储
- `data-augmentation`: 数据增强——时间拉伸/压缩、高斯噪声、左右镜像翻转

### Modified Capabilities
<!-- 无 -->

## Impact

- **新增代码**：`src/data/` 目录下新增多个 Python 模块
- **新增目录**：`data/templates/` 存放标准动作模板
- **新增脚本**：`scripts/record_action.py` 录制工具
- **依赖 Phase 1**：使用 PoseEstimator、PoseSequence、FeatureExtractor
