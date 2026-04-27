## ADDED Requirements

### Requirement: 单帧姿态估计
系统 SHALL 能够从单张 RGB 图像中提取 33 个人体骨骼关键点坐标。每个关键点 SHALL 包含归一化坐标 (x, y, z) 和可见度 (visibility) 四个属性值。

#### Scenario: 正常单帧检测
- **WHEN** 输入一张包含完整人体的 RGB 图像
- **THEN** 返回 PoseFrame 对象，包含 33 个关键点的坐标和可见度信息，所有坐标值在 [0, 1] 范围内

#### Scenario: 无人体图像处理
- **WHEN** 输入一张不包含人体的图像
- **THEN** 返回 None，不抛出异常

#### Scenario: 部分遮挡处理
- **WHEN** 输入一张人体部分被遮挡的图像
- **THEN** 返回 PoseFrame 对象，被遮挡关键点的 visibility 值 SHALL 低于 0.5

### Requirement: 视频流连续姿态估计
系统 SHALL 能够对视频流进行连续帧的姿态估计，输出按时间戳排序的关键点序列。系统 SHALL 利用 MediaPipe 的帧间追踪能力以提升连续帧的稳定性。

#### Scenario: 摄像头实时流处理
- **WHEN** 以摄像头作为视频源启动姿态估计
- **THEN** 系统 SHALL 逐帧提取关键点并输出 PoseFrame 序列，帧率不低于 20FPS（CPU 模式下）

#### Scenario: 视频文件批量处理
- **WHEN** 以视频文件作为输入源
- **THEN** 系统 SHALL 处理视频所有帧并输出 PoseSequence 对象，包含完整的时间戳和帧序号信息

#### Scenario: 处理结束后资源释放
- **WHEN** 视频流处理完成或被中断
- **THEN** 系统 SHALL 释放 MediaPipe 模型和 OpenCV 资源，不产生内存泄漏

### Requirement: 配置化模型参数
系统 SHALL 通过 YAML 配置文件控制 MediaPipe Pose 模型的参数，包括模型复杂度、最小检测置信度和最小追踪置信度。

#### Scenario: 使用默认配置初始化
- **WHEN** 未指定特定配置参数创建 PoseEstimator
- **THEN** 使用 configs/default.yaml 中的默认值（model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5）

#### Scenario: 自定义配置初始化
- **WHEN** 通过参数或配置文件指定 model_complexity=2
- **THEN** PoseEstimator SHALL 使用 Heavy 模型，精度更高但速度更慢

### Requirement: 关键点数据序列化
系统 SHALL 支持将 PoseSequence 数据保存为文件和从文件加载，支持 JSON 和 NumPy (.npy) 两种格式。

#### Scenario: 保存为 NumPy 格式
- **WHEN** 调用 PoseSequence.save() 并指定 .npy 扩展名
- **THEN** 系统 SHALL 将关键点数据保存为形状为 (T, 33, 4) 的 NumPy 数组文件

#### Scenario: 保存为 JSON 格式
- **WHEN** 调用 PoseSequence.save() 并指定 .json 扩展名
- **THEN** 系统 SHALL 将完整的关键点数据（含元信息如 fps、帧数）保存为 JSON 文件

#### Scenario: 从文件加载数据
- **WHEN** 调用 PoseSequence.load() 加载已保存的文件
- **THEN** 加载后的 PoseSequence SHALL 与保存前数据一致（数值精度误差 < 1e-6）
