# pose-visualization Specification

## Purpose
TBD - created by archiving change pose-estimation-module. Update Purpose after archive.
## Requirements
### Requirement: 骨骼连接图绘制
系统 SHALL 能够在视频帧上绘制人体骨骼连接图，包括关键点标记和骨骼连线。

#### Scenario: 在图像上绘制完整骨骼图
- **WHEN** 输入一个 RGB 图像和对应的 PoseFrame
- **THEN** 在图像上绘制 33 个关键点（圆点）和标准骨骼连线，返回标注后的图像

#### Scenario: 可见度过滤绘制
- **WHEN** 输入包含低可见度关键点的 PoseFrame
- **THEN** visibility < 0.5 的关键点和相关连线 SHALL 使用半透明或不同颜色标识

#### Scenario: 自定义绘制样式
- **WHEN** 指定关键点颜色、大小和连线粗细参数
- **THEN** 系统 SHALL 按照指定样式绘制骨骼图

### Requirement: 关节角度标注
系统 SHALL 能够在骨骼图上标注关节角度数值。

#### Scenario: 标注核心关节角度
- **WHEN** 提供 PoseFrame 和对应的关节角度字典
- **THEN** 在对应关节位置显示角度值文本标注

### Requirement: 实时可视化窗口
系统 SHALL 提供基于 OpenCV 的实时可视化窗口，支持显示带骨骼标注的视频流。

#### Scenario: 实时显示骨骼标注视频
- **WHEN** 以摄像头为视频源运行姿态估计
- **THEN** 系统 SHALL 打开一个窗口实时显示带骨骼标注的视频画面，并在窗口标题栏显示当前 FPS

#### Scenario: 按键退出
- **WHEN** 用户按下 'q' 键或 ESC 键
- **THEN** 系统 SHALL 关闭可视化窗口并释放所有资源

