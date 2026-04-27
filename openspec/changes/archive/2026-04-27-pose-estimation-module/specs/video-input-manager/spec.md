## ADDED Requirements

### Requirement: 统一视频源接口
系统 SHALL 提供统一的 VideoSource 抽象接口，屏蔽摄像头和视频文件之间的差异。所有视频源实现 SHALL 提供 read()、release() 和 fps 属性。

#### Scenario: 创建摄像头视频源
- **WHEN** 指定设备 ID（如 0）创建 CameraSource
- **THEN** 系统 SHALL 打开对应摄像头并返回 CameraSource 实例，支持逐帧读取

#### Scenario: 创建文件视频源
- **WHEN** 指定视频文件路径创建 FileSource
- **THEN** 系统 SHALL 打开视频文件并返回 FileSource 实例，包含正确的帧率和总帧数信息

#### Scenario: 文件不存在处理
- **WHEN** 指定的视频文件路径不存在
- **THEN** 系统 SHALL 抛出 FileNotFoundError 并提供清晰的错误消息

### Requirement: 逐帧读取
所有 VideoSource 实现 SHALL 支持逐帧读取，返回 (success: bool, frame: np.ndarray) 元组。

#### Scenario: 正常读取帧
- **WHEN** 调用 read() 方法且视频源有可用帧
- **THEN** 返回 (True, frame)，其中 frame 为 BGR 格式的 NumPy 数组

#### Scenario: 视频结束
- **WHEN** 视频文件已播放到末尾时调用 read()
- **THEN** 返回 (False, None)

### Requirement: 视频元数据访问
系统 SHALL 提供视频源的元数据访问能力，包括帧率、分辨率和总帧数。

#### Scenario: 获取摄像头帧率
- **WHEN** 查询 CameraSource 的 fps 属性
- **THEN** 返回摄像头的实际帧率值

#### Scenario: 获取视频文件元数据
- **WHEN** 查询 FileSource 的 fps、width、height、total_frames 属性
- **THEN** 返回视频文件的正确元数据信息

### Requirement: 资源管理
所有 VideoSource 实现 SHALL 支持上下文管理器协议（with 语句），确保资源正确释放。

#### Scenario: 使用 with 语句管理资源
- **WHEN** 使用 `with CameraSource(0) as source:` 语法
- **THEN** 退出 with 块时 SHALL 自动调用 release() 释放摄像头资源

#### Scenario: 手动释放资源
- **WHEN** 调用 release() 方法
- **THEN** 系统 SHALL 释放 OpenCV VideoCapture 对象，后续调用 read() 返回 (False, None)
