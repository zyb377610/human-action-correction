## Context

当前项目 `src/pose_estimation/` 目录仅包含空的 `__init__.py`。本设计文档定义姿态估计模块的技术架构，作为整个人体动作矫正系统的数据入口层。

该模块需要处理两种场景：
1. **实时模式**：摄像头 30FPS 视频流，要求逐帧处理延迟 < 50ms
2. **离线模式**：视频文件批量处理，输出关键点序列数据文件

### 数据流图

```
┌─────────────┐     ┌──────────────────┐     ┌──────────────────┐
│ VideoSource  │────▶│  PoseEstimator   │────▶│  PoseData (raw)  │
│ (摄像头/文件) │     │  (MediaPipe Pose) │     │  33 landmarks    │
└─────────────┘     └──────────────────┘     └────────┬─────────┘
                                                       │
                                                       ▼
                    ┌──────────────────┐     ┌──────────────────┐
                    │  PoseVisualizer  │◀────│ FeatureExtractor │
                    │  (骨骼图绘制)     │     │ (角度/速度/距离)  │
                    └──────────────────┘     └────────┬─────────┘
                                                       │
                                                       ▼
                                              ┌──────────────────┐
                                              │  PoseSequence    │
                                              │  (时间序列输出)   │
                                              │  → DTW / Model   │
                                              └──────────────────┘
```

## Goals / Non-Goals

**Goals:**
- 封装 MediaPipe Pose 为统一接口，屏蔽底层框架细节
- 定义标准化的关键点数据结构，作为系统内部的数据契约
- 实现关节角度、骨骼比例、运动速度等特征工程
- 支持实时摄像头和视频文件两种输入源
- 提供骨骼可视化能力用于调试和展示
- 实时模式下单帧处理延迟 < 50ms（CPU 模式）

**Non-Goals:**
- 不实现 3D 姿态重建（当前阶段仅使用 2D + depth 的 MediaPipe 原生输出）
- 不实现多人姿态估计（聚焦单人场景）
- 不实现 OpenPose 后端（作为备选方案，后续按需扩展）
- 不实现深度学习模型训练（属于 `src/models/` 模块）

## Decisions

### D1: 选择 MediaPipe 而非 OpenPose

| 维度 | MediaPipe | OpenPose |
|------|-----------|----------|
| 安装复杂度 | pip install，开箱即用 | 需要编译 Caffe/CMake |
| 关键点数量 | 33 个全身关键点 | 25 个（BODY_25） |
| 实时性能 | CPU 下 ~30FPS | 需要 GPU 才能实时 |
| 3D 支持 | 原生 world landmarks (x,y,z) | 仅 2D |
| 跨平台 | 桌面/移动/Web | 主要桌面 |

**决策**：选择 MediaPipe。理由：(1) 安装简便适合毕业设计演示环境；(2) 33 个关键点覆盖更全面；(3) CPU 下即可达到实时性能要求；(4) 参考 ASD 综述中 MediaPipe 在多项研究中的广泛应用验证。

### D2: 关键点数据结构设计

采用 **dataclass + NumPy 混合**方案：

```python
@dataclass
class PoseLandmark:
    x: float          # 归一化坐标 [0, 1]
    y: float
    z: float          # 相对深度
    visibility: float  # 可见度 [0, 1]

@dataclass
class PoseFrame:
    timestamp: float              # 时间戳（秒）
    frame_index: int              # 帧序号
    landmarks: List[PoseLandmark] # 33 个关键点
    
    def to_numpy(self) -> np.ndarray:
        """转为 (33, 4) 的 NumPy 数组 [x, y, z, visibility]"""

@dataclass
class PoseSequence:
    frames: List[PoseFrame]       # 帧序列
    fps: float                    # 帧率
    
    def to_numpy(self) -> np.ndarray:
        """转为 (T, 33, 4) 的 NumPy 数组"""
    
    def save(self, path: str): ...
    def load(cls, path: str): ...
```

**替代方案**：纯 NumPy 数组——灵活性不足，缺少元数据；Protocol Buffer——过于重量级。dataclass 方案在类型安全和易用性之间取得平衡。

### D3: 特征工程策略

基于关键点原始坐标计算三类衍生特征：

| 特征类别 | 计算方法 | 输出维度 | 用途 |
|---------|---------|---------|------|
| 关节角度 | 三点向量夹角 | 选取 8-12 个核心关节 | DTW 对比的主要特征 |
| 骨骼长度比 | 相邻关键点欧氏距离的比值 | ~15 个比值 | 姿态归一化 |
| 运动速度 | 相邻帧关键点位移/时间差 | 33 个关键点 | 动态特征 |

核心关节角度（初步选取）：
- 左/右肘角（肩-肘-腕）
- 左/右膝角（髋-膝-踝）
- 左/右肩角（肘-肩-髋）
- 左/右髋角（肩-髋-膝）

### D4: 视频输入管理策略

采用 **策略模式**统一摄像头和视频文件的接口：

```python
class VideoSource(ABC):
    @abstractmethod
    def read(self) -> Tuple[bool, np.ndarray]: ...
    @abstractmethod
    def release(self): ...
    @property
    def fps(self) -> float: ...

class CameraSource(VideoSource): ...   # cv2.VideoCapture(device_id)
class FileSource(VideoSource): ...     # cv2.VideoCapture(file_path)
```

### D5: 模块文件组织

```
src/pose_estimation/
├── __init__.py           # 公共 API 导出
├── estimator.py          # PoseEstimator 核心类
├── data_types.py         # PoseLandmark, PoseFrame, PoseSequence
├── feature_extractor.py  # 角度、速度、骨骼比例计算
├── visualizer.py         # 骨骼图绘制
└── video_source.py       # VideoSource 抽象与实现

src/utils/
├── config.py             # YAML 配置加载
└── io_utils.py           # 数据序列化工具
```

## Risks / Trade-offs

- **[MediaPipe 版本兼容]** → MediaPipe 更新频繁，API 可能变化。缓解：锁定 `mediapipe>=0.10.0,<1.0.0`，在代码中做版本适配层
- **[单人假设限制]** → 实际场景可能出现多人。缓解：当前取检测到的第一个人，后续可扩展为选择目标人物
- **[遮挡问题]** → 部分关键点被遮挡时 visibility 值低。缓解：在特征工程中根据 visibility 加权，低于阈值的关键点标记为缺失
- **[实时性与精度权衡]** → MediaPipe model_complexity 参数：0(Lite) 快但精度低，2(Heavy) 精度高但慢。缓解：默认使用 1(Full) 平衡方案，通过配置文件可调
- **[坐标归一化]** → MediaPipe 输出归一化坐标 [0,1]，不同分辨率下一致，但丢失了绝对位置信息。缓解：同时保存 world landmarks 的绝对坐标
