## 1. 基础设施与数据类型

- [ ] 1.1 实现 `src/utils/config.py` — YAML 配置加载工具，支持从 `configs/default.yaml` 读取配置并提供字典式访问。验证：能正确加载并返回 pose_estimation 配置段
- [ ] 1.2 实现 `src/pose_estimation/data_types.py` — 定义 PoseLandmark、PoseFrame、PoseSequence 三个 dataclass，包含 to_numpy() 方法。验证：PoseFrame.to_numpy() 返回 (33, 4) 数组，PoseSequence.to_numpy() 返回 (T, 33, 4) 数组
- [ ] 1.3 实现 PoseSequence 的 save/load 方法 — 支持 .npy 和 .json 两种格式的序列化/反序列化。验证：保存后再加载，数据完全一致（误差 < 1e-6）

## 2. 视频输入管理

- [ ] 2.1 实现 `src/pose_estimation/video_source.py` — VideoSource 抽象基类 + CameraSource 和 FileSource 两个实现。验证：CameraSource 能打开摄像头，FileSource 能打开视频文件并返回正确帧率
- [ ] 2.2 实现上下文管理器协议 — VideoSource 支持 with 语句自动资源释放。验证：with 块退出后摄像头/文件资源被正确释放
- [ ] 2.3 实现视频元数据访问 — fps、width、height、total_frames 属性。验证：FileSource 返回与 OpenCV 查询一致的元数据值

## 3. 姿态估计核心

- [ ] 3.1 实现 `src/pose_estimation/estimator.py` — PoseEstimator 类，封装 MediaPipe Pose 模型。验证：对单张含人体图像调用 estimate_frame() 返回 PoseFrame（33 个关键点）
- [ ] 3.2 实现无人体图像处理 — estimate_frame() 在无人体时返回 None。验证：对纯背景图像返回 None 且不抛异常
- [ ] 3.3 实现视频流处理方法 — estimate_video() 接收 VideoSource 输出 PoseSequence。验证：处理一段测试视频后生成包含正确帧数的 PoseSequence
- [ ] 3.4 实现配置化初始化 — PoseEstimator 从配置文件或参数读取 model_complexity 等参数。验证：设置不同 model_complexity 时 MediaPipe 模型正确切换

## 4. 特征工程

- [ ] 4.1 实现 `src/pose_estimation/feature_extractor.py` — 关节角度计算函数 calculate_angle() 和 get_joint_angles()。验证：已知三点坐标计算角度，与手动计算结果一致
- [ ] 4.2 实现骨骼长度比计算 — get_bone_lengths() 和 get_normalized_bone_ratios()。验证：归一化后的比值不受人体大小影响
- [ ] 4.3 实现运动速度计算 — calculate_velocity() 支持单帧对和序列输入。验证：匀速运动序列的速度计算结果恒定
- [ ] 4.4 实现综合特征向量生成 — extract_features() 组合角度+比例+速度为统一向量。验证：输出向量维度一致，PoseSequence 输入返回 (T, D) 矩阵

## 5. 可视化

- [ ] 5.1 实现 `src/pose_estimation/visualizer.py` — draw_skeleton() 在图像上绘制骨骼连接图和关键点。验证：输出图像包含可见的骨骼连线和关键点标记
- [ ] 5.2 实现可见度过滤绘制 — 低 visibility 关键点使用不同颜色/透明度。验证：遮挡关键点的视觉效果与正常关键点有明显区别
- [ ] 5.3 实现关节角度标注 — draw_angles() 在关节位置显示角度值。验证：标注文本位置正确且不重叠
- [ ] 5.4 实现实时可视化窗口 — run_realtime_demo() 打开摄像头并实时显示骨骼标注视频，支持 q/ESC 退出。验证：窗口正常显示，FPS 显示在画面上

## 6. 模块集成与导出

- [ ] 6.1 更新 `src/pose_estimation/__init__.py` — 导出所有公共 API（PoseEstimator, PoseFrame, PoseSequence, FeatureExtractor, VideoSource 等）。验证：from src.pose_estimation import PoseEstimator 可正常导入
- [ ] 6.2 实现 `src/utils/io_utils.py` — 数据序列化辅助工具（JSON 编码器等）。验证：PoseSequence 能正确序列化为 JSON
- [ ] 6.3 编写 `tests/test_pose_estimation.py` — 单元测试覆盖数据类型、特征工程和视频源。验证：所有测试通过

## 7. 端到端演示

- [ ] 7.1 编写 `scripts/demo_pose.py` — 端到端演示脚本：打开摄像头 → 姿态估计 → 绘制骨骼图 → 显示 FPS。验证：运行脚本后看到实时骨骼标注视频
- [ ] 7.2 编写 `scripts/process_video.py` — 离线视频处理脚本：输入视频文件 → 提取关键点 → 保存 PoseSequence → 输出标注视频。验证：生成的 .npy 文件和标注视频文件存在且内容正确
