## 1. 数据预处理

- [ ] 1.1 创建 `src/data/__init__.py` 和 `src/data/preprocessing.py` — 实现缺失值线性插值函数 `interpolate_missing()`，对 visibility < 阈值的关键点进行插值填充
- [ ] 1.2 实现平滑滤波 `smooth_sequence()` — 基于 scipy Savitzky-Golay 滤波器对关键点序列进行平滑去噪
- [ ] 1.3 实现序列长度归一化 `resample_sequence()` — 将不等长 PoseSequence 重采样到固定帧数（如 60 帧）
- [ ] 1.4 实现预处理流水线 `preprocess_pipeline()` — 按顺序执行插值→滤波→归一化

## 2. 标准模板库

- [ ] 2.1 实现 `src/data/template_library.py` — TemplateLibrary 类，管理动作类别和模板文件（增删查）
- [ ] 2.2 实现模板加载与保存 — 从 `data/templates/<action>/` 加载/保存 PoseSequence
- [ ] 2.3 创建初始动作类别配置 — 在 `configs/default.yaml` 中添加 actions 配置段（squat/arm_raise/side_bend 等）

## 3. 数据增强

- [ ] 3.1 实现 `src/data/augmentation.py` — 时间拉伸/压缩 `time_warp()`，按比例调整序列速度
- [ ] 3.2 实现高斯噪声添加 `add_noise()` — 对关键点坐标添加随机噪声
- [ ] 3.3 实现左右镜像翻转 `mirror_sequence()` — 交换左右关键点生成镜像动作
- [ ] 3.4 实现批量增强 `augment_batch()` — 组合多种增强策略批量生成样本

## 4. 录制工具与集成

- [ ] 4.1 实现 `src/data/recorder.py` — ActionRecorder 类，交互式录制标准动作（倒计时→录制→预览→保存）
- [ ] 4.2 编写 `scripts/record_action.py` — 命令行录制脚本，支持指定动作类别和保存路径
- [ ] 4.3 更新 `src/data/__init__.py` — 导出所有公共 API
- [ ] 4.4 编写 `tests/test_data.py` — 单元测试覆盖预处理、增强和模板库
