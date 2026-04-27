## Context

系统已完成 6 个阶段的开发：

| Phase | 模块 | 目录 | 状态 |
|-------|------|------|------|
| 1 | 姿态估计 | `src/pose_estimation/` | ✅ |
| 2 | 数据预处理 | `src/data/` | ✅ |
| 3 | 动作对比 | `src/action_comparison/` | ✅ |
| 4 | 深度学习 | `src/models/` | ✅ |
| 5 | 矫正反馈 | `src/correction/` | ✅ |
| 6 | Web 界面 | `src/app/` | ✅ |

现有测试：
- `tests/test_pose_estimation.py` — 姿态估计单元测试
- `tests/test_data.py` — 数据预处理单元测试
- `tests/test_action_comparison.py` — DTW 对比单元测试
- `tests/test_models.py` — 深度学习模型单元测试
- `tests/test_correction.py` — 矫正反馈单元测试
- `tests/test_app.py` — Web 应用单元测试

缺失：端到端集成测试、离线批处理脚本、完整文档、统一配置。

## Goals / Non-Goals

**Goals:**
- 确保所有模块协同工作：从合成测试数据通过完整流水线
- 提供统一配置文件，便于调整路径和参数
- 实现离线批处理，支持处理一整个文件夹的视频
- 提供演示准备脚本，一键生成模板数据
- 更新所有文档到最终状态
- 确保全量测试通过（目标：0 failure）

**Non-Goals:**
- 不做性能基准测试（FPS 测量留作论文实验）
- 不做 Docker 容器化部署
- 不做 CI/CD 流水线搭建

## Decisions

### 1. 统一配置：YAML 配置文件

在 `configs/default_config.yaml` 中集中管理所有可配置项：

```yaml
# 路径配置
paths:
  templates_dir: data/templates
  models_dir: models/checkpoints
  output_dir: outputs

# 模型配置
model:
  type: bilstm          # stgcn / bilstm / transformer
  checkpoint: null       # null = 无预训练模型（跳过自动分类）

# 姿态估计
pose_estimation:
  model_path: models/pose_landmarker_heavy.task
  max_frames: 300

# Web 服务
server:
  port: 7860
  share: false
```

### 2. 集成测试策略

使用 **合成数据** 驱动的端到端测试，避免依赖实际视频文件：
- 用 numpy 生成模拟骨骼序列 (T, 33, 4)
- 通过 `AppPipeline.analyze_sequence()` 走完整流水线
- 验证返回的 AnalysisResult 结构完整
- 用 Mock 跳过 MediaPipe（不需要实际 GPU/摄像头）

### 3. 离线批处理

`scripts/batch_process.py` 功能：
- 输入：视频文件夹路径
- 处理：逐个视频调用 AppPipeline.analyze_video()
- 输出：CSV 汇总（文件名, 动作类型, 评分, 矫正条数）+ 每个视频的详细报告

### 4. README 更新

保留原有结构，更新：
- 快速开始：实际的启动命令
- 开发计划：所有 Phase 打钩
- 功能介绍：补充各模块实际实现的技术

## Risks / Trade-offs

| 风险 | 缓解 |
|------|------|
| 集成测试中 MediaPipe 依赖实际模型文件 | 使用 Mock + 合成数据绕过 |
| 批处理长视频耗时长 | 复用 MAX_FRAMES=300 截断策略 |

## Open Questions

- 无
