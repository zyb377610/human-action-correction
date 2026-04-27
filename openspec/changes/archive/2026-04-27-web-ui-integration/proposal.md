## Why

系统已完成 Phase 1-5 的核心算法模块（姿态估计、数据预处理、DTW 对比、深度学习分类、矫正反馈），但目前所有功能仅能通过 Python 脚本调用，缺少统一的可视化交互界面。毕业答辩需要一个直观的 Web 演示平台，让评审老师能通过摄像头实时或上传视频的方式体验完整的动作矫正流程。

## What Changes

- 新增基于 **Gradio** 的 Web 交互界面（`src/app/`）
- 实现 **视频上传模式**：上传视频文件 → 姿态估计 → 动作分类 → DTW 对比 → 矫正报告
- 实现 **实时摄像头模式**：摄像头采集 → 实时骨骼叠加 → 动作完成后自动分析
- 实现 **模板管理页面**：查看/录入标准动作模板
- 整合所有 Phase 1-5 模块为端到端流水线
- 新增 `scripts/launch_app.py` 一键启动脚本
- 更新 `requirements.txt` 添加 Gradio 依赖

## Capabilities

### New Capabilities
- `web-interface`: Gradio Web 交互界面，包含多 Tab 布局（视频上传分析、实时摄像头、模板管理、系统说明）
- `app-pipeline`: 应用层端到端流水线，串联姿态估计 → 分类 → 对比 → 矫正的完整流程，提供统一调用接口
- `session-manager`: 会话状态管理，维护摄像头采集的帧序列、分析结果缓存、模板加载状态

### Modified Capabilities
- `correction-pipeline`: 增加对 Gradio 回调的适配，支持进度回调和中间结果返回

## Impact

- **新增依赖**: `gradio >= 4.0`
- **影响代码**: `src/app/` 目录（新建）、`src/correction/pipeline.py`（微调接口）
- **影响文件**: `requirements.txt`、`README.md`
- **预计开发时间**: 1-2 天
