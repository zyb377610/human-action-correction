#!/usr/bin/env python
"""
一键启动 Web 应用

使用方法：
    python scripts/launch_app.py
    python scripts/launch_app.py --share            # 公网分享链接
    python scripts/launch_app.py --port 8080        # 指定端口
    python scripts/launch_app.py --templates data/templates  # 指定模板目录
"""

import argparse
import logging
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.app.pipeline import AppPipeline
from src.app.gradio_ui import create_gradio_app


def parse_args():
    parser = argparse.ArgumentParser(
        description="启动人体动作矫正系统 Web 界面"
    )
    parser.add_argument(
        "--templates", type=str, default=None,
        help="模板库目录路径 (默认: data/templates)"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="分类模型 checkpoint 路径 (可选，用于自动识别)"
    )
    parser.add_argument(
        "--model-type", type=str, default="bilstm",
        choices=["stgcn", "bilstm", "transformer"],
        help="分类模型类型 (默认: bilstm)"
    )
    parser.add_argument(
        "--port", type=int, default=7860,
        help="Web 服务端口 (默认: 7860)"
    )
    parser.add_argument(
        "--share", action="store_true",
        help="生成公网分享链接"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="启用调试模式"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 配置日志
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    logger.info("=" * 50)
    logger.info("🏋️ 人体动作矫正系统 - Web 界面启动")
    logger.info("=" * 50)

    # 初始化流水线
    logger.info("正在初始化系统组件…")
    try:
        pipeline = AppPipeline(
            templates_dir=args.templates,
            checkpoint_path=args.checkpoint,
            model_type=args.model_type,
        )
    except Exception as e:
        logger.error(f"初始化失败: {e}")
        logger.info("提示: 确保 models/ 目录下有 MediaPipe 模型文件")
        sys.exit(1)

    # 创建 Gradio 应用
    logger.info("正在创建 Web 界面…")
    app = create_gradio_app(pipeline)

    # 启动
    logger.info(f"启动 Web 服务: http://0.0.0.0:{args.port}")
    if args.share:
        logger.info("公网分享链接已启用")

    app.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        show_error=True,
    )


if __name__ == "__main__":
    main()
