"""
端到端矫正分析演示脚本

用法:
    python -m scripts.run_correction --input data/samples/test.json --action squat
    python -m scripts.run_correction --input data/samples/test.npy --action arm_raise --output outputs/correction/
"""

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pose_estimation.data_types import PoseSequence
from src.correction.pipeline import CorrectionPipeline
from src.correction.report_visualizer import ReportVisualizer


def parse_args():
    parser = argparse.ArgumentParser(description="端到端动作矫正分析")

    parser.add_argument(
        "--input", type=str, required=True,
        help="输入序列文件路径 (.json 或 .npy)",
    )
    parser.add_argument(
        "--action", type=str, default=None,
        help="动作类别 (squat/arm_raise/side_bend/lunge/standing_stretch)，不指定则自动识别",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="分类模型 checkpoint 路径（自动模式需要）",
    )
    parser.add_argument(
        "--model", type=str, default="bilstm",
        help="模型类型 (stgcn/bilstm/transformer)",
    )
    parser.add_argument(
        "--templates_dir", type=str, default=None,
        help="模板库目录",
    )
    parser.add_argument(
        "--algorithm", type=str, default="dtw",
        choices=["dtw", "fastdtw", "ddtw"],
        help="DTW 算法",
    )
    parser.add_argument(
        "--output", type=str, default="outputs/correction",
        help="输出目录",
    )

    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("run_correction")

    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载用户动作序列
    logger.info(f"加载序列: {args.input}")
    sequence = PoseSequence.load(args.input)
    logger.info(f"序列信息: {sequence.num_frames} 帧, {sequence.fps} FPS")

    # 创建流水线
    pipeline = CorrectionPipeline(
        templates_dir=args.templates_dir,
        checkpoint_path=args.checkpoint,
        model_type=args.model,
        algorithm=args.algorithm,
    )

    # 执行分析
    try:
        report = pipeline.analyze(
            user_sequence=sequence,
            action_name=args.action,
        )
    except ValueError as e:
        logger.error(f"分析失败: {e}")
        sys.exit(1)

    # 输出报告
    visualizer = ReportVisualizer()

    # 1. 控制台打印
    visualizer.print_report(report)

    # 2. 偏差柱状图
    bar_path = str(output_dir / "deviation_bar.png")
    visualizer.plot_deviation_bar(report, save_path=bar_path)

    # 3. 偏差骨骼图
    skeleton_path = str(output_dir / "deviation_skeleton.png")
    visualizer.draw_deviation_skeleton(report, save_path=skeleton_path)

    # 4. JSON 报告
    json_path = output_dir / "correction_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(report.to_json())
    logger.info(f"JSON 报告已保存: {json_path}")

    # 5. 文本报告
    text_path = output_dir / "correction_report.txt"
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(report.to_text())
    logger.info(f"文本报告已保存: {text_path}")

    logger.info("=" * 50)
    logger.info("矫正分析完成!")
    logger.info(f"输出目录: {output_dir}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()