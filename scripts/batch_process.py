#!/usr/bin/env python
"""
离线批处理脚本

批量处理视频文件夹，输出 CSV 汇总和逐个视频的矫正报告。

使用方法：
    python scripts/batch_process.py --input videos/ --output results/
    python scripts/batch_process.py --input videos/ --action squat
"""

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.app.pipeline import AppPipeline

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}


def parse_args():
    parser = argparse.ArgumentParser(description="批量处理视频文件夹")
    parser.add_argument("--input", "-i", type=str, required=True, help="视频文件夹路径")
    parser.add_argument("--output", "-o", type=str, default="outputs/batch", help="输出目录")
    parser.add_argument("--action", "-a", type=str, default=None, help="指定动作类型（不指定则自动识别）")
    parser.add_argument("--templates", type=str, default=None, help="模板库目录")
    parser.add_argument("--checkpoint", type=str, default=None, help="分类模型权重路径")
    parser.add_argument("--model-type", type=str, default="bilstm", help="模型类型")
    return parser.parse_args()


def find_videos(input_dir: str):
    """查找文件夹中的视频文件"""
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"❌ 输入目录不存在: {input_dir}")
        sys.exit(1)

    videos = sorted(
        p for p in input_path.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )
    return videos


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    # 查找视频
    videos = find_videos(args.input)
    if not videos:
        print(f"⚠️ 在 {args.input} 中未找到视频文件")
        sys.exit(0)

    print(f"📂 找到 {len(videos)} 个视频文件")

    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(exist_ok=True)

    # 初始化流水线
    print("🔧 正在初始化系统…")
    pipeline = AppPipeline(
        templates_dir=args.templates,
        checkpoint_path=args.checkpoint,
        model_type=args.model_type,
        output_dir=str(output_dir),
    )

    # 批处理
    csv_path = output_dir / "summary.csv"
    results = []

    for idx, video_path in enumerate(videos, 1):
        print(f"\n{'='*50}")
        print(f"[{idx}/{len(videos)}] 处理: {video_path.name}")
        print(f"{'='*50}")

        start_time = time.time()

        try:
            result = pipeline.analyze_video(
                video_path=str(video_path),
                action_name=args.action,
            )
            elapsed = time.time() - start_time

            row = {
                "文件名": video_path.name,
                "动作类型": result.action_display_name or result.action_name or "未识别",
                "评分": f"{result.quality_score:.1f}",
                "相似度": f"{result.similarity:.3f}",
                "矫正建议数": result.num_corrections,
                "耗时(秒)": f"{elapsed:.1f}",
                "状态": "成功",
            }
            results.append(row)

            # 保存详细报告
            report_path = reports_dir / f"{video_path.stem}_report.txt"
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(f"文件: {video_path.name}\n")
                f.write(f"{'='*40}\n\n")
                f.write(result.report_text)

            print(f"✅ 完成 | 评分: {result.quality_score:.1f} | 建议: {result.num_corrections} 条 | 耗时: {elapsed:.1f}s")

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"处理失败: {e}")
            results.append({
                "文件名": video_path.name,
                "动作类型": "-",
                "评分": "-",
                "相似度": "-",
                "矫正建议数": "-",
                "耗时(秒)": f"{elapsed:.1f}",
                "状态": f"失败: {str(e)[:50]}",
            })
            print(f"❌ 失败: {e}")

    # 写入 CSV 汇总
    if results:
        fieldnames = ["文件名", "动作类型", "评分", "相似度", "矫正建议数", "耗时(秒)", "状态"]
        with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    # 总结
    print(f"\n{'='*50}")
    print(f"📊 批处理完成")
    print(f"   总视频: {len(videos)}")
    success = sum(1 for r in results if r["状态"] == "成功")
    print(f"   成功: {success}")
    print(f"   失败: {len(videos) - success}")
    print(f"   CSV 汇总: {csv_path}")
    print(f"   详细报告: {reports_dir}/")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
