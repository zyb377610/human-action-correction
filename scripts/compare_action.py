"""
命令行动作对比脚本

用法:
    python scripts/compare_action.py --query data/user_seq.json --action squat
    python scripts/compare_action.py --query data/user_seq.json --template data/templates/squat/standard_01.json
    python scripts/compare_action.py --query data/user_seq.json --action squat --algorithm fastdtw --metric cosine
"""

import argparse
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pose_estimation.data_types import PoseSequence
from src.data.template_library import TemplateLibrary
from src.action_comparison.comparison import ActionComparator
from src.action_comparison.deviation_analyzer import JointDeviationAnalyzer
from src.data.preprocessing import preprocess_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="动作对比工具 — 基于 DTW 比较用户动作与标准模板"
    )
    parser.add_argument(
        "--query", type=str, required=True,
        help="用户动作序列文件路径 (.json / .npy)"
    )
    parser.add_argument(
        "--template", type=str, default=None,
        help="单个标准模板文件路径 (.json / .npy)"
    )
    parser.add_argument(
        "--action", type=str, default=None,
        help="动作类别名称（与模板库中所有模板对比）"
    )
    parser.add_argument(
        "--algorithm", type=str, default="dtw",
        choices=["dtw", "fastdtw", "ddtw"],
        help="DTW 算法 (默认: dtw)"
    )
    parser.add_argument(
        "--metric", type=str, default="euclidean",
        choices=["euclidean", "cosine", "manhattan"],
        help="距离度量 (默认: euclidean)"
    )
    parser.add_argument(
        "--window-size", type=int, default=None,
        help="Sakoe-Chiba 带宽约束"
    )
    parser.add_argument(
        "--target-frames", type=int, default=60,
        help="归一化帧数 (默认: 60)"
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="输出偏差最大的前 K 个关节 (默认: 5)"
    )
    parser.add_argument(
        "--save-plots", type=str, default=None,
        help="保存可视化图的目录"
    )
    parser.add_argument(
        "--templates-dir", type=str, default=None,
        help="模板库目录"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # 加载用户序列
    query_seq = PoseSequence.load(args.query)
    print(f"用户序列: {query_seq.num_frames} 帧")

    # 构建对比器
    comparator = ActionComparator(
        algorithm=args.algorithm,
        metric=args.metric,
        window_size=args.window_size,
        preprocess=True,
        target_frames=args.target_frames,
    )

    analyzer = JointDeviationAnalyzer(top_k=args.top_k)

    if args.template:
        # 单模板对比
        template_seq = PoseSequence.load(args.template)
        print(f"模板序列: {template_seq.num_frames} 帧")

        result = comparator.compare(query_seq, template_seq, template_name=args.template)

        # 偏差分析（需要预处理后的序列）
        q_pp = preprocess_pipeline(query_seq, target_frames=args.target_frames)
        t_pp = preprocess_pipeline(template_seq, target_frames=args.target_frames)
        report = analyzer.analyze(q_pp, t_pp, result)

        _print_result(result, report)

        if args.save_plots:
            _save_plots(result, report, args.save_plots, "single")

    elif args.action:
        # 模板库批量对比
        library = TemplateLibrary(args.templates_dir)
        results = comparator.compare_with_templates(query_seq, library, args.action)

        if not results:
            print(f"动作 '{args.action}' 没有可用模板")
            return

        print(f"\n{'='*60}")
        print(f"  动作对比结果: {args.action} ({len(results)} 个模板)")
        print(f"  算法: {args.algorithm} | 度量: {args.metric}")
        print(f"{'='*60}")

        for i, result in enumerate(results):
            print(f"\n--- 模板 #{i+1}: {result.template_name} ---")
            print(f"  距离: {result.distance:.4f}")
            print(f"  相似度: {result.similarity:.2%}")

        # 最佳匹配的详细偏差分析
        best = results[0]
        print(f"\n{'='*60}")
        print(f"  最佳匹配: {best.template_name}")
        print(f"{'='*60}")

        q_pp = preprocess_pipeline(query_seq, target_frames=args.target_frames)
        # 重新加载最佳模板
        t_seq = library.load_template(args.action, best.template_name)
        t_pp = preprocess_pipeline(t_seq, target_frames=args.target_frames)
        report = analyzer.analyze(q_pp, t_pp, best)
        print(f"\n{report.summary()}")

        if args.save_plots:
            _save_plots(best, report, args.save_plots, args.action)
    else:
        print("请指定 --template 或 --action 参数")


def _print_result(result, report):
    """打印单次对比结果"""
    print(f"\n{'='*60}")
    print(f"  对比结果")
    print(f"  算法: {result.algorithm} | 度量: {result.metric}")
    print(f"{'='*60}")
    print(f"  DTW 距离: {result.distance:.4f}")
    print(f"  归一化距离: {result.normalized_distance:.4f}")
    print(f"  相似度: {result.similarity:.2%}")
    print(f"  对齐路径长度: {result.path_length}")
    print(f"\n{report.summary()}")


def _save_plots(result, report, output_dir, prefix):
    """保存可视化图"""
    os.makedirs(output_dir, exist_ok=True)

    from src.action_comparison.visualizer import (
        plot_alignment_path,
        plot_cost_heatmap,
        plot_deviation_radar,
        plot_frame_deviation_curve,
    )

    plot_alignment_path(
        result,
        save_path=os.path.join(output_dir, f"{prefix}_alignment.png"),
    )
    plot_cost_heatmap(
        result,
        save_path=os.path.join(output_dir, f"{prefix}_heatmap.png"),
    )
    plot_deviation_radar(
        report,
        save_path=os.path.join(output_dir, f"{prefix}_radar.png"),
    )
    plot_frame_deviation_curve(
        report,
        save_path=os.path.join(output_dir, f"{prefix}_deviation_curve.png"),
    )
    print(f"\n可视化图已保存至: {output_dir}")
    import matplotlib.pyplot as plt
    plt.close("all")


if __name__ == "__main__":
    main()
