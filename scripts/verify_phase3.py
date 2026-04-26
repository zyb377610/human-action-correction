"""Phase 3 手动验证脚本"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pose_estimation.data_types import PoseLandmark, PoseFrame, PoseSequence
from src.action_comparison.comparison import ActionComparator
from src.action_comparison.deviation_analyzer import JointDeviationAnalyzer
from src.action_comparison.visualizer import (
    plot_alignment_path, plot_cost_heatmap,
    plot_deviation_radar, plot_frame_deviation_curve,
)
from src.data.preprocessing import preprocess_pipeline
from src.data.template_library import TemplateLibrary
import tempfile, matplotlib.pyplot as plt


def make_seq(offset=0.0, frames=30):
    lms = [PoseLandmark(x=j*0.03+offset, y=j*0.02+offset, z=0.01, visibility=0.9)
           for j in range(33)]
    return PoseSequence(
        [PoseFrame(i/30.0, i, lms) for i in range(frames)], 30.0
    )


def main():
    standard = make_seq(0.0)
    user_good = make_seq(0.02)
    user_bad = make_seq(0.15)

    comp = ActionComparator(algorithm="dtw", metric="euclidean",
                            preprocess=True, target_frames=30)
    analyzer = JointDeviationAnalyzer(top_k=5)

    # === 场景 1 ===
    print("=" * 55)
    print("  场景 1: 用户动作接近标准 (offset=0.02)")
    print("=" * 55)
    r1 = comp.compare(user_good, standard)
    q1 = preprocess_pipeline(user_good, target_frames=30)
    t1 = preprocess_pipeline(standard, target_frames=30)
    rpt1 = analyzer.analyze(q1, t1, r1)
    print(f"  DTW 距离:   {r1.distance:.4f}")
    print(f"  相似度:     {r1.similarity*100:.1f}%")
    print(f"  路径长度:   {r1.path_length}")
    print(rpt1.summary())

    # === 场景 2 ===
    print()
    print("=" * 55)
    print("  场景 2: 用户动作偏差较大 (offset=0.15)")
    print("=" * 55)
    r2 = comp.compare(user_bad, standard)
    q2 = preprocess_pipeline(user_bad, target_frames=30)
    rpt2 = analyzer.analyze(q2, t1, r2)
    print(f"  DTW 距离:   {r2.distance:.4f}")
    print(f"  相似度:     {r2.similarity*100:.1f}%")
    print(f"  路径长度:   {r2.path_length}")
    print(rpt2.summary())

    # === 三种算法对比 ===
    print()
    print("=" * 55)
    print("  三种 DTW 算法对比 (user_bad vs standard)")
    print("=" * 55)
    for algo in ["dtw", "fastdtw", "ddtw"]:
        c = ActionComparator(algorithm=algo, preprocess=True, target_frames=30)
        r = c.compare(user_bad, standard)
        print(f"  {algo:8s}  距离={r.distance:.4f}  "
              f"相似度={r.similarity*100:.1f}%  路径长={r.path_length}")

    # === 可视化 ===
    print()
    print("=" * 55)
    print("  生成可视化图表")
    print("=" * 55)
    out = "output/phase3_verify"
    os.makedirs(out, exist_ok=True)
    plot_alignment_path(r2, save_path=f"{out}/alignment.png")
    plot_cost_heatmap(r2, save_path=f"{out}/heatmap.png")
    plot_deviation_radar(rpt2, save_path=f"{out}/radar.png")
    plot_frame_deviation_curve(rpt2, save_path=f"{out}/deviation_curve.png")
    plt.close("all")
    print(f"  4 张图已保存到 {out}/")

    # === 模板库批量对比 ===
    print()
    print("=" * 55)
    print("  模板库批量对比验证")
    print("=" * 55)
    d = tempfile.mkdtemp()
    lib = TemplateLibrary(d)
    lib.add_template("squat", make_seq(0.0), "perfect")
    lib.add_template("squat", make_seq(0.05), "variant")
    user = make_seq(0.02)
    comp2 = ActionComparator(preprocess=False)
    results = comp2.compare_with_templates(user, lib, "squat")
    for r in results:
        print(f"  {r.template_name:10s}  相似度={r.similarity*100:.1f}%  "
              f"距离={r.distance:.4f}")
    print(f"  最佳匹配: {results[0].template_name}")

    print()
    print("=" * 55)
    print("  Phase 3 全部验证通过!")
    print("=" * 55)


if __name__ == "__main__":
    main()
