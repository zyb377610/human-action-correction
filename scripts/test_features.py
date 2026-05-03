"""测试角度特征的区分度"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging, glob
logging.basicConfig(level=logging.WARNING)
import numpy as np

from src.pose_estimation.estimator import PoseEstimator
from src.pose_estimation.video_source import FileSource
from src.correction.angle_utils import AngleCalculator, ANGLE_DEFINITIONS
from src.data.preprocessing import preprocess_pipeline, extract_action_segment, filter_skeleton_outliers
from src.action_comparison.dtw_algorithms import compute_dtw

TEMPLATE = r"E:\毕业设计\test-resource\hmdb51_org\throw\ScottKazmir_throw_f_cm_np1_fr_med_0.avi"
TEST1 = r"E:\毕业设计\test-resource\hmdb51_org\throw\ScottKazmir_throw_f_cm_np1_fr_med_1.avi"
TEST2 = r"E:\毕业设计\test-resource\hmdb51_org\throw\ScottKazmir_throw_f_cm_np1_fr_med_2.avi"

estimator = PoseEstimator()
calc = AngleCalculator()

def load_and_preprocess(path):
    with FileSource(path) as s:
        seq = estimator.estimate_video(s, show_progress=False)
    seq = filter_skeleton_outliers(seq)
    seq = extract_action_segment(seq)
    seq = preprocess_pipeline(seq, target_frames=60)
    return seq

def seq_to_angle_matrix(seq):
    """序列 → 角度特征矩阵 (T, num_angles)"""
    arr = seq.to_numpy()
    T = arr.shape[0]
    n_angles = len(ANGLE_DEFINITIONS)
    matrix = np.zeros((T, n_angles), dtype=np.float64)
    for t in range(T):
        angles = calc.compute_frame_angles(arr[t])
        for k, name in enumerate(ANGLE_DEFINITIONS.keys()):
            matrix[t, k] = angles[name] / 180.0  # 归一化到 [0,1]
    return matrix

def seq_to_hybrid_matrix(seq):
    """混合特征：角度 + 中心归一化坐标"""
    from src.action_comparison.distance_metrics import sequence_to_feature_matrix, CORE_JOINT_INDICES
    coord_feat = sequence_to_feature_matrix(seq, normalize_body_scale=True, center_normalize=True)
    angle_feat = seq_to_angle_matrix(seq)
    # 角度特征权重放大（角度是 0-1，坐标是 ~0.1-0.5 量级）
    return np.hstack([coord_feat, angle_feat * 3.0])

print("加载视频...")
tpl = load_and_preprocess(TEMPLATE)
t1 = load_and_preprocess(TEST1)
t2 = load_and_preprocess(TEST2)

# 加载不同动作
other_dirs = glob.glob(r"E:\毕业设计\test-resource\hmdb51_org\*")
others = []
for d in other_dirs:
    if "throw" not in d:
        vids = glob.glob(os.path.join(d, "*.avi"))
        if vids:
            try:
                seq = load_and_preprocess(vids[0])
                if seq.num_frames >= 10:
                    others.append((os.path.basename(d), seq))
            except: pass
    if len(others) >= 3:
        break

def test_feature(name, feat_fn):
    tpl_f = feat_fn(tpl)
    t1_f = feat_fn(t1)
    t2_f = feat_fn(t2)

    for algo in ["dtw", "ddtw"]:
        d_self, p_self, _ = compute_dtw(tpl_f, tpl_f, algorithm=algo)
        d1, p1, _ = compute_dtw(t1_f, tpl_f, algorithm=algo)
        d2, p2, _ = compute_dtw(t2_f, tpl_f, algorithm=algo)
        n1 = d1/max(len(p1),1)
        n2 = d2/max(len(p2),1)

        other_results = []
        for oname, oseq in others:
            of = feat_fn(oseq)
            d, p, _ = compute_dtw(of, tpl_f, algorithm=algo)
            nd = d/max(len(p),1)
            other_results.append((oname, nd))

        print(f"\n  [{name} + {algo}]")
        print(f"    自对比: norm_dist=0.0000")
        print(f"    测试1 (同类): norm_dist={n1:.4f}")
        print(f"    测试2 (同类): norm_dist={n2:.4f}")
        for oname, nd in other_results:
            print(f"    {oname:15s} (异类): norm_dist={nd:.4f}")

        # 检查区分度
        same_max = max(n1, n2)
        diff_min = min(nd for _, nd in other_results) if other_results else 999
        gap = diff_min - same_max
        print(f"    >> 区分间隔: {gap:.4f} {'✅ 能区分' if gap > 0.05 else '❌ 不能区分'}")

print("\n" + "=" * 60)
print("特征区分度测试")
print("=" * 60)

test_feature("纯角度", seq_to_angle_matrix)
test_feature("混合(坐标+角度)", seq_to_hybrid_matrix)

# 也测试纯坐标（作为对照）
from src.action_comparison.distance_metrics import sequence_to_feature_matrix
test_feature("纯坐标(中心归一化)", lambda s: sequence_to_feature_matrix(s, normalize_body_scale=True, center_normalize=True))

estimator.close()
