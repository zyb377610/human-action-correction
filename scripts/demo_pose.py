"""
实时姿态估计演示

打开摄像头 → MediaPipe 姿态估计 → 绘制骨骼图 → 显示 FPS
按 q 或 ESC 退出
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.pose_estimation import PoseEstimator, run_realtime_demo


def main():
    print("=" * 50)
    print("  基于深度学习的人体动作矫正系统 — 姿态估计演示")
    print("=" * 50)

    with PoseEstimator() as estimator:
        print(f"模型已加载: {estimator}")
        run_realtime_demo(
            estimator=estimator,
            device_id=0,
            width=640,
            height=480,
            show_angles=True,
        )


if __name__ == "__main__":
    main()
