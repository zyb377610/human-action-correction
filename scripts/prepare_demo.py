#!/usr/bin/env python
"""
演示准备脚本

为 5 种动作各生成一个合成模板，保存到模板库。
用于在没有真实视频的情况下快速准备演示环境。

使用方法：
    python scripts/prepare_demo.py
    python scripts/prepare_demo.py --templates data/templates
"""

import argparse
import sys
from pathlib import Path

import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.pose_estimation.data_types import PoseFrame, PoseSequence, PoseLandmark
from src.data.template_library import TemplateLibrary

# 5 种标准动作
DEMO_ACTIONS = {
    "squat": "深蹲",
    "arm_raise": "手臂举起",
    "side_bend": "侧弯",
    "lunge": "弓步",
    "standing_stretch": "站立拉伸",
}


def generate_synthetic_sequence(action: str, num_frames: int = 60, fps: float = 30.0) -> PoseSequence:
    """
    为指定动作生成合成骨骼序列

    根据动作类型，对关键关节施加不同的运动模式。
    """
    np.random.seed(hash(action) % 2**31)
    seq = PoseSequence(fps=fps)

    for i in range(num_frames):
        t = i / num_frames  # 归一化时间 [0, 1)
        landmarks = []

        for j in range(33):
            # 基础位置：T-pose 近似
            base_x = 0.5
            base_y = j / 33.0
            base_z = 0.0

            # 根据动作类型添加运动
            dx, dy, dz = 0.0, 0.0, 0.0

            if action == "squat":
                # 深蹲：膝盖弯曲，重心下移
                if j in [25, 26]:  # 膝盖
                    dy += 0.1 * np.sin(np.pi * t)
                    dz += 0.05 * np.sin(np.pi * t)
                if j in [23, 24]:  # 髋部
                    dy += 0.08 * np.sin(np.pi * t)

            elif action == "arm_raise":
                # 手臂举起：肩膀以上的关节上举
                if j in [11, 12, 13, 14, 15, 16]:  # 手臂
                    dy -= 0.15 * np.sin(np.pi * t)
                    dx += 0.05 * np.sin(np.pi * t) * (1 if j % 2 == 0 else -1)

            elif action == "side_bend":
                # 侧弯：躯干侧向弯曲
                if j in range(11, 24):  # 上半身
                    dx += 0.08 * np.sin(np.pi * t)

            elif action == "lunge":
                # 弓步：一腿前伸
                if j in [25, 27]:  # 左腿
                    dz += 0.1 * np.sin(np.pi * t)
                    dy += 0.05 * np.sin(np.pi * t)
                if j in [26, 28]:  # 右腿
                    dz -= 0.05 * np.sin(np.pi * t)

            elif action == "standing_stretch":
                # 站立拉伸：双手上举，身体微微后仰
                if j in [15, 16, 19, 20]:  # 手腕、手指
                    dy -= 0.2 * np.sin(np.pi * t)
                if j in [11, 12]:  # 肩膀
                    dy -= 0.1 * np.sin(np.pi * t)

            landmarks.append(PoseLandmark(
                x=base_x + dx + np.random.randn() * 0.005,
                y=base_y + dy + np.random.randn() * 0.005,
                z=base_z + dz + np.random.randn() * 0.005,
                visibility=0.95 + np.random.randn() * 0.02,
            ))

        seq.add_frame(PoseFrame(
            timestamp=i / fps,
            frame_index=i,
            landmarks=landmarks,
        ))

    return seq


def parse_args():
    parser = argparse.ArgumentParser(description="生成演示用模板数据")
    parser.add_argument("--templates", type=str, default="data/templates", help="模板库目录")
    return parser.parse_args()


def main():
    args = parse_args()

    templates_dir = Path(args.templates)
    templates_dir.mkdir(parents=True, exist_ok=True)

    lib = TemplateLibrary(str(templates_dir))

    print("🏋️ 演示准备 — 生成标准动作模板")
    print("=" * 40)

    for action_name, display_name in DEMO_ACTIONS.items():
        print(f"\n📝 生成 {display_name} ({action_name}) 模板…")

        # 创建动作类别
        if action_name not in lib.list_actions():
            lib.add_action(action_name, display_name)

        # 生成合成序列
        seq = generate_synthetic_sequence(action_name, num_frames=60)

        # 保存为模板
        template_name = f"demo_standard"
        lib.add_template(action_name, seq, template_name)

        print(f"   ✅ 已保存: {action_name}/{template_name} ({seq.num_frames} 帧)")

    print(f"\n{'='*40}")
    print(f"🎉 完成！共生成 {len(DEMO_ACTIONS)} 个动作模板")
    print(f"   模板库位置: {templates_dir.resolve()}")
    print(f"\n可以使用以下命令启动 Web 界面：")
    print(f"   python scripts/launch_app.py --templates {args.templates}")


if __name__ == "__main__":
    main()
