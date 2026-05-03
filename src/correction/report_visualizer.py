"""
矫正报告可视化

提供骨骼偏差标注图、偏差柱状图和控制台报告输出。
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from .data_types import CorrectionReport

logger = logging.getLogger(__name__)

# 尝试设置中文字体
def _setup_chinese_font():
    """设置 matplotlib 中文字体"""
    for font_name in ["SimHei", "Microsoft YaHei", "STSong", "Arial Unicode MS"]:
        if any(font_name in f.name for f in fm.fontManager.ttflist):
            plt.rcParams["font.sans-serif"] = [font_name]
            plt.rcParams["axes.unicode_minus"] = False
            return
    # 如果都没有，使用默认字体（中文可能显示为方框）
    plt.rcParams["axes.unicode_minus"] = False

_setup_chinese_font()


# 偏差颜色映射
def _deviation_color(dev: float) -> str:
    """根据偏差值返回颜色"""
    if dev < 0.08:
        return "#2ecc71"   # 绿色 - 正常
    elif dev < 0.18:
        return "#f39c12"   # 黄色 - 轻微
    else:
        return "#e74c3c"   # 红色 - 严重


class ReportVisualizer:
    """
    矫正报告可视化工具

    使用示例:
        viz = ReportVisualizer()
        viz.print_report(report)
        viz.plot_deviation_bar(report, save_path="outputs/deviation.png")
    """

    def print_report(self, report: CorrectionReport):
        """将报告以格式化方式打印到控制台"""
        print(report.to_text())

    def plot_deviation_bar(
        self,
        report: CorrectionReport,
        save_path: str = "outputs/deviation_bar.png",
        top_k: int = 15,
    ):
        """
        绘制各关节偏差的水平柱状图

        Args:
            report: 矫正报告
            save_path: 保存路径
            top_k: 显示偏差最大的前 K 个关节
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if not report.joint_deviations:
            logger.warning("无偏差数据，跳过柱状图绘制")
            return

        # 按偏差排序取 top-K
        sorted_joints = sorted(
            report.joint_deviations.items(),
            key=lambda x: x[1], reverse=True,
        )[:top_k]

        # 反转顺序使偏差最大的在最上面
        sorted_joints.reverse()

        names = [name for name, _ in sorted_joints]
        values = [val for _, val in sorted_joints]
        colors = [_deviation_color(val) for val in values]

        fig, ax = plt.subplots(figsize=(10, max(6, len(names) * 0.4)))

        bars = ax.barh(names, values, color=colors, edgecolor="white", height=0.6)

        # 添加阈值线
        ax.axvline(x=0.08, color="#f39c12", linestyle="--", alpha=0.7, label="Mild")
        ax.axvline(x=0.18, color="#e74c3c", linestyle="--", alpha=0.7, label="Severe")

        # 在柱状图上标注数值
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}",
                va="center", fontsize=9,
            )

        ax.set_xlabel("偏差值")
        ax.set_title(
            f"{report.action_display_name or report.action_name} — "
            f"关节偏差分布 (评分: {report.quality_score:.1f})"
        )
        ax.legend(loc="lower right", fontsize=9)
        ax.set_xlim(0, max(values) * 1.3 if values else 0.2)

        plt.tight_layout()
        plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"偏差柱状图已保存: {save_path}")

    def draw_deviation_skeleton(
        self,
        report: CorrectionReport,
        save_path: str = "outputs/deviation_skeleton.png",
    ):
        """
        绘制偏差标注骨骼图

        在骨骼图上用颜色标注各关节的偏差程度：
        - 绿色: 正常 (< 0.08)
        - 黄色: 轻微 (0.08-0.18)
        - 红色: 严重 (> 0.18)

        Args:
            report: 矫正报告
            save_path: 保存路径
        """
        from src.pose_estimation.data_types import POSE_CONNECTIONS, LANDMARK_NAMES

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if not report.joint_deviations:
            logger.warning("无偏差数据，跳过骨骼图绘制")
            return

        fig, ax = plt.subplots(figsize=(8, 10))

        # 简化的骨骼布局坐标（2D 示意图）
        # 基于标准人体正面比例的关键点近似位置
        layout = _get_skeleton_layout()

        # 绘制骨骼连接
        for i, j in POSE_CONNECTIONS:
            if i < len(layout) and j < len(layout):
                x_vals = [layout[i][0], layout[j][0]]
                y_vals = [layout[i][1], layout[j][1]]
                ax.plot(x_vals, y_vals, "gray", linewidth=1.5, alpha=0.5)

        # 绘制关节点（颜色根据偏差）
        for idx, name in enumerate(LANDMARK_NAMES):
            if idx >= len(layout):
                break
            x, y = layout[idx]
            dev = report.joint_deviations.get(name, 0.0)
            color = _deviation_color(dev)
            size = 80 if dev > 0.08 else 40
            ax.scatter(x, y, c=color, s=size, zorder=5, edgecolors="black", linewidths=0.5)

        # 标注偏差最大的关节名称
        top_devs = sorted(
            report.joint_deviations.items(),
            key=lambda x: x[1], reverse=True,
        )[:5]
        for name, dev in top_devs:
            idx = LANDMARK_NAMES.index(name) if name in LANDMARK_NAMES else -1
            if 0 <= idx < len(layout):
                x, y = layout[idx]
                from .rules import JOINT_DISPLAY_NAMES
                display = JOINT_DISPLAY_NAMES.get(name, name)
                ax.annotate(
                    f"{display}\n{dev:.3f}",
                    (x, y), textcoords="offset points",
                    xytext=(15, 10), fontsize=8,
                    arrowprops=dict(arrowstyle="->", color="gray"),
                )

        ax.set_title(
            f"{report.action_display_name or report.action_name} — 偏差标注骨骼图",
            fontsize=14,
        )
        ax.set_xlim(-0.3, 1.3)
        ax.set_ylim(-0.1, 1.1)
        ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.axis("off")

        # 图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#2ecc71", label="OK (<0.08)"),
            Patch(facecolor="#f39c12", label="Mild (0.08-0.18)"),
            Patch(facecolor="#e74c3c", label="Severe (>0.18)"),
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

        plt.tight_layout()
        plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"偏差骨骼图已保存: {save_path}")


def _get_skeleton_layout():
    """
    获取 33 个关键点的 2D 示意图布局坐标

    返回列表，索引对应 MediaPipe 33 关键点。
    坐标范围约 [0, 1]，y 轴向下。
    """
    # 简化布局 (x, y) — 基于正面人体比例
    layout = [None] * 33

    # 头部
    layout[0] = (0.5, 0.08)   # nose
    layout[1] = (0.47, 0.06)  # left_eye_inner
    layout[2] = (0.45, 0.05)  # left_eye
    layout[3] = (0.43, 0.06)  # left_eye_outer
    layout[4] = (0.53, 0.06)  # right_eye_inner
    layout[5] = (0.55, 0.05)  # right_eye
    layout[6] = (0.57, 0.06)  # right_eye_outer
    layout[7] = (0.40, 0.07)  # left_ear
    layout[8] = (0.60, 0.07)  # right_ear
    layout[9] = (0.48, 0.10)  # mouth_left
    layout[10] = (0.52, 0.10) # mouth_right

    # 躯干
    layout[11] = (0.35, 0.22)  # left_shoulder
    layout[12] = (0.65, 0.22)  # right_shoulder
    layout[23] = (0.40, 0.50)  # left_hip
    layout[24] = (0.60, 0.50)  # right_hip

    # 左臂
    layout[13] = (0.25, 0.35)  # left_elbow
    layout[15] = (0.18, 0.48)  # left_wrist
    layout[17] = (0.15, 0.52)  # left_pinky
    layout[19] = (0.16, 0.51)  # left_index
    layout[21] = (0.17, 0.50)  # left_thumb

    # 右臂
    layout[14] = (0.75, 0.35)  # right_elbow
    layout[16] = (0.82, 0.48)  # right_wrist
    layout[18] = (0.85, 0.52)  # right_pinky
    layout[20] = (0.84, 0.51)  # right_index
    layout[22] = (0.83, 0.50)  # right_thumb

    # 左腿
    layout[25] = (0.38, 0.68)  # left_knee
    layout[27] = (0.37, 0.85)  # left_ankle
    layout[29] = (0.35, 0.90)  # left_heel
    layout[31] = (0.39, 0.92)  # left_foot_index

    # 右腿
    layout[26] = (0.62, 0.68)  # right_knee
    layout[28] = (0.63, 0.85)  # right_ankle
    layout[30] = (0.65, 0.90)  # right_heel
    layout[32] = (0.61, 0.92)  # right_foot_index

    return layout