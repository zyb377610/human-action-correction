"""
对比结果可视化

提供对齐路径图、距离热力图和偏差雷达图。
"""

import logging
from typing import List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")  # 无头模式，支持无 GUI 环境
import matplotlib.pyplot as plt
from matplotlib import rcParams

from .comparison import ComparisonResult
from .deviation_analyzer import DeviationReport

logger = logging.getLogger(__name__)

# 中文字体支持
rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False


def plot_alignment_path(
    result: ComparisonResult,
    title: str = "DTW 对齐路径",
    save_path: Optional[str] = None,
    figsize: tuple = (8, 8),
) -> plt.Figure:
    """
    绘制 DTW 对齐路径图

    在代价矩阵上叠加最优路径。

    Args:
        result: ComparisonResult（需含 cost_matrix）
        title: 图标题
        save_path: 保存路径（None 则不保存）
        figsize: 图大小

    Returns:
        matplotlib Figure 对象
    """
    fig, ax = plt.subplots(figsize=figsize)

    # 如果有代价矩阵，绘制背景热力图
    if result.cost_matrix is not None:
        # 截断 inf 以便显示
        cm = result.cost_matrix.copy()
        finite_mask = np.isfinite(cm)
        if finite_mask.any():
            cm[~finite_mask] = np.max(cm[finite_mask]) * 1.1
        ax.imshow(cm, origin="lower", cmap="YlOrRd", aspect="auto", alpha=0.7)

    # 绘制对齐路径
    path_i = [p[0] for p in result.path]
    path_j = [p[1] for p in result.path]
    ax.plot(path_j, path_i, "b-", linewidth=2, label="对齐路径")
    ax.plot(path_j[0], path_i[0], "go", markersize=10, label="起点")
    ax.plot(path_j[-1], path_i[-1], "rs", markersize=10, label="终点")

    ax.set_xlabel("模板帧索引")
    ax.set_ylabel("查询帧索引")
    ax.set_title(f"{title}\n距离={result.distance:.4f}  相似度={result.similarity:.2%}")
    ax.legend()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"对齐路径图已保存: {save_path}")

    return fig


def plot_cost_heatmap(
    result: ComparisonResult,
    title: str = "DTW 代价矩阵热力图",
    save_path: Optional[str] = None,
    figsize: tuple = (10, 8),
) -> Optional[plt.Figure]:
    """
    绘制逐帧代价矩阵热力图

    Args:
        result: ComparisonResult（需含 cost_matrix）
        title: 图标题
        save_path: 保存路径
        figsize: 图大小

    Returns:
        matplotlib Figure 对象，无 cost_matrix 时返回 None
    """
    if result.cost_matrix is None:
        logger.warning("FastDTW 不产生代价矩阵，无法绘制热力图")
        return None

    fig, ax = plt.subplots(figsize=figsize)

    cm = result.cost_matrix.copy()
    finite_mask = np.isfinite(cm)
    if finite_mask.any():
        cm[~finite_mask] = np.max(cm[finite_mask]) * 1.1

    im = ax.imshow(cm, origin="lower", cmap="viridis", aspect="auto")
    plt.colorbar(im, ax=ax, label="累积代价")

    ax.set_xlabel("模板帧索引")
    ax.set_ylabel("查询帧索引")
    ax.set_title(title)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"热力图已保存: {save_path}")

    return fig


def plot_deviation_radar(
    report: DeviationReport,
    title: str = "关节偏差雷达图",
    save_path: Optional[str] = None,
    figsize: tuple = (10, 10),
    max_joints: int = 12,
) -> plt.Figure:
    """
    绘制关节偏差雷达图

    显示偏差最大的 N 个关节在极坐标下的偏差分布。

    Args:
        report: DeviationReport
        title: 图标题
        save_path: 保存路径
        figsize: 图大小
        max_joints: 雷达图最多显示的关节数

    Returns:
        matplotlib Figure 对象
    """
    # 取偏差最大的 N 个关节
    sorted_joints = sorted(
        report.joint_deviations.items(), key=lambda x: x[1], reverse=True
    )[:max_joints]

    labels = [name for name, _ in sorted_joints]
    values = [dev for _, dev in sorted_joints]

    # 闭合雷达图
    num = len(labels)
    angles = np.linspace(0, 2 * np.pi, num, endpoint=False).tolist()
    angles += angles[:1]
    values_closed = values + values[:1]

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))

    ax.fill(angles, values_closed, alpha=0.25, color="red")
    ax.plot(angles, values_closed, "o-", color="red", linewidth=2)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    severity_cn = {"mild": "轻微", "moderate": "中等", "severe": "严重"}
    severity_label = severity_cn.get(report.severity, report.severity)
    ax.set_title(
        f"{title}\n整体偏差: {report.overall_deviation:.4f} ({severity_label})",
        pad=20,
    )

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"雷达图已保存: {save_path}")

    return fig


def plot_frame_deviation_curve(
    report: DeviationReport,
    title: str = "逐帧偏差曲线",
    save_path: Optional[str] = None,
    figsize: tuple = (12, 4),
) -> plt.Figure:
    """
    绘制逐帧（沿对齐路径）偏差曲线

    Args:
        report: DeviationReport
        title: 图标题
        save_path: 保存路径
        figsize: 图大小

    Returns:
        matplotlib Figure 对象
    """
    fig, ax = plt.subplots(figsize=figsize)

    steps = np.arange(len(report.frame_deviations))
    ax.plot(steps, report.frame_deviations, "b-", linewidth=1.5)
    ax.fill_between(steps, report.frame_deviations, alpha=0.2)

    ax.set_xlabel("对齐步骤")
    ax.set_ylabel("总偏差（33 关节之和）")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"偏差曲线已保存: {save_path}")

    return fig
