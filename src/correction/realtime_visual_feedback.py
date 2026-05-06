"""
实时视觉反馈增强模块

提供两大能力：
1. 骨骼画面视觉高亮（偏差着色 + 方向箭头 + HUD 文字）
2. 右侧面板信息持久化（建议钉住 + 分层显示 + 降频刷新）

设计原则：
- "同时以视觉高亮、文字提示等形式提供实时矫正反馈"
- 解决信息闪过太快、用户来不及看的问题
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.correction.realtime_feedback import (
    FeedbackSnapshot,
    FeedbackItem,
    Severity,
    JOINT_DISPLAY_NAMES,
)
from src.pose_estimation.data_types import LANDMARK_NAMES
from src.action_comparison.distance_metrics import CORE_JOINT_INDICES, CORE_JOINT_NAMES


# ================================================================
# 颜色常量 (BGR)
# ================================================================

_COLOR_GOOD = (46, 204, 113)       # 绿色 — 姿态正确
_COLOR_WARNING = (44, 156, 243)    # 橙黄 — 中等偏差
_COLOR_ERROR = (52, 73, 231)       # 红色 — 严重偏差
_COLOR_WHITE = (255, 255, 255)
_COLOR_BLACK = (0, 0, 0)
_COLOR_HUD_BG = (20, 20, 20)      # HUD 背景（深色半透明）
_COLOR_ARROW = (0, 220, 255)       # 箭头颜色（亮黄）

# 关节索引 → 英文名 映射
_IDX_TO_NAME = {i: name for i, name in enumerate(LANDMARK_NAMES)}

# 英文名 → 关节索引
_NAME_TO_IDX = {name: i for i, name in enumerate(LANDMARK_NAMES)}


# ================================================================
# 钉住的反馈条目
# ================================================================

@dataclass
class PinnedAdvice:
    """持久化的矫正建议条目"""
    key: str              # 去重键 (joint_name + direction)
    text: str             # 显示文本
    severity: Severity    # 严重等级
    created_at: float     # 创建时间
    expires_at: float     # 过期时间
    joint_name: str = ""  # 关节英文名
    direction: str = ""   # 方向描述


# ================================================================
# 实时视觉反馈管理器
# ================================================================

class RealtimeVisualFeedback:
    """
    实时视觉反馈管理器

    功能：
    1. 在骨骼画面上叠加偏差着色 + 方向箭头 + HUD 文字
    2. 管理持久化建议队列，控制面板刷新频率
    3. 生成分层的 Markdown 面板内容

    使用方式：
        vfb = RealtimeVisualFeedback()
        vfb.update(snapshot)
        annotated = vfb.render_overlay(frame, landmarks)
        markdown = vfb.get_panel_markdown()
    """

    def __init__(
        self,
        pin_duration: float = 3.0,
        max_pinned: int = 5,
        panel_refresh_interval: int = 5,
        max_history: int = 8,
    ):
        """
        Args:
            pin_duration: 建议钉住时长（秒）
            max_pinned: 最大同时钉住的建议数
            panel_refresh_interval: 面板刷新间隔（帧数），非紧急情况下 N 帧才更新一次
            max_history: 历史建议最大保留数
        """
        self._pin_duration = pin_duration
        self._max_pinned = max_pinned
        self._panel_refresh_interval = panel_refresh_interval
        self._max_history = max_history

        # 当前钉住的建议
        self._pinned: List[PinnedAdvice] = []

        # 历史建议记录（用于面板底部的滚动显示）
        self._history: deque = deque(maxlen=max_history)

        # 最近的 snapshot
        self._current_snapshot: Optional[FeedbackSnapshot] = None

        # 帧计数器（用于控制面板刷新频率）
        self._frame_counter: int = 0

        # 上次面板更新时的 markdown 缓存
        self._cached_panel_md: str = "**等待分析…**"

        # 是否有紧急更新（新的 ERROR 级别建议触发立即刷新）
        self._urgent_update: bool = False

        # 当前帧的偏差关节信息（用于骨骼画面渲染）
        self._current_joint_deviations: Dict[int, Tuple[float, Severity, str]] = {}
        # {joint_idx: (deviation, severity, direction)}

        # 整体相似度
        self._current_similarity: float = 0.0

        # 上次整体评级
        self._overall_grade: str = "等待中"
        self._overall_color: str = "⚪"

    def reset(self):
        """重置所有状态（新一轮实时开始时调用）"""
        self._pinned.clear()
        self._history.clear()
        self._current_snapshot = None
        self._frame_counter = 0
        self._cached_panel_md = "**等待分析…**"
        self._urgent_update = False
        self._current_joint_deviations.clear()
        self._current_similarity = 0.0
        self._overall_grade = "等待中"
        self._overall_color = "⚪"

    # ================================================================
    # 核心方法 1: 更新反馈状态
    # ================================================================

    def update(self, snapshot: Optional[FeedbackSnapshot]):
        """
        接收新的反馈快照，更新内部状态

        Args:
            snapshot: RealtimeFeedbackEngine 产出的反馈快照（可为 None）
        """
        self._frame_counter += 1

        if snapshot is None:
            return

        self._current_snapshot = snapshot
        self._current_similarity = snapshot.window_similarity
        now = time.time()

        # 清理过期的钉住建议
        self._pinned = [p for p in self._pinned if p.expires_at > now]

        # 更新整体评级
        self._update_overall_grade(snapshot)

        # 更新偏差关节信息
        self._update_joint_deviations(snapshot)

        # 处理新的建议条目
        if snapshot.has_issues:
            for item in snapshot.items:
                self._process_feedback_item(item, now)

        # 检查是否需要紧急更新
        has_new_error = any(
            item.severity == Severity.ERROR for item in snapshot.items
        )
        if has_new_error:
            self._urgent_update = True

    # ================================================================
    # 核心方法 2: 渲染骨骼画面叠加层
    # ================================================================

    def render_overlay(
        self,
        frame: np.ndarray,
        landmarks: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        在骨骼画面上叠加视觉反馈

        包含：
        1. 偏差关节着色（大圆圈标记）
        2. 方向箭头
        3. 顶部/底部 HUD 文字条

        Args:
            frame: 已绘制骨骼的 BGR 图像
            landmarks: (33, 4) 当前帧关键点（用于确定关节屏幕位置）

        Returns:
            叠加后的 BGR 图像
        """
        if frame is None:
            return frame

        output = frame.copy()
        h, w = output.shape[:2]

        if landmarks is not None and len(self._current_joint_deviations) > 0:
            # 1. 偏差关节高亮圆圈 + 方向箭头
            self._draw_joint_highlights(output, landmarks, w, h)

        # 2. HUD 文字条（底部）
        self._draw_hud(output, w, h)

        # 3. 顶部相似度指示条
        self._draw_similarity_bar(output, w, h)

        return output

    # ================================================================
    # 核心方法 3: 获取面板 Markdown
    # ================================================================

    def get_panel_markdown(self) -> Optional[str]:
        """
        获取右侧面板的 Markdown 内容

        使用降频策略：
        - 非紧急情况下，每 N 帧才真正重新生成内容
        - 有紧急更新（新的 ERROR 建议）时立即生成
        - 第一次有数据时立即生成

        Returns:
            Markdown 字符串，None 表示本帧不需要更新面板
        """
        # 第一次有 snapshot 数据时总是刷新
        first_data = (
            self._current_snapshot is not None
            and self._cached_panel_md == "**等待分析…**"
        )

        should_refresh = (
            first_data
            or self._urgent_update
            or (self._frame_counter % self._panel_refresh_interval == 0)
        )

        if not should_refresh:
            return None  # 不更新，UI 层保持原文不变

        self._urgent_update = False
        self._cached_panel_md = self._build_panel_markdown()
        return self._cached_panel_md

    # ================================================================
    # 内部方法
    # ================================================================

    def _update_overall_grade(self, snapshot: FeedbackSnapshot):
        """更新整体评级"""
        sim = snapshot.window_similarity
        if sim >= 0.85:
            self._overall_grade = "优秀"
            self._overall_color = "🟢"
        elif sim >= 0.70:
            self._overall_grade = "良好"
            self._overall_color = "🟡"
        elif sim >= 0.50:
            self._overall_grade = "需改进"
            self._overall_color = "🟠"
        else:
            self._overall_grade = "偏差较大"
            self._overall_color = "🔴"

    def _update_joint_deviations(self, snapshot: FeedbackSnapshot):
        """从 snapshot 中提取偏差关节信息"""
        self._current_joint_deviations.clear()

        for item in snapshot.items:
            # 通过关节名找索引
            idx = _NAME_TO_IDX.get(item.joint_name, -1)
            if idx < 0:
                # 尝试匹配带后缀的名字
                for name, i in _NAME_TO_IDX.items():
                    if item.joint_name in name or name in item.joint_name:
                        idx = i
                        break
            if idx < 0:
                continue

            # 计算综合偏差值
            dev = item.deviation_deg if item.source == "angle" else item.deviation_spatial * 100
            self._current_joint_deviations[idx] = (
                dev, item.severity, item.direction
            )

    def _process_feedback_item(self, item: FeedbackItem, now: float):
        """处理单个反馈条目，更新钉住队列和历史"""
        # 生成去重键
        key = f"{item.joint_name}_{item.direction}"

        # 构建显示文本
        icon = {
            Severity.ERROR: "🔴",
            Severity.WARNING: "🟡",
            Severity.INFO: "🟢",
        }.get(item.severity, "⚪")

        if item.source == "spatial":
            text = f"{icon} **{item.joint_display}** — {item.advice}（偏移 {item.deviation_spatial:.3f}）"
        else:
            text = f"{icon} **{item.joint_display}** — {item.advice}（偏差 {item.deviation_deg:.0f}°）"

        # 查找是否已有相同 key 的钉住条目
        existing = next((p for p in self._pinned if p.key == key), None)
        if existing:
            # 刷新过期时间
            existing.expires_at = now + self._pin_duration
            existing.text = text
            existing.severity = item.severity
        else:
            # 新增钉住条目
            advice = PinnedAdvice(
                key=key,
                text=text,
                severity=item.severity,
                created_at=now,
                expires_at=now + self._pin_duration,
                joint_name=item.joint_name,
                direction=item.direction,
            )
            self._pinned.append(advice)

            # 超过上限时移除最旧的
            if len(self._pinned) > self._max_pinned:
                self._pinned.sort(key=lambda p: p.created_at)
                self._pinned = self._pinned[-self._max_pinned:]

            # 加入历史记录
            self._history.append(text)

    def _draw_joint_highlights(
        self, frame: np.ndarray, landmarks: np.ndarray, w: int, h: int
    ):
        """在偏差关节处绘制高亮圆圈和方向箭头"""
        for joint_idx, (dev, severity, direction) in self._current_joint_deviations.items():
            if joint_idx >= landmarks.shape[0]:
                continue

            # 关键点坐标（归一化 → 像素）
            lm = landmarks[joint_idx]
            px = int(lm[0] * w)
            py = int(lm[1] * h)
            vis = lm[3] if landmarks.shape[1] > 3 else 1.0

            if vis < 0.3:
                continue

            # 选择颜色
            if severity == Severity.ERROR:
                color = _COLOR_ERROR
                radius = int(max(w, h) * 0.025)
                thickness = 3
            elif severity == Severity.WARNING:
                color = _COLOR_WARNING
                radius = int(max(w, h) * 0.020)
                thickness = 2
            else:
                color = _COLOR_GOOD
                radius = int(max(w, h) * 0.015)
                thickness = 2

            # 绘制高亮圆圈
            cv2.circle(frame, (px, py), radius, color, thickness)

            # 绘制脉冲外圈（更大、更淡）
            if severity in (Severity.ERROR, Severity.WARNING):
                pulse_radius = radius + 5
                cv2.circle(frame, (px, py), pulse_radius, color, 1)

            # 绘制方向箭头
            self._draw_direction_arrow(frame, px, py, direction, color, radius)

    def _draw_direction_arrow(
        self,
        frame: np.ndarray,
        px: int, py: int,
        direction: str,
        color: Tuple[int, int, int],
        radius: int,
    ):
        """在关节位置绘制方向指示箭头"""
        if not direction:
            return

        arrow_len = radius + 12
        arrow_thickness = 2

        # 确定箭头端点
        if "上" in direction or "偏低" in direction:
            end_pt = (px, py - arrow_len)
        elif "下" in direction or "偏高" in direction:
            end_pt = (px, py + arrow_len)
        elif "左" in direction:
            end_pt = (px - arrow_len, py)
        elif "右" in direction:
            end_pt = (px + arrow_len, py)
        else:
            return

        cv2.arrowedLine(
            frame,
            (px, py),
            end_pt,
            _COLOR_ARROW,
            arrow_thickness,
            tipLength=0.4,
        )

    def _draw_hud(self, frame: np.ndarray, w: int, h: int):
        """在画面底部绘制 HUD 文字条"""
        # 获取当前最紧急的 1-2 条建议
        now = time.time()
        active_pinned = [p for p in self._pinned if p.expires_at > now]
        # 按严重度排序
        severity_order = {Severity.ERROR: 0, Severity.WARNING: 1, Severity.INFO: 2}
        active_pinned.sort(key=lambda p: severity_order.get(p.severity, 3))

        if not active_pinned and self._current_snapshot and not self._current_snapshot.has_issues:
            # 姿态正确的提示
            hud_text = "OK! Keep going!"
            hud_color = _COLOR_GOOD
        elif active_pinned:
            # 取最紧急的一条
            top = active_pinned[0]
            # 简化显示文本
            display_name = JOINT_DISPLAY_NAMES.get(
                top.joint_name, top.joint_name
            )
            if top.direction:
                hud_text = f"{display_name}: {top.direction}"
            else:
                hud_text = f"{display_name}: need correction"
            hud_color = {
                Severity.ERROR: _COLOR_ERROR,
                Severity.WARNING: _COLOR_WARNING,
            }.get(top.severity, _COLOR_GOOD)
        else:
            return

        # 绘制底部半透明背景条
        bar_h = 36
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - bar_h), (w, h), _COLOR_HUD_BG, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # 绘制文字
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        text_size = cv2.getTextSize(hud_text, font, font_scale, 1)[0]
        text_x = (w - text_size[0]) // 2
        text_y = h - bar_h + (bar_h + text_size[1]) // 2

        cv2.putText(
            frame, hud_text, (text_x, text_y),
            font, font_scale, hud_color, 1, cv2.LINE_AA
        )

    def _draw_similarity_bar(self, frame: np.ndarray, w: int, h: int):
        """在画面顶部绘制相似度进度条"""
        if self._current_snapshot is None:
            return

        sim = self._current_similarity
        bar_h = 8
        bar_margin = 4

        # 背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (bar_margin, bar_margin),
                      (w - bar_margin, bar_margin + bar_h),
                      (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        # 前景（按相似度长度）
        bar_width = int((w - 2 * bar_margin) * min(sim, 1.0))
        if sim >= 0.85:
            bar_color = _COLOR_GOOD
        elif sim >= 0.70:
            bar_color = _COLOR_WARNING
        else:
            bar_color = _COLOR_ERROR

        if bar_width > 0:
            cv2.rectangle(
                frame,
                (bar_margin, bar_margin),
                (bar_margin + bar_width, bar_margin + bar_h),
                bar_color, -1,
            )

    def _build_panel_markdown(self) -> str:
        """构建分层的面板 Markdown 内容"""
        lines = []
        now = time.time()

        # ===== 顶部：整体评级 =====
        sim_pct = int(self._current_similarity * 100)
        lines.append(
            f"## {self._overall_color} 动作评级：{self._overall_grade}"
        )
        lines.append(f"**实时相似度**: {sim_pct}%")
        lines.append("")

        # ===== 中部：当前钉住的建议（持久显示） =====
        active_pinned = [p for p in self._pinned if p.expires_at > now]
        severity_order = {Severity.ERROR: 0, Severity.WARNING: 1, Severity.INFO: 2}
        active_pinned.sort(key=lambda p: severity_order.get(p.severity, 3))

        if not active_pinned:
            if self._current_snapshot and self._current_snapshot.has_pose:
                lines.append("### ✅ 姿态正确，继续保持！")
                lines.append("")
                lines.append("> 当前动作与标准模板吻合良好")
            else:
                lines.append("### ⚠️ 未检测到姿态")
                lines.append("> 请确保全身在画面中")
        else:
            lines.append("### 📋 矫正建议")
            lines.append("")
            for i, pin in enumerate(active_pinned[:3]):
                # 计算剩余时间指示
                remaining = pin.expires_at - now
                fade = "▓" if remaining > 2.0 else ("▒" if remaining > 1.0 else "░")
                lines.append(f"{fade} {pin.text}")
            lines.append("")

        # ===== 底部：补充信息 =====
        if self._current_snapshot:
            snap = self._current_snapshot
            lines.append("---")
            info_parts = [f"帧 {snap.frame_index}"]
            if snap.dtw_aligned_template_frame >= 0:
                info_parts.append(f"模板帧 {snap.dtw_aligned_template_frame}")
            if snap.spatial_deviation_overall >= 0:
                level = ("轻微" if snap.spatial_deviation_overall < 0.08
                         else "中等" if snap.spatial_deviation_overall < 0.18
                         else "较大")
                info_parts.append(f"空间偏差: {level}")
            lines.append("  ".join(info_parts))

        # ===== 历史记录（最近几条） =====
        if self._history:
            lines.append("")
            lines.append("<details><summary>📜 近期记录</summary>")
            lines.append("")
            for item_text in list(self._history)[-5:]:
                lines.append(f"- {item_text}")
            lines.append("")
            lines.append("</details>")

        return "\n".join(lines)
