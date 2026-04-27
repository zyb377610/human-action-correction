"""
矫正规则引擎

基于关节偏差和角度差异的规则匹配系统，
将底层偏差数据映射为用户可理解的矫正建议。
"""

import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from src.pose_estimation.data_types import LANDMARK_NAMES

from .data_types import CorrectionItem, PRIORITY_HIGH, PRIORITY_MEDIUM, PRIORITY_LOW

logger = logging.getLogger(__name__)

# 关节名称 → 中文名称
JOINT_DISPLAY_NAMES = {
    "nose": "鼻子", "left_eye": "左眼", "right_eye": "右眼",
    "left_ear": "左耳", "right_ear": "右耳",
    "left_shoulder": "左肩", "right_shoulder": "右肩",
    "left_elbow": "左肘", "right_elbow": "右肘",
    "left_wrist": "左腕", "right_wrist": "右腕",
    "left_hip": "左髋", "right_hip": "右髋",
    "left_knee": "左膝", "right_knee": "右膝",
    "left_ankle": "左踝", "right_ankle": "右踝",
    "left_heel": "左脚跟", "right_heel": "右脚跟",
    "left_foot_index": "左脚尖", "right_foot_index": "右脚尖",
}

# 偏差阈值
DEVIATION_THRESHOLD_MODERATE = 0.05
DEVIATION_THRESHOLD_SEVERE = 0.12


@dataclass
class CorrectionRule:
    """
    矫正规则

    Attributes:
        name: 规则名称
        joints: 涉及的关节名称列表
        angle_name: 相关的角度名称（可选）
        condition: 触发条件函数 (deviation, angle_diff) -> bool
        description_template: 偏差描述模板
        advice_template: 矫正建议模板
        priority: 优先级
    """
    name: str
    joints: List[str]
    angle_name: Optional[str] = None
    condition: Callable = None
    description_template: str = ""
    advice_template: str = ""
    priority: str = PRIORITY_MEDIUM

    def check(self, deviation: float, angle_diff: Optional[float] = None) -> bool:
        """检查规则是否触发"""
        if self.condition is None:
            return deviation > DEVIATION_THRESHOLD_MODERATE
        return self.condition(deviation, angle_diff)

    def render(self, **kwargs) -> Tuple[str, str]:
        """
        渲染描述和建议文本

        Args:
            **kwargs: 模板变量 (joint_name, deviation, angle_diff 等)

        Returns:
            (描述文本, 建议文本)
        """
        try:
            desc = self.description_template.format(**kwargs)
        except (KeyError, ValueError):
            desc = self.description_template

        try:
            advice = self.advice_template.format(**kwargs)
        except (KeyError, ValueError):
            advice = self.advice_template

        return desc, advice


class CorrectionRuleEngine:
    """
    矫正规则引擎

    管理各动作类别的矫正规则集，支持规则注册和匹配。

    使用示例:
        engine = CorrectionRuleEngine()
        items = engine.match_rules(
            action="squat",
            joint_deviations={"left_knee": 0.15},
            angle_deviations={"left_knee_angle": (80, 95, -15)},
        )
    """

    def __init__(self, load_builtin: bool = True):
        """
        Args:
            load_builtin: 是否加载内置规则集
        """
        # {动作名称: [规则列表]}
        self._rules: Dict[str, List[CorrectionRule]] = {}
        self._generic_rules: List[CorrectionRule] = []

        if load_builtin:
            self._load_builtin_rules()

    def register_rule(self, action: str, rule: CorrectionRule):
        """注册一条规则到指定动作"""
        if action not in self._rules:
            self._rules[action] = []
        self._rules[action].append(rule)

    def register_action_rules(self, action: str, rules: List[CorrectionRule]):
        """批量注册规则"""
        for rule in rules:
            self.register_rule(action, rule)

    def match_rules(
        self,
        action: str,
        joint_deviations: Dict[str, float],
        angle_deviations: Optional[Dict[str, Tuple[float, float, float]]] = None,
    ) -> List[CorrectionItem]:
        """
        根据偏差数据匹配规则，生成矫正建议

        Args:
            action: 动作类别
            joint_deviations: {关节名: 偏差值}
            angle_deviations: {角度名: (用户角度, 模板角度, 差值)}

        Returns:
            CorrectionItem 列表
        """
        angle_deviations = angle_deviations or {}
        items = []
        matched_joints = set()

        # 1. 匹配动作专属规则
        action_rules = self._rules.get(action, [])
        for rule in action_rules:
            for joint in rule.joints:
                dev = joint_deviations.get(joint, 0.0)
                angle_diff = None

                if rule.angle_name and rule.angle_name in angle_deviations:
                    _, _, angle_diff = angle_deviations[rule.angle_name]

                if rule.check(dev, angle_diff):
                    display_name = JOINT_DISPLAY_NAMES.get(joint, joint)
                    desc, advice = rule.render(
                        joint_name=display_name,
                        deviation=dev,
                        angle_diff=abs(angle_diff) if angle_diff is not None else 0,
                    )
                    items.append(CorrectionItem(
                        joint_name=joint,
                        joint_display_name=display_name,
                        deviation=dev,
                        description=desc,
                        advice=advice,
                        priority=rule.priority,
                        angle_diff=angle_diff,
                    ))
                    matched_joints.add(joint)

        # 2. 兜底规则：对未匹配的偏差关节生成通用建议
        for joint, dev in joint_deviations.items():
            if joint in matched_joints:
                continue
            if dev > DEVIATION_THRESHOLD_SEVERE:
                display_name = JOINT_DISPLAY_NAMES.get(joint, joint)
                items.append(CorrectionItem(
                    joint_name=joint,
                    joint_display_name=display_name,
                    deviation=dev,
                    description=f"{display_name}位置与标准动作存在较大偏差",
                    advice=f"请注意调整{display_name}的位置，使其更接近标准姿势",
                    priority=PRIORITY_MEDIUM,
                ))
            elif dev > DEVIATION_THRESHOLD_MODERATE:
                display_name = JOINT_DISPLAY_NAMES.get(joint, joint)
                items.append(CorrectionItem(
                    joint_name=joint,
                    joint_display_name=display_name,
                    deviation=dev,
                    description=f"{display_name}位置存在轻微偏差",
                    advice=f"可适当注意{display_name}的位置调整",
                    priority=PRIORITY_LOW,
                ))

        return items

    def get_rule_count(self, action: Optional[str] = None) -> int:
        """获取规则数量"""
        if action:
            return len(self._rules.get(action, []))
        return sum(len(rules) for rules in self._rules.values())

    # ===== 内置规则定义 =====

    def _load_builtin_rules(self):
        """加载 5 类动作的内置规则集"""
        self._load_squat_rules()
        self._load_arm_raise_rules()
        self._load_side_bend_rules()
        self._load_lunge_rules()
        self._load_standing_stretch_rules()
        logger.info(f"内置规则加载完成: {self.get_rule_count()} 条规则")

    def _load_squat_rules(self):
        """深蹲矫正规则"""
        rules = [
            CorrectionRule(
                name="squat_knee_not_bent_enough",
                joints=["left_knee", "right_knee"],
                angle_name="left_knee_angle",
                condition=lambda dev, ad: ad is not None and ad > 10,
                description_template="{joint_name}弯曲角度不够",
                advice_template="建议再下蹲，膝关节多弯曲约{angle_diff:.0f}°",
                priority=PRIORITY_HIGH,
            ),
            CorrectionRule(
                name="squat_knee_over_bent",
                joints=["left_knee", "right_knee"],
                angle_name="left_knee_angle",
                condition=lambda dev, ad: ad is not None and ad < -15,
                description_template="{joint_name}弯曲过度",
                advice_template="膝关节弯曲过深，建议适当抬高约{angle_diff:.0f}°",
                priority=PRIORITY_HIGH,
            ),
            CorrectionRule(
                name="squat_hip_not_low",
                joints=["left_hip", "right_hip"],
                angle_name="left_hip_angle",
                condition=lambda dev, ad: ad is not None and abs(ad) > 12,
                description_template="{joint_name}下沉不够或过度",
                advice_template="注意髋部下沉深度，调整约{angle_diff:.0f}°",
                priority=PRIORITY_HIGH,
            ),
            CorrectionRule(
                name="squat_back_lean",
                joints=["left_shoulder", "right_shoulder"],
                condition=lambda dev, ad: dev > 0.08,
                description_template="上半身前倾或后仰幅度偏大",
                advice_template="保持躯干挺直，避免过度前倾",
                priority=PRIORITY_MEDIUM,
            ),
            CorrectionRule(
                name="squat_knees_inward",
                joints=["left_knee", "right_knee"],
                condition=lambda dev, ad: dev > 0.10,
                description_template="{joint_name}有内扣趋势",
                advice_template="注意膝盖方向与脚尖一致，避免内扣",
                priority=PRIORITY_MEDIUM,
            ),
            CorrectionRule(
                name="squat_ankle_stability",
                joints=["left_ankle", "right_ankle"],
                condition=lambda dev, ad: dev > 0.08,
                description_template="{joint_name}稳定性不足",
                advice_template="保持脚掌完全着地，重心均匀分布",
                priority=PRIORITY_LOW,
            ),
            CorrectionRule(
                name="squat_shoulder_uneven",
                joints=["left_shoulder"],
                condition=lambda dev, ad: dev > 0.06,
                description_template="双肩不够平稳",
                advice_template="注意保持双肩水平，身体对称",
                priority=PRIORITY_LOW,
            ),
            CorrectionRule(
                name="squat_head_position",
                joints=["nose"],
                condition=lambda dev, ad: dev > 0.10,
                description_template="头部位置偏差较大",
                advice_template="目视前方，保持头颈自然伸直",
                priority=PRIORITY_LOW,
            ),
        ]
        self.register_action_rules("squat", rules)

    def _load_arm_raise_rules(self):
        """举臂矫正规则"""
        rules = [
            CorrectionRule(
                name="arm_raise_not_high_enough",
                joints=["left_wrist", "right_wrist"],
                angle_name="left_shoulder_angle",
                condition=lambda dev, ad: ad is not None and ad < -15,
                description_template="{joint_name}举臂高度不够",
                advice_template="手臂再抬高约{angle_diff:.0f}°，直至完全伸直过头",
                priority=PRIORITY_HIGH,
            ),
            CorrectionRule(
                name="arm_raise_elbow_bent",
                joints=["left_elbow", "right_elbow"],
                angle_name="left_elbow_angle",
                condition=lambda dev, ad: ad is not None and abs(ad) > 10,
                description_template="{joint_name}未完全伸直",
                advice_template="注意手臂完全伸直，肘关节调整约{angle_diff:.0f}°",
                priority=PRIORITY_HIGH,
            ),
            CorrectionRule(
                name="arm_raise_shoulder_uneven",
                joints=["left_shoulder", "right_shoulder"],
                condition=lambda dev, ad: dev > 0.06,
                description_template="双肩高度不一致",
                advice_template="保持双肩水平，两臂对称抬起",
                priority=PRIORITY_MEDIUM,
            ),
            CorrectionRule(
                name="arm_raise_trunk_lean",
                joints=["left_hip", "right_hip"],
                condition=lambda dev, ad: dev > 0.06,
                description_template="躯干存在倾斜",
                advice_template="保持身体正直，核心收紧",
                priority=PRIORITY_MEDIUM,
            ),
            CorrectionRule(
                name="arm_raise_wrist_position",
                joints=["left_wrist", "right_wrist"],
                condition=lambda dev, ad: dev > 0.10,
                description_template="{joint_name}位置偏差较大",
                advice_template="确保手腕在肩关节正上方",
                priority=PRIORITY_MEDIUM,
            ),
            CorrectionRule(
                name="arm_raise_head_tilt",
                joints=["nose"],
                condition=lambda dev, ad: dev > 0.08,
                description_template="头部位置偏移",
                advice_template="保持目视前方，颈部自然",
                priority=PRIORITY_LOW,
            ),
            CorrectionRule(
                name="arm_raise_knee_locked",
                joints=["left_knee", "right_knee"],
                condition=lambda dev, ad: dev > 0.06,
                description_template="{joint_name}过度锁定或弯曲",
                advice_template="保持膝关节微屈，不要完全锁死",
                priority=PRIORITY_LOW,
            ),
            CorrectionRule(
                name="arm_raise_foot_position",
                joints=["left_ankle", "right_ankle"],
                condition=lambda dev, ad: dev > 0.08,
                description_template="脚部位置偏差",
                advice_template="双脚与肩同宽，平稳站立",
                priority=PRIORITY_LOW,
            ),
        ]
        self.register_action_rules("arm_raise", rules)

    def _load_side_bend_rules(self):
        """侧弯矫正规则"""
        rules = [
            CorrectionRule(
                name="side_bend_not_enough",
                joints=["left_shoulder", "right_shoulder"],
                angle_name="trunk_angle",
                condition=lambda dev, ad: ad is not None and abs(ad) > 10,
                description_template="侧弯幅度不足或过度",
                advice_template="调整躯干侧弯角度约{angle_diff:.0f}°",
                priority=PRIORITY_HIGH,
            ),
            CorrectionRule(
                name="side_bend_hip_shift",
                joints=["left_hip", "right_hip"],
                condition=lambda dev, ad: dev > 0.08,
                description_template="{joint_name}侧移偏大",
                advice_template="保持髋部固定，仅躯干侧弯",
                priority=PRIORITY_HIGH,
            ),
            CorrectionRule(
                name="side_bend_shoulder_rotation",
                joints=["left_shoulder", "right_shoulder"],
                condition=lambda dev, ad: dev > 0.10,
                description_template="肩部存在旋转",
                advice_template="保持双肩在同一平面内侧弯，避免旋转",
                priority=PRIORITY_MEDIUM,
            ),
            CorrectionRule(
                name="side_bend_arm_position",
                joints=["left_elbow", "right_elbow"],
                angle_name="left_elbow_angle",
                condition=lambda dev, ad: dev > 0.08,
                description_template="{joint_name}位置偏差",
                advice_template="引导臂保持伸直，另一手叉腰",
                priority=PRIORITY_MEDIUM,
            ),
            CorrectionRule(
                name="side_bend_knee_bend",
                joints=["left_knee", "right_knee"],
                condition=lambda dev, ad: dev > 0.06,
                description_template="{joint_name}弯曲",
                advice_template="保持双腿伸直，不要弯膝",
                priority=PRIORITY_LOW,
            ),
            CorrectionRule(
                name="side_bend_head_alignment",
                joints=["nose"],
                condition=lambda dev, ad: dev > 0.08,
                description_template="头部未跟随躯干侧弯",
                advice_template="头部自然跟随身体侧弯方向",
                priority=PRIORITY_LOW,
            ),
            CorrectionRule(
                name="side_bend_foot_stable",
                joints=["left_ankle", "right_ankle"],
                condition=lambda dev, ad: dev > 0.06,
                description_template="脚部不够稳定",
                advice_template="双脚与肩同宽站稳，不要移动",
                priority=PRIORITY_LOW,
            ),
            CorrectionRule(
                name="side_bend_wrist_position",
                joints=["left_wrist", "right_wrist"],
                condition=lambda dev, ad: dev > 0.10,
                description_template="{joint_name}位置偏差较大",
                advice_template="注意手臂的引导方向",
                priority=PRIORITY_LOW,
            ),
        ]
        self.register_action_rules("side_bend", rules)

    def _load_lunge_rules(self):
        """弓步矫正规则"""
        rules = [
            CorrectionRule(
                name="lunge_front_knee_angle",
                joints=["left_knee"],
                angle_name="left_knee_angle",
                condition=lambda dev, ad: ad is not None and abs(ad) > 10,
                description_template="前腿膝关节角度偏差",
                advice_template="前腿膝关节应约90°，调整约{angle_diff:.0f}°",
                priority=PRIORITY_HIGH,
            ),
            CorrectionRule(
                name="lunge_back_knee_low",
                joints=["right_knee"],
                angle_name="right_knee_angle",
                condition=lambda dev, ad: ad is not None and abs(ad) > 12,
                description_template="后腿膝关节高度偏差",
                advice_template="后膝接近地面，调整约{angle_diff:.0f}°",
                priority=PRIORITY_HIGH,
            ),
            CorrectionRule(
                name="lunge_hip_alignment",
                joints=["left_hip", "right_hip"],
                angle_name="left_hip_angle",
                condition=lambda dev, ad: ad is not None and abs(ad) > 10,
                description_template="髋部角度偏差",
                advice_template="保持髋部正对前方，调整约{angle_diff:.0f}°",
                priority=PRIORITY_HIGH,
            ),
            CorrectionRule(
                name="lunge_trunk_upright",
                joints=["left_shoulder", "right_shoulder"],
                condition=lambda dev, ad: dev > 0.08,
                description_template="上半身未保持直立",
                advice_template="躯干保持挺直，不要前倾",
                priority=PRIORITY_MEDIUM,
            ),
            CorrectionRule(
                name="lunge_front_knee_over_toe",
                joints=["left_knee", "left_ankle"],
                condition=lambda dev, ad: dev > 0.10,
                description_template="前膝可能超过脚尖",
                advice_template="注意前膝不要超过脚尖，保持垂直",
                priority=PRIORITY_MEDIUM,
            ),
            CorrectionRule(
                name="lunge_balance",
                joints=["left_ankle", "right_ankle"],
                condition=lambda dev, ad: dev > 0.08,
                description_template="脚部稳定性不足",
                advice_template="保持身体平衡，双脚稳定着地",
                priority=PRIORITY_MEDIUM,
            ),
            CorrectionRule(
                name="lunge_arm_position",
                joints=["left_wrist", "right_wrist"],
                condition=lambda dev, ad: dev > 0.10,
                description_template="手臂位置偏差",
                advice_template="双手自然垂放或叉腰",
                priority=PRIORITY_LOW,
            ),
            CorrectionRule(
                name="lunge_head_position",
                joints=["nose"],
                condition=lambda dev, ad: dev > 0.08,
                description_template="头部位置偏差",
                advice_template="目视前方，颈部自然伸直",
                priority=PRIORITY_LOW,
            ),
        ]
        self.register_action_rules("lunge", rules)

    def _load_standing_stretch_rules(self):
        """站立拉伸矫正规则"""
        rules = [
            CorrectionRule(
                name="stretch_arm_not_high",
                joints=["left_wrist", "right_wrist"],
                angle_name="left_shoulder_angle",
                condition=lambda dev, ad: ad is not None and abs(ad) > 12,
                description_template="手臂上举高度不够",
                advice_template="双臂再向上伸展，肩关节调整约{angle_diff:.0f}°",
                priority=PRIORITY_HIGH,
            ),
            CorrectionRule(
                name="stretch_elbow_straight",
                joints=["left_elbow", "right_elbow"],
                angle_name="left_elbow_angle",
                condition=lambda dev, ad: ad is not None and abs(ad) > 10,
                description_template="{joint_name}未完全伸直",
                advice_template="手臂伸直，肘关节调整约{angle_diff:.0f}°",
                priority=PRIORITY_HIGH,
            ),
            CorrectionRule(
                name="stretch_body_extension",
                joints=["left_shoulder", "right_shoulder"],
                condition=lambda dev, ad: dev > 0.08,
                description_template="身体延伸不够充分",
                advice_template="全身向上延伸，感受脊柱拉伸",
                priority=PRIORITY_MEDIUM,
            ),
            CorrectionRule(
                name="stretch_hip_stable",
                joints=["left_hip", "right_hip"],
                condition=lambda dev, ad: dev > 0.06,
                description_template="髋部不够稳定",
                advice_template="保持髋部水平，双腿站稳",
                priority=PRIORITY_MEDIUM,
            ),
            CorrectionRule(
                name="stretch_knee_straight",
                joints=["left_knee", "right_knee"],
                condition=lambda dev, ad: dev > 0.06,
                description_template="{joint_name}弯曲",
                advice_template="保持膝关节伸直但不锁死",
                priority=PRIORITY_LOW,
            ),
            CorrectionRule(
                name="stretch_shoulder_even",
                joints=["left_shoulder", "right_shoulder"],
                condition=lambda dev, ad: dev > 0.06,
                description_template="双肩不对称",
                advice_template="保持双肩同时向上，对称发力",
                priority=PRIORITY_LOW,
            ),
            CorrectionRule(
                name="stretch_head_up",
                joints=["nose"],
                condition=lambda dev, ad: dev > 0.08,
                description_template="头部位置偏差",
                advice_template="头部自然向上延伸，目视前上方",
                priority=PRIORITY_LOW,
            ),
            CorrectionRule(
                name="stretch_foot_stable",
                joints=["left_ankle", "right_ankle"],
                condition=lambda dev, ad: dev > 0.06,
                description_template="脚部位置偏差",
                advice_template="双脚与肩同宽，脚掌完全着地",
                priority=PRIORITY_LOW,
            ),
        ]
        self.register_action_rules("standing_stretch", rules)