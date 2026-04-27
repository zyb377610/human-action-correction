"""
矫正反馈模块单元测试

覆盖: 角度计算、规则引擎、反馈生成、报告序列化
"""

import json
import pytest
import numpy as np

from src.correction.data_types import (
    CorrectionItem, CorrectionReport,
    PRIORITY_HIGH, PRIORITY_MEDIUM, PRIORITY_LOW,
)
from src.correction.angle_utils import AngleCalculator, compute_angle_3d
from src.correction.rules import CorrectionRule, CorrectionRuleEngine
from src.correction.feedback import FeedbackGenerator
from src.action_comparison.deviation_analyzer import DeviationReport


# ===== 角度计算测试 =====

class TestAngleCalculation:
    def test_right_angle(self):
        """直角 = 90°"""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 0.0, 0.0])
        c = np.array([0.0, 1.0, 0.0])
        angle = compute_angle_3d(a, b, c)
        assert abs(angle - 90.0) < 0.1

    def test_straight_angle(self):
        """平角 = 180°"""
        a = np.array([-1.0, 0.0, 0.0])
        b = np.array([0.0, 0.0, 0.0])
        c = np.array([1.0, 0.0, 0.0])
        angle = compute_angle_3d(a, b, c)
        assert abs(angle - 180.0) < 0.1

    def test_zero_angle(self):
        """零角 = 0°"""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 0.0, 0.0])
        c = np.array([1.0, 0.0, 0.0])
        angle = compute_angle_3d(a, b, c)
        assert abs(angle - 0.0) < 0.1

    def test_45_degree(self):
        """45 度角"""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 0.0, 0.0])
        c = np.array([1.0, 1.0, 0.0])
        angle = compute_angle_3d(a, b, c)
        assert abs(angle - 45.0) < 0.1

    def test_degenerate_zero_length(self):
        """退化情况：零向量"""
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([0.0, 0.0, 0.0])
        c = np.array([1.0, 0.0, 0.0])
        angle = compute_angle_3d(a, b, c)
        assert angle == 0.0

    def test_3d_angle(self):
        """3D 角度计算"""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 0.0, 0.0])
        c = np.array([0.0, 0.0, 1.0])
        angle = compute_angle_3d(a, b, c)
        assert abs(angle - 90.0) < 0.1


class TestAngleCalculator:
    def test_compute_frame_angles(self):
        """计算单帧角度"""
        calc = AngleCalculator()
        # 创建一个简单的 landmarks (33, 4) 数组
        landmarks = np.random.rand(33, 4).astype(np.float32)
        angles = calc.compute_frame_angles(landmarks)
        assert len(angles) > 0
        for name, val in angles.items():
            assert 0 <= val <= 180

    def test_display_name(self):
        assert AngleCalculator.get_display_name("left_knee_angle") == "左膝关节角"
        assert AngleCalculator.get_display_name("unknown") == "unknown"


# ===== 规则引擎测试 =====

class TestCorrectionRule:
    def test_rule_check_default(self):
        """默认条件：偏差 > 0.05"""
        rule = CorrectionRule(
            name="test", joints=["left_knee"],
            description_template="test", advice_template="test",
        )
        assert rule.check(0.06) is True
        assert rule.check(0.03) is False

    def test_rule_check_custom(self):
        """自定义条件"""
        rule = CorrectionRule(
            name="test", joints=["left_knee"],
            condition=lambda dev, ad: ad is not None and ad > 10,
            description_template="test", advice_template="test",
        )
        assert rule.check(0.01, 15) is True
        assert rule.check(0.01, 5) is False
        assert rule.check(0.01, None) is False

    def test_rule_render(self):
        """文本渲染"""
        rule = CorrectionRule(
            name="test", joints=["left_knee"],
            description_template="{joint_name}弯曲不够",
            advice_template="调整约{angle_diff:.0f}°",
        )
        desc, advice = rule.render(joint_name="左膝", angle_diff=12.3)
        assert desc == "左膝弯曲不够"
        assert advice == "调整约12°"


class TestCorrectionRuleEngine:
    def test_builtin_rules_loaded(self):
        """内置规则加载"""
        engine = CorrectionRuleEngine()
        assert engine.get_rule_count("squat") >= 8
        assert engine.get_rule_count("arm_raise") >= 8
        assert engine.get_rule_count("side_bend") >= 8
        assert engine.get_rule_count("lunge") >= 8
        assert engine.get_rule_count("standing_stretch") >= 8

    def test_match_rules_with_deviation(self):
        """偏差触发规则匹配"""
        engine = CorrectionRuleEngine()
        items = engine.match_rules(
            action="squat",
            joint_deviations={"left_knee": 0.15, "right_knee": 0.15},
        )
        assert len(items) > 0

    def test_match_rules_with_angle(self):
        """角度偏差触发规则"""
        engine = CorrectionRuleEngine()
        # angle_diff=15 (>10) 触发 squat_knee_not_bent_enough
        items = engine.match_rules(
            action="squat",
            joint_deviations={"left_knee": 0.15},
            angle_deviations={"left_knee_angle": (80, 95, 15)},
        )
        # 应包含角度相关的建议
        assert any(item.angle_diff is not None for item in items)

    def test_fallback_rules(self):
        """兜底规则：未注册动作也能产生建议"""
        engine = CorrectionRuleEngine()
        items = engine.match_rules(
            action="unknown_action",
            joint_deviations={"left_knee": 0.15},
        )
        assert len(items) > 0  # 兜底建议

    def test_register_custom_rule(self):
        """注册自定义规则"""
        engine = CorrectionRuleEngine(load_builtin=False)
        rule = CorrectionRule(
            name="custom", joints=["nose"],
            condition=lambda dev, ad: dev > 0.01,
            description_template="自定义描述",
            advice_template="自定义建议",
        )
        engine.register_rule("test_action", rule)
        assert engine.get_rule_count("test_action") == 1

        items = engine.match_rules("test_action", {"nose": 0.05})
        assert len(items) >= 1
        assert items[0].description == "自定义描述"

    def test_no_deviation_no_match(self):
        """无偏差时不触发规则"""
        engine = CorrectionRuleEngine()
        items = engine.match_rules(
            action="squat",
            joint_deviations={"left_knee": 0.01},
        )
        # 可能只有极少兜底（因为 0.01 < moderate 阈值）
        assert all(item.deviation <= 0.01 or item.priority == PRIORITY_LOW for item in items) or len(items) == 0


# ===== 反馈生成测试 =====

class TestFeedbackGenerator:
    def _make_deviation_report(self, overall=0.08, severity="moderate"):
        return DeviationReport(
            joint_deviations={"left_knee": 0.15, "right_knee": 0.10, "nose": 0.02},
            worst_joints=["left_knee", "right_knee"],
            frame_deviations=np.array([0.1, 0.2, 0.15]),
            severity=severity,
            overall_deviation=overall,
            worst_joint_details=[
                {"name": "left_knee", "index": 25, "deviation": 0.15},
            ],
        )

    def test_generate_report(self):
        """生成完整报告"""
        gen = FeedbackGenerator()
        report = gen.generate(
            action_name="squat",
            deviation_report=self._make_deviation_report(),
            similarity=0.75,
            quality_score=72.0,
        )
        assert isinstance(report, CorrectionReport)
        assert report.action_name == "squat"
        assert report.quality_score == 72.0
        assert len(report.overall_comment) > 0

    def test_comment_excellent(self):
        assert "优秀" in FeedbackGenerator._generate_comment(95)

    def test_comment_good(self):
        assert "良好" in FeedbackGenerator._generate_comment(75)

    def test_comment_needs_improvement(self):
        assert "偏差" in FeedbackGenerator._generate_comment(55)

    def test_comment_poor(self):
        assert "较大" in FeedbackGenerator._generate_comment(40)

    def test_corrections_sorted(self):
        """建议按优先级排序"""
        gen = FeedbackGenerator()
        report = gen.generate(
            action_name="squat",
            deviation_report=self._make_deviation_report(),
            angle_deviations={"left_knee_angle": (80, 95, -15)},
            similarity=0.65,
            quality_score=60.0,
        )
        if len(report.corrections) > 1:
            from src.correction.data_types import PRIORITY_ORDER
            for i in range(len(report.corrections) - 1):
                a = report.corrections[i]
                b = report.corrections[i + 1]
                assert PRIORITY_ORDER.get(a.priority, 99) <= PRIORITY_ORDER.get(b.priority, 99)


# ===== 报告序列化测试 =====

class TestCorrectionReport:
    def test_to_text(self):
        report = CorrectionReport(
            action_name="squat",
            action_display_name="深蹲",
            quality_score=85.0,
            similarity=0.85,
            overall_comment="动作整体良好",
            severity="mild",
            corrections=[
                CorrectionItem(
                    joint_name="left_knee",
                    joint_display_name="左膝",
                    deviation=0.15,
                    description="弯曲不够",
                    advice="再下蹲约12°",
                    priority=PRIORITY_HIGH,
                    angle_diff=-12.0,
                ),
            ],
        )
        text = report.to_text()
        assert "深蹲" in text
        assert "85.0" in text
        assert "左膝" in text

    def test_to_dict(self):
        report = CorrectionReport(
            action_name="squat",
            quality_score=85.0,
            similarity=0.85,
            overall_comment="test",
            corrections=[],
            angle_deviations={"left_knee_angle": (80.0, 95.0, -15.0)},
        )
        d = report.to_dict()
        assert d["action_name"] == "squat"
        assert d["quality_score"] == 85.0
        # 可以 JSON 序列化
        json_str = json.dumps(d, ensure_ascii=False)
        assert len(json_str) > 0

    def test_to_json(self):
        report = CorrectionReport(action_name="test", quality_score=50.0)
        j = report.to_json()
        data = json.loads(j)
        assert data["action_name"] == "test"

    def test_properties(self):
        report = CorrectionReport(
            corrections=[
                CorrectionItem("a", "A", 0.1, "d", "a", PRIORITY_HIGH),
                CorrectionItem("b", "B", 0.2, "d", "a", PRIORITY_MEDIUM),
                CorrectionItem("c", "C", 0.3, "d", "a", PRIORITY_HIGH),
            ],
        )
        assert report.num_corrections == 3
        assert report.high_priority_count == 2