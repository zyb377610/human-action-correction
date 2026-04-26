"""
标准动作模板库管理

基于文件系统管理动作类别和标准模板，支持增删查操作。
模板存储结构: data/templates/<action_name>/template_XX.json
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from src.pose_estimation.data_types import PoseSequence
from src.utils.config import get_config, PROJECT_ROOT

logger = logging.getLogger(__name__)

# 默认模板目录
DEFAULT_TEMPLATES_DIR = PROJECT_ROOT / "data" / "templates"
METADATA_FILE = "metadata.json"


class TemplateLibrary:
    """
    标准动作模板库

    管理动作类别和对应的标准模板文件。

    使用示例:
        lib = TemplateLibrary()
        lib.add_template("squat", sequence, "standard_01")
        templates = lib.list_templates("squat")
        seq = lib.load_template("squat", "standard_01")
    """

    def __init__(self, templates_dir: Optional[str] = None):
        """
        Args:
            templates_dir: 模板根目录，None 时使用默认 data/templates/
        """
        if templates_dir:
            self._root = Path(templates_dir)
        else:
            self._root = DEFAULT_TEMPLATES_DIR

        self._root.mkdir(parents=True, exist_ok=True)
        self._metadata_path = self._root / METADATA_FILE

        # 加载或初始化元信息
        if self._metadata_path.exists():
            with open(self._metadata_path, "r", encoding="utf-8") as f:
                self._metadata = json.load(f)
        else:
            self._metadata = {"actions": {}}
            self._save_metadata()

    def _save_metadata(self):
        """保存元信息"""
        with open(self._metadata_path, "w", encoding="utf-8") as f:
            json.dump(self._metadata, f, ensure_ascii=False, indent=2)

    # ===== 动作类别管理 =====

    def add_action(self, name: str, display_name: str = "", description: str = ""):
        """
        添加动作类别

        Args:
            name: 动作标识名（英文，如 squat）
            display_name: 显示名称（如"深蹲"）
            description: 动作描述
        """
        action_dir = self._root / name
        action_dir.mkdir(parents=True, exist_ok=True)

        if name not in self._metadata["actions"]:
            self._metadata["actions"][name] = {
                "display_name": display_name or name,
                "description": description,
                "template_count": 0,
            }
            self._save_metadata()
            logger.info(f"添加动作类别: {name} ({display_name})")

    def list_actions(self) -> List[str]:
        """列出所有动作类别名称"""
        return list(self._metadata["actions"].keys())

    def get_action_info(self, name: str) -> Optional[Dict]:
        """获取动作类别信息"""
        return self._metadata["actions"].get(name)

    # ===== 模板管理 =====

    def add_template(
        self, action: str, sequence: PoseSequence, template_name: str
    ):
        """
        保存一个标准模板

        Args:
            action: 动作类别名称
            sequence: 姿态序列
            template_name: 模板名称（不含扩展名）
        """
        if action not in self._metadata["actions"]:
            self.add_action(action)

        action_dir = self._root / action
        action_dir.mkdir(parents=True, exist_ok=True)

        path = action_dir / f"{template_name}.json"
        sequence.save(str(path))

        # 更新计数
        count = len(list(action_dir.glob("*.json")))
        self._metadata["actions"][action]["template_count"] = count
        self._save_metadata()

        logger.info(f"保存模板: {action}/{template_name} ({sequence.num_frames} 帧)")

    def load_template(self, action: str, template_name: str) -> PoseSequence:
        """
        加载一个标准模板

        Args:
            action: 动作类别名称
            template_name: 模板名称（不含扩展名）

        Returns:
            PoseSequence
        """
        path = self._root / action / f"{template_name}.json"
        if not path.exists():
            raise FileNotFoundError(f"模板不存在: {path}")
        return PoseSequence.load(str(path))

    def list_templates(self, action: str) -> List[str]:
        """列出某个动作类别下的所有模板名称"""
        action_dir = self._root / action
        if not action_dir.exists():
            return []
        return [p.stem for p in sorted(action_dir.glob("*.json"))]

    def delete_template(self, action: str, template_name: str):
        """删除一个模板"""
        path = self._root / action / f"{template_name}.json"
        if path.exists():
            path.unlink()
            count = len(list((self._root / action).glob("*.json")))
            if action in self._metadata["actions"]:
                self._metadata["actions"][action]["template_count"] = count
                self._save_metadata()
            logger.info(f"删除模板: {action}/{template_name}")

    def load_all_templates(self, action: str) -> Dict[str, PoseSequence]:
        """加载某个动作类别下的所有模板"""
        templates = {}
        for name in self.list_templates(action):
            templates[name] = self.load_template(action, name)
        return templates

    def __repr__(self) -> str:
        actions = self.list_actions()
        return f"TemplateLibrary(root='{self._root}', actions={actions})"
