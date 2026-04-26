"""
配置加载工具模块

从 YAML 配置文件中读取系统配置，提供字典式访问接口。
支持默认配置加载和自定义配置覆盖。
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


# 项目根目录（从 src/utils/config.py 向上两级）
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# 默认配置文件路径
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


class Config:
    """
    系统配置管理类

    支持从 YAML 文件加载配置，提供字典式和属性式访问。

    使用示例:
        config = Config()  # 加载默认配置
        config = Config("configs/custom.yaml")  # 加载自定义配置

        # 字典式访问
        backend = config["pose_estimation"]["backend"]

        # 获取嵌套配置段
        pose_cfg = config.get_section("pose_estimation")
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置

        Args:
            config_path: 配置文件路径。为 None 时加载默认配置 configs/default.yaml
        """
        if config_path is None:
            self._path = DEFAULT_CONFIG_PATH
        else:
            self._path = Path(config_path)
            if not self._path.is_absolute():
                self._path = PROJECT_ROOT / self._path

        if not self._path.exists():
            raise FileNotFoundError(f"配置文件不存在: {self._path}")

        with open(self._path, "r", encoding="utf-8") as f:
            self._data: Dict[str, Any] = yaml.safe_load(f) or {}

    def __getitem__(self, key: str) -> Any:
        """字典式访问配置项"""
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        """检查配置项是否存在"""
        return key in self._data

    def get(self, key: str, default: Any = None) -> Any:
        """
        安全获取配置项，不存在时返回默认值

        Args:
            key: 配置键名
            default: 默认值
        """
        return self._data.get(key, default)

    def get_section(self, section: str) -> Dict[str, Any]:
        """
        获取配置段（子字典）

        Args:
            section: 配置段名称，如 "pose_estimation"

        Returns:
            配置段字典，不存在时返回空字典
        """
        result = self._data.get(section, {})
        return result if isinstance(result, dict) else {}

    def to_dict(self) -> Dict[str, Any]:
        """返回完整配置的字典副本"""
        return dict(self._data)

    @property
    def path(self) -> Path:
        """配置文件路径"""
        return self._path

    def __repr__(self) -> str:
        return f"Config(path='{self._path}')"


# 全局默认配置实例（延迟初始化）
_default_config: Optional[Config] = None


def get_config(config_path: Optional[str] = None) -> Config:
    """
    获取配置实例

    首次调用时创建实例，后续调用返回同一实例（单例模式）。
    传入 config_path 时总是创建新实例。

    Args:
        config_path: 配置文件路径，为 None 时使用默认配置

    Returns:
        Config 实例
    """
    global _default_config

    if config_path is not None:
        return Config(config_path)

    if _default_config is None:
        _default_config = Config()

    return _default_config
