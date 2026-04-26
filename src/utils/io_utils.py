"""
数据序列化辅助工具

提供 JSON 编码器和通用的数据 I/O 工具函数。
"""

import json
from pathlib import Path
from typing import Any

import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """
    支持 NumPy 类型的 JSON 编码器

    自动将 NumPy 数组和标量转换为 Python 原生类型。
    """

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return super().default(obj)


def save_json(data: Any, path: str, indent: int = 2):
    """
    保存数据为 JSON 文件（支持 NumPy 类型）

    Args:
        data: 要保存的数据
        path: 文件路径
        indent: 缩进空格数
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(str(path), "w", encoding="utf-8") as f:
        json.dump(data, f, cls=NumpyEncoder, ensure_ascii=False, indent=indent)


def load_json(path: str) -> Any:
    """
    从 JSON 文件加载数据

    Args:
        path: 文件路径

    Returns:
        解析后的数据
    """
    with open(str(path), "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: str) -> Path:
    """
    确保目录存在，不存在则创建

    Args:
        path: 目录路径

    Returns:
        Path 对象
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
