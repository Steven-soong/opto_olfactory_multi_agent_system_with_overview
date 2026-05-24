import os
import json
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """固定随机种子，方便复现实验。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    """创建目录。"""
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(data: Dict[str, Any], path: str) -> None:
    """保存 JSON 报告。"""
    parent = os.path.dirname(path)
    if parent:
        ensure_dir(parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def tensor_to_float_list(x: torch.Tensor):
    return [float(v) for v in x.detach().cpu().flatten()]
