from dataclasses import dataclass
from typing import Tuple


@dataclass
class SystemConfig:
    """系统基础配置。"""

    seed: int = 42

    # 电子鼻时序信号维度：8 个传感通道，每个通道 128 个采样点
    num_sensors: int = 8
    seq_len: int = 128

    # 光谱图像尺寸：单通道 32 x 32
    image_size: int = 32

    # 类别定义：safe 表示无明显危险挥发物
    class_names: Tuple[str, ...] = (
        "safe",
        "methanol",
        "ammonia",
        "hf_like",
        "mixed_organic",
    )

    # 模型参数
    time_feature_dim: int = 64
    image_feature_dim: int = 64
    hidden_dim: int = 128

    # 训练参数
    batch_size: int = 32
    epochs: int = 5
    learning_rate: float = 1e-3

    # 报警阈值
    warning_threshold: float = 0.55
    danger_threshold: float = 0.75

    # 路径
    checkpoint_path: str = "checkpoints/fusion_model.pt"
    report_path: str = "outputs/demo_report.json"
