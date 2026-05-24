from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import SystemConfig


# 传感器响应剖面：不同化学类别在 8 个电子鼻通道上的相对响应强度
E_NOSE_RESPONSE: Dict[str, np.ndarray] = {
    "safe": np.array([0.05, 0.04, 0.03, 0.03, 0.02, 0.03, 0.02, 0.02]),
    "methanol": np.array([0.90, 0.75, 0.50, 0.20, 0.15, 0.20, 0.10, 0.08]),
    "ammonia": np.array([0.15, 0.25, 0.85, 0.90, 0.40, 0.20, 0.15, 0.10]),
    "hf_like": np.array([0.10, 0.12, 0.20, 0.35, 0.90, 0.80, 0.35, 0.20]),
    "mixed_organic": np.array([0.75, 0.85, 0.55, 0.35, 0.25, 0.30, 0.70, 0.55]),
}


# 光谱图像中不同类别的模拟响应中心点
SPECTRAL_CENTERS: Dict[str, Tuple[Tuple[int, int], ...]] = {
    "safe": ((8, 8),),
    "methanol": ((9, 21), (13, 24)),
    "ammonia": ((22, 9), (25, 13)),
    "hf_like": ((17, 17), (20, 19)),
    "mixed_organic": ((8, 22), (22, 22), (16, 10)),
}


class SyntheticOlfactoryDataGenerator:
    """
    生成模拟光电融合数据。

    输出：
    - time_signal: shape = [num_sensors, seq_len]
    - spectral_image: shape = [1, image_size, image_size]
    - env: [temperature, humidity]，归一化到大致 0~1
    - label_id
    """

    def __init__(self, config: SystemConfig):
        self.cfg = config
        self.class_names = list(config.class_names)

    def _generate_time_signal(self, class_name: str, temperature: float, humidity: float) -> np.ndarray:
        t = np.linspace(0, 1, self.cfg.seq_len)
        signal = np.random.normal(0, 0.025, size=(self.cfg.num_sensors, self.cfg.seq_len))

        response = E_NOSE_RESPONSE[class_name].copy()

        # 温湿度交叉干扰：湿度偏高时，部分通道漂移更明显
        humidity_factor = 1.0 + 0.45 * max(humidity - 0.55, 0)
        temp_factor = 1.0 + 0.25 * max(temperature - 0.55, 0)
        response = response * humidity_factor * temp_factor

        for i in range(self.cfg.num_sensors):
            rise = 1 / (1 + np.exp(-25 * (t - 0.28)))
            decay = np.exp(-1.8 * np.maximum(t - 0.55, 0))
            dynamic_curve = response[i] * rise * decay

            # 低频漂移 + 高频噪声
            drift = 0.025 * np.sin(2 * np.pi * (i + 1) * t / 6)
            noise = np.random.normal(0, 0.018, size=self.cfg.seq_len)
            signal[i] += dynamic_curve + drift + noise

        return signal.astype(np.float32)

    def _generate_spectral_image(self, class_name: str) -> np.ndarray:
        size = self.cfg.image_size
        yy, xx = np.mgrid[0:size, 0:size]
        image = np.random.normal(0, 0.02, size=(size, size))

        centers = SPECTRAL_CENTERS[class_name]
        for cx, cy in centers:
            amplitude = np.random.uniform(0.55, 1.0) if class_name != "safe" else np.random.uniform(0.05, 0.15)
            sigma = np.random.uniform(2.0, 4.2)
            blob = amplitude * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
            image += blob

        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        return image[None, :, :].astype(np.float32)

    def generate_one(self, label_id: int = None):
        if label_id is None:
            label_id = np.random.randint(0, len(self.class_names))
        class_name = self.class_names[label_id]

        # 归一化温度/湿度，用于模型和推理 Agent
        # temperature_norm 大致对应 15~45 ℃，humidity_norm 对应 20%~90% RH
        temperature_norm = np.random.uniform(0.2, 0.9)
        humidity_norm = np.random.uniform(0.2, 0.95)

        time_signal = self._generate_time_signal(class_name, temperature_norm, humidity_norm)
        spectral_image = self._generate_spectral_image(class_name)
        env = np.array([temperature_norm, humidity_norm], dtype=np.float32)

        return time_signal, spectral_image, env, label_id


class SyntheticOlfactoryDataset(Dataset):
    def __init__(self, config: SystemConfig, num_samples: int = 800):
        self.cfg = config
        self.generator = SyntheticOlfactoryDataGenerator(config)
        self.samples = [self.generator.generate_one() for _ in range(num_samples)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        time_signal, spectral_image, env, label_id = self.samples[idx]
        return {
            "time_signal": torch.tensor(time_signal, dtype=torch.float32),
            "spectral_image": torch.tensor(spectral_image, dtype=torch.float32),
            "env": torch.tensor(env, dtype=torch.float32),
            "label": torch.tensor(label_id, dtype=torch.long),
        }
