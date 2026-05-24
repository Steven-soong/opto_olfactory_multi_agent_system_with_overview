import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import SystemConfig


class ResidualBlock1D(nn.Module):
    """一维残差块，用于电子鼻高频时序信号。"""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class TimeSeriesEncoder(nn.Module):
    """电子鼻时序分支 CNN。"""

    def __init__(self, num_sensors: int, feature_dim: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(num_sensors, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.res1 = ResidualBlock1D(32)
        self.down = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.res2 = ResidualBlock1D(64)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, feature_dim)

    def forward(self, x):
        x = self.stem(x)
        x = self.res1(x)
        x = self.down(x)
        x = self.res2(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


class ResidualBlock2D(nn.Module):
    """二维残差块，用于光谱图像分支。"""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class SpectralImageEncoder(nn.Module):
    """光谱图像分支 CNN。"""

    def __init__(self, feature_dim: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.res1 = ResidualBlock2D(24)
        self.down = nn.Sequential(
            nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
        )
        self.res2 = ResidualBlock2D(48)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(48, feature_dim)

    def forward(self, x):
        x = self.stem(x)
        x = self.res1(x)
        x = self.down(x)
        x = self.res2(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


class FusionNet(nn.Module):
    """
    光电融合分类网络。

    输入：
    - time_signal: [B, num_sensors, seq_len]
    - spectral_image: [B, 1, H, W]
    - env: [B, 2]，温度和湿度
    """

    def __init__(self, cfg: SystemConfig):
        super().__init__()
        self.cfg = cfg
        self.time_encoder = TimeSeriesEncoder(cfg.num_sensors, cfg.time_feature_dim)
        self.image_encoder = SpectralImageEncoder(cfg.image_feature_dim)

        fusion_dim = cfg.time_feature_dim + cfg.image_feature_dim + 2
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim // 2, len(cfg.class_names)),
        )

    def extract_features(self, time_signal, spectral_image, env):
        z_time = self.time_encoder(time_signal)
        z_img = self.image_encoder(spectral_image)
        z = torch.cat([z_time, z_img, env], dim=1)
        return {
            "time_feature": z_time,
            "image_feature": z_img,
            "fusion_feature": z,
        }

    def forward(self, time_signal, spectral_image, env):
        features = self.extract_features(time_signal, spectral_image, env)
        logits = self.fusion_head(features["fusion_feature"])
        return logits, features
