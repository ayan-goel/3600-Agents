from __future__ import annotations

import torch
import torch.nn as nn


class ChickenNet(nn.Module):
    """Shared convolutional policy + value network (copied for Jamal)."""

    def __init__(self, board_size: int, in_channels: int = 9, channels: int = 64):
        super().__init__()
        self.board_size = board_size
        self.conv_tower = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        flat_size = channels * board_size * board_size
        self.scalar_fc = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )
        self.policy_head = nn.Sequential(
            nn.Linear(flat_size + 64, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 12),  # 4 directions * 3 MoveType entries
        )
        self.value_head = nn.Sequential(
            nn.Linear(flat_size + 64, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

    def forward(self, board_tensor: torch.Tensor, scalar_features: torch.Tensor):
        x = self.conv_tower(board_tensor)
        x = x.view(x.size(0), -1)
        scalars = self.scalar_fc(scalar_features)
        h = torch.cat([x, scalars], dim=1)
        policy_logits = self.policy_head(h)
        value = self.value_head(h).squeeze(-1)
        return policy_logits, value


