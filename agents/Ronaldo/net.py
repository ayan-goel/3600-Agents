"""
Neural network architecture for Ronaldo agent.
Dual-head CNN: policy (12 actions) + value (win probability).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block with two conv layers and skip connection."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + residual)
        return out


class RonaldoNet(nn.Module):
    """
    CNN policy + value network for the chicken game.
    
    Input:
        - board_tensor: (batch, 10, 8, 8) spatial features
        - scalar_features: (batch, 8) scalar game state
    
    Output:
        - policy_logits: (batch, 12) - 4 directions × 3 move types
        - value: (batch,) - win probability in [-1, 1]
    """
    
    def __init__(
        self,
        board_size: int = 8,
        in_channels: int = 10,
        channels: int = 128,
        num_res_blocks: int = 4,
    ):
        super().__init__()
        self.board_size = board_size
        
        # Initial convolution
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        
        # Residual tower
        self.res_blocks = nn.ModuleList([
            ResidualBlock(channels) for _ in range(num_res_blocks)
        ])
        
        # Flatten size
        flat_size = channels * board_size * board_size
        
        # Scalar feature processing
        self.scalar_fc = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )
        
        # Policy head
        self.policy_conv = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        policy_flat = 32 * board_size * board_size
        self.policy_fc = nn.Sequential(
            nn.Linear(policy_flat + 64, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 12),  # 4 directions × 3 move types
        )
        
        # Value head
        self.value_conv = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        value_flat = 32 * board_size * board_size
        self.value_fc = nn.Sequential(
            nn.Linear(value_flat + 64, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
            nn.Tanh(),
        )
    
    def forward(
        self,
        board_tensor: torch.Tensor,
        scalar_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Convolutional tower
        x = self.input_conv(board_tensor)
        for block in self.res_blocks:
            x = block(x)
        
        # Process scalars
        scalars = self.scalar_fc(scalar_features)
        
        # Policy head
        p = self.policy_conv(x)
        p = p.view(p.size(0), -1)
        p = torch.cat([p, scalars], dim=1)
        policy_logits = self.policy_fc(p)
        
        # Value head
        v = self.value_conv(x)
        v = v.view(v.size(0), -1)
        v = torch.cat([v, scalars], dim=1)
        value = self.value_fc(v).squeeze(-1)
        
        return policy_logits, value


class RonaldoNetSmall(nn.Module):
    """Smaller/faster version for quick iteration."""
    
    def __init__(self, board_size: int = 8, in_channels: int = 10, channels: int = 64):
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
            nn.Linear(8, 64),
            nn.ReLU(inplace=True),
        )
        
        self.policy_head = nn.Sequential(
            nn.Linear(flat_size + 64, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 12),
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(flat_size + 64, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Tanh(),
        )
    
    def forward(
        self,
        board_tensor: torch.Tensor,
        scalar_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.conv_tower(board_tensor)
        x = x.view(x.size(0), -1)
        scalars = self.scalar_fc(scalar_features)
        h = torch.cat([x, scalars], dim=1)
        policy_logits = self.policy_head(h)
        value = self.value_head(h).squeeze(-1)
        return policy_logits, value






