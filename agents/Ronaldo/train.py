"""
Training script for Ronaldo agent.
Supervised learning from match data with optional winner-weighting.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from agents.Ronaldo.net import RonaldoNet, RonaldoNetSmall
from agents.Ronaldo.data import create_dataloaders, load_csv_dataset, ChickenDataset


def train_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    policy_weight: float = 1.0,
    value_weight: float = 0.5,
) -> dict:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_policy_acc = 0.0
    num_batches = 0
    
    for batch in train_loader:
        board, scalars, actions, values, weights = batch
        board = board.to(device)
        scalars = scalars.to(device)
        actions = actions.to(device)
        values = values.to(device).float()
        weights = weights.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        policy_logits, value_pred = model(board, scalars)
        
        # Policy loss (weighted cross-entropy)
        policy_loss = F.cross_entropy(policy_logits, actions, reduction='none')
        policy_loss = (policy_loss * weights).mean()
        
        # Value loss (weighted MSE)
        value_loss = F.mse_loss(value_pred, values, reduction='none')
        value_loss = (value_loss * weights).mean()
        
        # Combined loss
        loss = policy_weight * policy_loss + value_weight * value_loss
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        
        # Policy accuracy
        pred_actions = policy_logits.argmax(dim=1)
        total_policy_acc += (pred_actions == actions).float().mean().item()
        
        num_batches += 1
    
    return {
        "loss": total_loss / num_batches,
        "policy_loss": total_policy_loss / num_batches,
        "value_loss": total_value_loss / num_batches,
        "policy_acc": total_policy_acc / num_batches,
    }


def validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    policy_weight: float = 1.0,
    value_weight: float = 0.5,
) -> dict:
    """Validate the model."""
    model.eval()
    
    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_policy_acc = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            board, scalars, actions, values, weights = batch
            board = board.to(device)
            scalars = scalars.to(device)
            actions = actions.to(device)
            values = values.to(device).float()
            weights = weights.to(device)
            
            # Forward pass
            policy_logits, value_pred = model(board, scalars)
            
            # Policy loss
            policy_loss = F.cross_entropy(policy_logits, actions, reduction='none')
            policy_loss = (policy_loss * weights).mean()
            
            # Value loss
            value_loss = F.mse_loss(value_pred, values, reduction='none')
            value_loss = (value_loss * weights).mean()
            
            # Combined loss
            loss = policy_weight * policy_loss + value_weight * value_loss
            
            # Metrics
            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            
            # Policy accuracy
            pred_actions = policy_logits.argmax(dim=1)
            total_policy_acc += (pred_actions == actions).float().mean().item()
            
            num_batches += 1
    
    return {
        "loss": total_loss / num_batches,
        "policy_loss": total_policy_loss / num_batches,
        "value_loss": total_value_loss / num_batches,
        "policy_acc": total_policy_acc / num_batches,
    }


def train(
    csv_path: str,
    output_path: str,
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    winner_only: bool = False,
    winner_weight: float = 1.0,
    loser_weight: float = 0.3,
    policy_weight: float = 1.0,
    value_weight: float = 0.5,
    limit: Optional[int] = None,
    min_egg_diff: int = 0,
    min_winner_eggs: int = 0,
    exclude_draws: bool = False,
    use_small_net: bool = False,
    device: Optional[str] = None,
):
    """Main training function."""
    # Device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"Using device: {device}")
    
    # Create dataloaders
    print(f"Loading data from {csv_path}...")
    if min_egg_diff > 0 or min_winner_eggs > 0 or exclude_draws:
        print(f"Filtering: min_egg_diff={min_egg_diff}, min_winner_eggs={min_winner_eggs}, exclude_draws={exclude_draws}")
    train_loader, val_loader = create_dataloaders(
        csv_path,
        batch_size=batch_size,
        winner_only=winner_only,
        winner_weight=winner_weight,
        loser_weight=loser_weight,
        limit=limit,
        min_egg_diff=min_egg_diff,
        min_winner_eggs=min_winner_eggs,
        exclude_draws=exclude_draws,
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Create model
    if use_small_net:
        model = RonaldoNetSmall(board_size=8, in_channels=10, channels=64)
        print("Using RonaldoNetSmall (64 channels)")
    else:
        model = RonaldoNet(board_size=8, in_channels=10, channels=128, num_res_blocks=4)
        print("Using RonaldoNet (128 channels, 4 res blocks)")
    
    model = model.to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    
    # Training loop
    best_val_loss = float("inf")
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device,
            policy_weight=policy_weight, value_weight=value_weight,
        )
        
        # Validate
        val_metrics = validate(
            model, val_loader, device,
            policy_weight=policy_weight, value_weight=value_weight,
        )
        
        # Update scheduler
        scheduler.step()
        
        # Print progress
        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['policy_acc']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['policy_acc']:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.6f}"
        )
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_val_acc = val_metrics['policy_acc']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'val_acc': best_val_acc,
                'use_small_net': use_small_net,
            }, output_path)
            print(f"  -> Saved best model (val_loss: {best_val_loss:.4f}, val_acc: {best_val_acc:.4f})")
    
    print(f"\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {output_path}")
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Train Ronaldo agent")
    parser.add_argument("--data", type=str, default="data-matches.csv",
                        help="Path to CSV data file")
    parser.add_argument("--output", type=str, default="agents/Ronaldo/ronaldo_weights.pt",
                        help="Output path for model weights")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay")
    parser.add_argument("--winner_only", action="store_true",
                        help="Train only on winner moves")
    parser.add_argument("--winner_weight", type=float, default=1.0,
                        help="Weight for winner moves")
    parser.add_argument("--loser_weight", type=float, default=0.3,
                        help="Weight for loser moves")
    parser.add_argument("--policy_weight", type=float, default=1.0,
                        help="Weight for policy loss")
    parser.add_argument("--value_weight", type=float, default=0.5,
                        help="Weight for value loss")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of matches to load")
    parser.add_argument("--min_egg_diff", type=int, default=0,
                        help="Minimum egg differential to include match (filters for dominant wins)")
    parser.add_argument("--min_winner_eggs", type=int, default=0,
                        help="Minimum eggs the winner must have")
    parser.add_argument("--exclude_draws", action="store_true",
                        help="Exclude draw games")
    parser.add_argument("--small", action="store_true",
                        help="Use smaller network")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    train(
        csv_path=args.data,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        winner_only=args.winner_only,
        winner_weight=args.winner_weight,
        loser_weight=args.loser_weight,
        policy_weight=args.policy_weight,
        value_weight=args.value_weight,
        limit=args.limit,
        min_egg_diff=args.min_egg_diff,
        min_winner_eggs=args.min_winner_eggs,
        exclude_draws=args.exclude_draws,
        use_small_net=args.small,
        device=args.device,
    )


if __name__ == "__main__":
    main()

