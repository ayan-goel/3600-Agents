"""
Data loading and preprocessing for Ronaldo agent.
Parses CSV match data and creates training samples.
"""
from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

# Action space: 4 directions × 3 move types = 12 actions
DIRECTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]
MOVE_TYPES = ["PLAIN", "EGG", "TURD"]

ACTION_SPACE: List[Tuple[str, str]] = [
    (d, m) for d in DIRECTIONS for m in MOVE_TYPES
]
ACTION_TO_IDX: Dict[Tuple[str, str], int] = {
    action: idx for idx, action in enumerate(ACTION_SPACE)
}
IDX_TO_ACTION: Dict[int, Tuple[str, str]] = {
    idx: action for action, idx in ACTION_TO_IDX.items()
}

# Direction deltas
DIR_DELTA = {
    "UP": (0, -1),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
    "RIGHT": (1, 0),
}


def pos_to_direction(prev: Tuple[int, int], cur: Tuple[int, int]) -> str:
    """Infer direction from position change."""
    dx = cur[0] - prev[0]
    dy = cur[1] - prev[1]
    
    if dx == 1 and dy == 0:
        return "RIGHT"
    elif dx == -1 and dy == 0:
        return "LEFT"
    elif dx == 0 and dy == 1:
        return "DOWN"
    elif dx == 0 and dy == -1:
        return "UP"
    else:
        # Invalid move (teleport, etc.) - default to UP
        return "UP"


def left_behind_to_move_type(lb: str) -> str:
    """Convert left_behind string to move type."""
    lb = lb.lower()
    if lb == "egg":
        return "EGG"
    elif lb == "turd":
        return "TURD"
    else:
        return "PLAIN"


@dataclass
class TrainingSample:
    """A single training sample."""
    board_tensor: np.ndarray  # (10, 8, 8)
    scalar_features: np.ndarray  # (8,)
    action_idx: int  # 0-11
    value: float  # -1, 0, or 1 (from mover's perspective)
    winner_move: bool  # True if this move was made by the eventual winner


class ChickenDataset(Dataset):
    """PyTorch dataset for chicken game matches."""
    
    def __init__(
        self,
        samples: List[TrainingSample],
        winner_only: bool = False,
        winner_weight: float = 1.0,
        loser_weight: float = 0.3,
    ):
        self.winner_only = winner_only
        self.winner_weight = winner_weight
        self.loser_weight = loser_weight
        
        if winner_only:
            self.samples = [s for s in samples if s.winner_move]
        else:
            self.samples = samples
        
        self.weights = np.array([
            self.winner_weight if s.winner_move else self.loser_weight
            for s in self.samples
        ], dtype=np.float32)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, float, float]:
        s = self.samples[idx]
        return (
            torch.from_numpy(s.board_tensor),
            torch.from_numpy(s.scalar_features),
            s.action_idx,
            s.value,
            self.weights[idx],
        )


def parse_match(match_data: dict, board_size: int = 8) -> List[TrainingSample]:
    """
    Parse a single match into training samples.
    
    Returns one sample per half-turn (both players' moves).
    """
    samples = []
    
    positions = match_data.get("pos", [])
    left_behind = match_data.get("left_behind", [])
    spawn_a = tuple(match_data.get("spawn_a", [0, 0]))
    spawn_b = tuple(match_data.get("spawn_b", [7, 7]))
    trapdoors = [tuple(t) for t in match_data.get("trapdoors", [])]
    result = match_data.get("result", 0)  # 1 = A wins, -1 = B wins, 0 = draw
    
    a_eggs_laid = match_data.get("a_eggs_laid", [])
    b_eggs_laid = match_data.get("b_eggs_laid", [])
    a_turds_left = match_data.get("a_turds_left", [])
    b_turds_left = match_data.get("b_turds_left", [])
    a_moves_left = match_data.get("a_moves_left", [])
    b_moves_left = match_data.get("b_moves_left", [])
    trapdoor_triggered = match_data.get("trapdoor_triggered", [])
    
    if len(positions) == 0:
        return samples
    
    # Track game state
    cur_a = spawn_a
    cur_b = spawn_b
    eggs_a: set = set()
    eggs_b: set = set()
    turds_a: set = set()
    turds_b: set = set()
    known_traps: set = set()
    
    for half_idx, pos in enumerate(positions):
        new_pos = tuple(pos)
        is_player_a = (half_idx % 2 == 0)
        turn_idx = half_idx // 2
        
        # Get previous position
        prev_pos = cur_a if is_player_a else cur_b
        
        # Infer action
        direction = pos_to_direction(prev_pos, new_pos)
        lb = left_behind[half_idx] if half_idx < len(left_behind) else "plain"
        move_type = left_behind_to_move_type(lb)
        action_idx = ACTION_TO_IDX.get((direction, move_type), 0)
        
        # Check if trapdoor was triggered this turn
        if half_idx < len(trapdoor_triggered) and trapdoor_triggered[half_idx]:
            # Add the trap that was triggered
            for trap in trapdoors:
                if trap == new_pos:
                    known_traps.add(trap)
        
        # Build board tensor (10 channels)
        board = np.zeros((10, board_size, board_size), dtype=np.float32)
        
        # Channel 0: Board mask (all 1s)
        board[0, :, :] = 1.0
        
        # Channels 1-2: Player positions (self, enemy)
        if is_player_a:
            self_pos, enemy_pos = cur_a, cur_b
            self_eggs, enemy_eggs = eggs_a, eggs_b
            self_turds, enemy_turds = turds_a, turds_b
        else:
            self_pos, enemy_pos = cur_b, cur_a
            self_eggs, enemy_eggs = eggs_b, eggs_a
            self_turds, enemy_turds = turds_b, turds_a
        
        board[1, self_pos[1], self_pos[0]] = 1.0
        board[2, enemy_pos[1], enemy_pos[0]] = 1.0
        
        # Channels 3-4: Eggs (self, enemy)
        for (x, y) in self_eggs:
            board[3, y, x] = 1.0
        for (x, y) in enemy_eggs:
            board[4, y, x] = 1.0
        
        # Channels 5-6: Turds (self, enemy)
        for (x, y) in self_turds:
            board[5, y, x] = 1.0
        for (x, y) in enemy_turds:
            board[6, y, x] = 1.0
        
        # Channel 7: Known trapdoors
        for (x, y) in known_traps:
            board[7, y, x] = 1.0
        
        # Channel 8: Distance from self (normalized)
        for y in range(board_size):
            for x in range(board_size):
                dist = abs(x - self_pos[0]) + abs(y - self_pos[1])
                board[8, y, x] = 1.0 - (dist / (2 * board_size))
        
        # Channel 9: Distance from enemy (normalized)
        for y in range(board_size):
            for x in range(board_size):
                dist = abs(x - enemy_pos[0]) + abs(y - enemy_pos[1])
                board[9, y, x] = 1.0 - (dist / (2 * board_size))
        
        # Build scalar features (8 values)
        if is_player_a:
            self_eggs_count = a_eggs_laid[turn_idx] if turn_idx < len(a_eggs_laid) else len(eggs_a)
            enemy_eggs_count = b_eggs_laid[turn_idx] if turn_idx < len(b_eggs_laid) else len(eggs_b)
            self_turds_left = a_turds_left[half_idx] if half_idx < len(a_turds_left) else 5
            enemy_turds_left = b_turds_left[half_idx] if half_idx < len(b_turds_left) else 5
            self_moves_left = a_moves_left[half_idx] if half_idx < len(a_moves_left) else 40
            enemy_moves_left = b_moves_left[half_idx] if half_idx < len(b_moves_left) else 40
        else:
            self_eggs_count = b_eggs_laid[turn_idx] if turn_idx < len(b_eggs_laid) else len(eggs_b)
            enemy_eggs_count = a_eggs_laid[turn_idx] if turn_idx < len(a_eggs_laid) else len(eggs_a)
            self_turds_left = b_turds_left[half_idx] if half_idx < len(b_turds_left) else 5
            enemy_turds_left = a_turds_left[half_idx] if half_idx < len(a_turds_left) else 5
            self_moves_left = b_moves_left[half_idx] if half_idx < len(b_moves_left) else 40
            enemy_moves_left = a_moves_left[half_idx] if half_idx < len(a_moves_left) else 40
        
        scalars = np.array([
            self_moves_left / 40.0,
            enemy_moves_left / 40.0,
            self_eggs_count / 40.0,
            enemy_eggs_count / 40.0,
            self_turds_left / 5.0,
            enemy_turds_left / 5.0,
            len(known_traps) / 2.0,  # 0, 0.5, or 1
            half_idx / 80.0,  # Game progress
        ], dtype=np.float32)
        
        # Value label (from mover's perspective)
        if result == 1:  # A wins
            value = 1.0 if is_player_a else -1.0
        elif result == -1:  # B wins
            value = -1.0 if is_player_a else 1.0
        else:  # Draw
            value = 0.0
        
        # Is this a winner's move?
        winner_move = (result == 1 and is_player_a) or (result == -1 and not is_player_a)
        
        samples.append(TrainingSample(
            board_tensor=board,
            scalar_features=scalars,
            action_idx=action_idx,
            value=value,
            winner_move=winner_move,
        ))
        
        # Update game state
        if is_player_a:
            if move_type == "EGG":
                eggs_a.add(cur_a)
            elif move_type == "TURD":
                turds_a.add(cur_a)
            cur_a = new_pos
        else:
            if move_type == "EGG":
                eggs_b.add(cur_b)
            elif move_type == "TURD":
                turds_b.add(cur_b)
            cur_b = new_pos
    
    return samples


def load_csv_dataset(
    csv_path: str,
    board_size: int = 8,
    limit: Optional[int] = None,
    min_turns: int = 20,
    min_egg_diff: int = 0,
    min_winner_eggs: int = 0,
    exclude_draws: bool = False,
) -> List[TrainingSample]:
    """
    Load training samples from CSV file.
    
    Args:
        csv_path: Path to data-matches.csv
        board_size: Board size (default 8)
        limit: Max number of matches to load (None = all)
        min_turns: Minimum turns for a valid match
        min_egg_diff: Minimum egg differential to include match (filters for dominant wins)
        min_winner_eggs: Minimum eggs the winner must have
        exclude_draws: Whether to exclude draws
    
    Returns:
        List of TrainingSample objects
    """
    all_samples = []
    matches_loaded = 0
    matches_skipped = 0
    
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            if limit is not None and matches_loaded >= limit:
                break
            
            try:
                match_log = json.loads(row["match_log"])
                
                # Skip matches with too few turns
                turn_count = match_log.get("turn_count", 0)
                if turn_count < min_turns:
                    matches_skipped += 1
                    continue
                
                # Skip draws if requested
                result = match_log.get("result", 2)  # 0=A wins, 1=B wins, 2=draw
                if exclude_draws and result == 2:
                    matches_skipped += 1
                    continue
                
                # Filter by egg differential
                a_eggs_list = match_log.get("a_eggs_laid", [0])
                b_eggs_list = match_log.get("b_eggs_laid", [0])
                a_eggs = a_eggs_list[-1] if a_eggs_list else 0
                b_eggs = b_eggs_list[-1] if b_eggs_list else 0
                egg_diff = abs(a_eggs - b_eggs)
                winner_eggs = max(a_eggs, b_eggs)
                
                if egg_diff < min_egg_diff:
                    matches_skipped += 1
                    continue
                
                if winner_eggs < min_winner_eggs:
                    matches_skipped += 1
                    continue
                
                samples = parse_match(match_log, board_size)
                all_samples.extend(samples)
                matches_loaded += 1
                
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                matches_skipped += 1
                continue
    
    print(f"Loaded {matches_loaded} matches, skipped {matches_skipped}")
    print(f"Total samples: {len(all_samples)}")
    
    return all_samples


def create_dataloaders(
    csv_path: str,
    batch_size: int = 256,
    train_split: float = 0.9,
    winner_only: bool = False,
    winner_weight: float = 1.0,
    loser_weight: float = 0.3,
    limit: Optional[int] = None,
    min_egg_diff: int = 0,
    min_winner_eggs: int = 0,
    exclude_draws: bool = False,
    num_workers: int = 0,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create train and validation dataloaders."""
    from torch.utils.data import DataLoader, random_split
    
    # Load all samples
    samples = load_csv_dataset(
        csv_path, 
        limit=limit,
        min_egg_diff=min_egg_diff,
        min_winner_eggs=min_winner_eggs,
        exclude_draws=exclude_draws,
    )
    
    # Create dataset
    dataset = ChickenDataset(
        samples,
        winner_only=winner_only,
        winner_weight=winner_weight,
        loser_weight=loser_weight,
    )
    
    # Split into train/val
    train_size = int(len(dataset) * train_split)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test data loading
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "data-matches.csv"
    
    samples = load_csv_dataset(csv_path, limit=100)
    print(f"\nSample board shape: {samples[0].board_tensor.shape}")
    print(f"Sample scalars shape: {samples[0].scalar_features.shape}")
    print(f"Sample action: {IDX_TO_ACTION[samples[0].action_idx]}")
    print(f"Sample value: {samples[0].value}")
    print(f"Winner move: {samples[0].winner_move}")
    
    # Count action distribution
    action_counts = defaultdict(int)
    winner_counts = defaultdict(int)
    for s in samples:
        action_counts[IDX_TO_ACTION[s.action_idx]] += 1
        if s.winner_move:
            winner_counts[IDX_TO_ACTION[s.action_idx]] += 1
    
    print("\nAction distribution:")
    for action, count in sorted(action_counts.items(), key=lambda x: -x[1]):
        print(f"  {action}: {count}")

