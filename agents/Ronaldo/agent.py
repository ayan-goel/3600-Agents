"""
Ronaldo Agent - Neural Network based agent trained on 16K+ matches.

Uses a CNN policy + value network with lightweight heuristic safety checks.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from game.board import Board
from game.enums import Direction, MoveType

# Action space mapping
DIRECTIONS = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]
MOVE_TYPES = [MoveType.PLAIN, MoveType.EGG, MoveType.TURD]

ACTION_SPACE: List[Tuple[Direction, MoveType]] = [
    (d, m) for d in DIRECTIONS for m in MOVE_TYPES
]
ACTION_TO_IDX: Dict[Tuple[Direction, MoveType], int] = {
    action: idx for idx, action in enumerate(ACTION_SPACE)
}
IDX_TO_ACTION: Dict[int, Tuple[Direction, MoveType]] = {
    idx: action for action, idx in ACTION_TO_IDX.items()
}

# Direction deltas
DIR_DELTA = {
    Direction.UP: (0, -1),
    Direction.DOWN: (0, 1),
    Direction.LEFT: (-1, 0),
    Direction.RIGHT: (1, 0),
}


class PlayerAgent:
    """Neural network agent with heuristic safety checks."""
    
    def __init__(self, board: Board, time_left: Callable[[], float]):
        del time_left  # Not used
        self.model = None
        self.device = None
        self.size = board.game_map.MAP_SIZE
        self.known_traps: Set[Tuple[int, int]] = set()
        self.turn_count = 0
        
        # Load model if available
        self._load_model()
    
    def _load_model(self):
        """Load trained model weights."""
        if not TORCH_AVAILABLE:
            print("PyTorch not available, using fallback heuristics")
            return
        
        # Try to find weights file (prefer self-play weights)
        agent_dir = Path(__file__).parent
        weights_paths = [
            agent_dir / "ronaldo_selfplay_best.pt",
            agent_dir / "ronaldo_weights.pt",
            agent_dir / "ronaldo_weights_winner.pt",
        ]
        
        weights_path = None
        for p in weights_paths:
            if p.exists():
                weights_path = p
                break
        
        if weights_path is None:
            print("No weights found, using fallback heuristics")
            return
        
        try:
            # Import network
            from agents.Ronaldo.net import RonaldoNet, RonaldoNetSmall
            
            # Load checkpoint
            self.device = torch.device("cpu")  # Use CPU for inference
            checkpoint = torch.load(weights_path, map_location=self.device)
            
            # Create model
            use_small = checkpoint.get('use_small_net', False)
            if use_small:
                self.model = RonaldoNetSmall(board_size=8, in_channels=10, channels=64)
            else:
                self.model = RonaldoNet(board_size=8, in_channels=10, channels=128, num_res_blocks=4)
            
            # Load weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            print(f"Loaded Ronaldo model from {weights_path}")
            val_acc = checkpoint.get('val_acc', checkpoint.get('avg_eggs', 'N/A'))
            if isinstance(val_acc, (int, float)):
                print(f"  Val accuracy/avg_eggs: {val_acc:.4f}")
            else:
                print(f"  Val accuracy: {val_acc}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
    
    def play(
        self,
        board: Board,
        sensor_data: List[Tuple[bool, bool]],
        time_left: Callable[[], float],
    ) -> Tuple[Direction, MoveType]:
        """Select a move using neural network + safety checks."""
        del sensor_data, time_left  # Not used
        self.turn_count += 1
        self.size = board.game_map.MAP_SIZE
        
        # Update known traps
        self._update_traps(board)
        
        # Get legal moves
        legal_moves = self._get_legal_moves(board)
        
        if not legal_moves:
            # Fallback to any direction with PLAIN
            return (Direction.UP, MoveType.PLAIN)
        
        # If only one legal move, take it
        if len(legal_moves) == 1:
            return legal_moves[0]
        
        # Try neural network
        if self.model is not None and TORCH_AVAILABLE:
            try:
                move = self._neural_move(board, legal_moves)
                if move is not None:
                    return move
            except Exception as e:
                pass  # Fall back to heuristics
        
        # Fallback to heuristics
        return self._heuristic_move(board, legal_moves)
    
    def _update_traps(self, board: Board):
        """Track revealed trapdoors."""
        # Use found_trapdoors attribute if available
        for loc in getattr(board, "found_trapdoors", set()):
            self.known_traps.add(loc)
    
    def _get_legal_moves(self, board: Board) -> List[Tuple[Direction, MoveType]]:
        """Get all legal moves from current position."""
        chicken = board.chicken_player
        cur = chicken.get_location()
        legal = []
        
        for direction in DIRECTIONS:
            dx, dy = DIR_DELTA[direction]
            nx, ny = cur[0] + dx, cur[1] + dy
            
            # Check bounds
            if not (0 <= nx < self.size and 0 <= ny < self.size):
                continue
            
            # Check for obstacles (turds, enemy)
            if (nx, ny) in board.turds_player or (nx, ny) in board.turds_enemy:
                continue
            if (nx, ny) == board.chicken_enemy.get_location():
                continue
            
            # PLAIN is always legal if direction is valid
            legal.append((direction, MoveType.PLAIN))
            
            # EGG is legal if we can lay
            if board.can_lay_egg():
                legal.append((direction, MoveType.EGG))
            
            # TURD is legal if we have turds left
            if chicken.turds_left > 0:
                legal.append((direction, MoveType.TURD))
        
        return legal
    
    def _neural_move(
        self,
        board: Board,
        legal_moves: List[Tuple[Direction, MoveType]],
    ) -> Optional[Tuple[Direction, MoveType]]:
        """Use neural network to select move."""
        # Build input tensors
        board_tensor, scalar_tensor = self._featurize(board)
        
        # Forward pass
        with torch.no_grad():
            policy_logits, value = self.model(board_tensor, scalar_tensor)
        
        # Mask illegal moves
        legal_indices = set(ACTION_TO_IDX[m] for m in legal_moves)
        mask = torch.full((12,), float('-inf'))
        for idx in legal_indices:
            mask[idx] = 0.0
        
        masked_logits = policy_logits[0] + mask
        probs = F.softmax(masked_logits, dim=0)
        
        # Apply safety adjustments
        adjusted_probs = self._apply_safety(board, legal_moves, probs)
        
        # Select best move
        best_idx = adjusted_probs.argmax().item()
        return IDX_TO_ACTION[best_idx]
    
    def _featurize(self, board: Board) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert board state to neural network input."""
        size = self.size
        state = np.zeros((10, size, size), dtype=np.float32)
        
        # Channel 0: Board mask
        state[0, :, :] = 1.0
        
        # Channel 1-2: Player positions
        px, py = board.chicken_player.get_location()
        ex, ey = board.chicken_enemy.get_location()
        state[1, py, px] = 1.0
        state[2, ey, ex] = 1.0
        
        # Channel 3-4: Eggs
        for (x, y) in board.eggs_player:
            state[3, y, x] = 1.0
        for (x, y) in board.eggs_enemy:
            state[4, y, x] = 1.0
        
        # Channel 5-6: Turds
        for (x, y) in board.turds_player:
            state[5, y, x] = 1.0
        for (x, y) in board.turds_enemy:
            state[6, y, x] = 1.0
        
        # Channel 7: Known traps
        for (x, y) in self.known_traps:
            state[7, y, x] = 1.0
        
        # Channel 8: Distance from self
        for y in range(size):
            for x in range(size):
                dist = abs(x - px) + abs(y - py)
                state[8, y, x] = 1.0 - (dist / (2 * size))
        
        # Channel 9: Distance from enemy
        for y in range(size):
            for x in range(size):
                dist = abs(x - ex) + abs(y - ey)
                state[9, y, x] = 1.0 - (dist / (2 * size))
        
        # Scalar features
        scalars = np.array([
            board.turns_left_player / 40.0,
            board.turns_left_enemy / 40.0,
            len(board.eggs_player) / 40.0,
            len(board.eggs_enemy) / 40.0,
            board.chicken_player.turds_left / 5.0,
            board.chicken_enemy.turds_left / 5.0,
            len(self.known_traps) / 2.0,
            self.turn_count / 40.0,
        ], dtype=np.float32)
        
        board_tensor = torch.from_numpy(state).unsqueeze(0)
        scalar_tensor = torch.from_numpy(scalars).unsqueeze(0)
        
        return board_tensor, scalar_tensor
    
    def _apply_safety(
        self,
        board: Board,
        legal_moves: List[Tuple[Direction, MoveType]],
        probs: torch.Tensor,
    ) -> torch.Tensor:
        """Apply safety heuristics to adjust probabilities."""
        adjusted = probs.clone()
        cur = board.chicken_player.get_location()
        enemy = board.chicken_enemy.get_location()
        
        for move in legal_moves:
            idx = ACTION_TO_IDX[move]
            direction, move_type = move
            dx, dy = DIR_DELTA[direction]
            nx, ny = cur[0] + dx, cur[1] + dy
            
            # Penalty for walking into known traps
            if (nx, ny) in self.known_traps:
                adjusted[idx] *= 0.1
            
            # Bonus for laying eggs in good positions
            if move_type == MoveType.EGG:
                # Bonus if we haven't laid here before
                if cur not in board.eggs_player:
                    adjusted[idx] *= 1.3
                # Bonus for laying near our other eggs (chaining)
                adjacent_own_eggs = sum(
                    1 for ex, ey in board.eggs_player
                    if abs(ex - cur[0]) + abs(ey - cur[1]) == 1
                )
                if adjacent_own_eggs > 0:
                    adjusted[idx] *= (1.0 + 0.15 * adjacent_own_eggs)
            
            # Bonus for moves that expand territory
            dist_from_center = abs(nx - 3.5) + abs(ny - 3.5)
            cur_dist_from_center = abs(cur[0] - 3.5) + abs(cur[1] - 3.5)
            if self.turn_count <= 10:
                # Early game: prefer outward expansion
                if dist_from_center > cur_dist_from_center:
                    adjusted[idx] *= 1.1
            
            # Penalty for moves toward enemy when we're ahead
            dist_to_enemy = abs(nx - enemy[0]) + abs(ny - enemy[1])
            cur_dist_to_enemy = abs(cur[0] - enemy[0]) + abs(cur[1] - enemy[1])
            if len(board.eggs_player) > len(board.eggs_enemy) + 2:
                # We're ahead, be cautious
                if dist_to_enemy < cur_dist_to_enemy:
                    adjusted[idx] *= 0.9
            
            # Turd strategy: place to block enemy
            if move_type == MoveType.TURD:
                # Good if it blocks enemy's path
                if dist_to_enemy <= 3:
                    adjusted[idx] *= 1.2
                else:
                    adjusted[idx] *= 0.7
        
        # Renormalize
        adjusted = adjusted / adjusted.sum()
        return adjusted
    
    def _heuristic_move(
        self,
        board: Board,
        legal_moves: List[Tuple[Direction, MoveType]],
    ) -> Tuple[Direction, MoveType]:
        """Fallback heuristic move selection."""
        cur = board.chicken_player.get_location()
        enemy = board.chicken_enemy.get_location()
        
        best_move = legal_moves[0]
        best_score = float('-inf')
        
        for move in legal_moves:
            direction, move_type = move
            dx, dy = DIR_DELTA[direction]
            nx, ny = cur[0] + dx, cur[1] + dy
            
            score = 0.0
            
            # Avoid known traps
            if (nx, ny) in self.known_traps:
                score -= 100.0
            
            # Prefer laying eggs
            if move_type == MoveType.EGG and cur not in board.eggs_player:
                score += 15.0
            
            # Prefer moving toward center in early game
            if self.turn_count <= 10:
                center_dist = abs(nx - 3.5) + abs(ny - 3.5)
                score -= center_dist * 0.5
            
            # Prefer expanding territory
            dist_from_start = abs(nx - cur[0]) + abs(ny - cur[1])
            score += dist_from_start * 2.0
            
            # Avoid corners in early game
            if self.turn_count <= 15:
                if (nx == 0 or nx == 7) and (ny == 0 or ny == 7):
                    score -= 5.0
            
            # Prefer moves away from enemy turds
            for tx, ty in board.turds_enemy:
                if abs(nx - tx) + abs(ny - ty) <= 2:
                    score -= 3.0
            
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move


# Alias for compatibility
Agent = PlayerAgent

