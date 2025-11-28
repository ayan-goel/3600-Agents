"""
Self-Play Reinforcement Learning for Ronaldo Agent.

This implements:
1. Self-play game generation with the current policy
2. Experience collection with state-action-reward tuples
3. Policy gradient training (REINFORCE with baseline)
4. Periodic evaluation against baseline agents
5. Checkpointing and logging
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
ENGINE_ROOT = PROJECT_ROOT / "engine"
sys.path.insert(0, str(ENGINE_ROOT))

from agents.Ronaldo.net import RonaldoNet, RonaldoNetSmall
from agents.Ronaldo.data import ACTION_SPACE, ACTION_TO_IDX, IDX_TO_ACTION, DIR_DELTA

# Import game engine components
from game.board import Board
from game.game_map import GameMap
from game.trapdoor_manager import TrapdoorManager
from game.enums import Direction, MoveType, Result

# Direction mapping
DIR_MAP = {
    "UP": Direction.UP,
    "DOWN": Direction.DOWN,
    "LEFT": Direction.LEFT,
    "RIGHT": Direction.RIGHT,
}
MOVE_MAP = {
    "PLAIN": MoveType.PLAIN,
    "EGG": MoveType.EGG,
    "TURD": MoveType.TURD,
}


@dataclass
class Experience:
    """Single experience tuple from self-play."""
    state_board: np.ndarray  # (10, 8, 8)
    state_scalar: np.ndarray  # (8,)
    action_idx: int
    log_prob: float
    reward: float = 0.0  # Filled in after game ends
    value: float = 0.0  # Value estimate at this state


@dataclass
class GameResult:
    """Result of a self-play game."""
    experiences_a: List[Experience]
    experiences_b: List[Experience]
    winner: int  # 1 = A wins, -1 = B wins, 0 = tie
    eggs_a: int
    eggs_b: int
    turns: int
    reason: str


class SelfPlayAgent:
    """Agent wrapper for self-play that collects experiences."""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        is_player_a: bool,
        temperature: float = 1.0,
        collect_experiences: bool = True,
    ):
        self.model = model
        self.device = device
        self.is_player_a = is_player_a
        self.temperature = temperature
        self.collect_experiences = collect_experiences
        self.experiences: List[Experience] = []
        self.turn_count = 0
        self.known_traps: set = set()
    
    def reset(self):
        """Reset for new game."""
        self.experiences = []
        self.turn_count = 0
        self.known_traps = set()
    
    def featurize(self, board: Board) -> Tuple[np.ndarray, np.ndarray]:
        """Convert board state to neural network input."""
        size = board.game_map.MAP_SIZE
        state = np.zeros((10, size, size), dtype=np.float32)
        
        # Channel 0: Board mask
        state[0, :, :] = 1.0
        
        # Get positions based on perspective
        if self.is_player_a:
            self_chicken = board.chicken_player if board.is_as_turn else board.chicken_enemy
            enemy_chicken = board.chicken_enemy if board.is_as_turn else board.chicken_player
            self_eggs = board.eggs_player if board.is_as_turn else board.eggs_enemy
            enemy_eggs = board.eggs_enemy if board.is_as_turn else board.eggs_player
            self_turds = board.turds_player if board.is_as_turn else board.turds_enemy
            enemy_turds = board.turds_enemy if board.is_as_turn else board.turds_player
        else:
            self_chicken = board.chicken_enemy if board.is_as_turn else board.chicken_player
            enemy_chicken = board.chicken_player if board.is_as_turn else board.chicken_enemy
            self_eggs = board.eggs_enemy if board.is_as_turn else board.eggs_player
            enemy_eggs = board.eggs_player if board.is_as_turn else board.eggs_enemy
            self_turds = board.turds_enemy if board.is_as_turn else board.turds_player
            enemy_turds = board.turds_player if board.is_as_turn else board.turds_enemy
        
        px, py = self_chicken.get_location()
        ex, ey = enemy_chicken.get_location()
        
        # Channel 1-2: Positions
        state[1, py, px] = 1.0
        state[2, ey, ex] = 1.0
        
        # Channel 3-4: Eggs
        for (x, y) in self_eggs:
            state[3, y, x] = 1.0
        for (x, y) in enemy_eggs:
            state[4, y, x] = 1.0
        
        # Channel 5-6: Turds
        for (x, y) in self_turds:
            state[5, y, x] = 1.0
        for (x, y) in enemy_turds:
            state[6, y, x] = 1.0
        
        # Channel 7: Known traps
        for loc in getattr(board, "found_trapdoors", set()):
            self.known_traps.add(loc)
        for (x, y) in self.known_traps:
            state[7, y, x] = 1.0
        
        # Channel 8-9: Distance maps
        for y in range(size):
            for x in range(size):
                state[8, y, x] = 1.0 - (abs(x - px) + abs(y - py)) / (2 * size)
                state[9, y, x] = 1.0 - (abs(x - ex) + abs(y - ey)) / (2 * size)
        
        # Scalars
        if self.is_player_a:
            self_turns = board.turns_left_player if board.is_as_turn else board.turns_left_enemy
            enemy_turns = board.turns_left_enemy if board.is_as_turn else board.turns_left_player
            self_eggs_count = len(self_eggs)
            enemy_eggs_count = len(enemy_eggs)
            self_turds_left = self_chicken.turds_left
            enemy_turds_left = enemy_chicken.turds_left
        else:
            self_turns = board.turns_left_enemy if board.is_as_turn else board.turns_left_player
            enemy_turns = board.turns_left_player if board.is_as_turn else board.turns_left_enemy
            self_eggs_count = len(self_eggs)
            enemy_eggs_count = len(enemy_eggs)
            self_turds_left = self_chicken.turds_left
            enemy_turds_left = enemy_chicken.turds_left
        
        scalars = np.array([
            self_turns / 40.0,
            enemy_turns / 40.0,
            self_eggs_count / 40.0,
            enemy_eggs_count / 40.0,
            self_turds_left / 5.0,
            enemy_turds_left / 5.0,
            len(self.known_traps) / 2.0,
            self.turn_count / 40.0,
        ], dtype=np.float32)
        
        return state, scalars
    
    def get_legal_moves(self, board: Board) -> List[int]:
        """Get indices of legal moves."""
        if self.is_player_a:
            chicken = board.chicken_player if board.is_as_turn else board.chicken_enemy
            turds = board.turds_player if board.is_as_turn else board.turds_enemy
            enemy_turds = board.turds_enemy if board.is_as_turn else board.turds_player
        else:
            chicken = board.chicken_enemy if board.is_as_turn else board.chicken_player
            turds = board.turds_enemy if board.is_as_turn else board.turds_player
            enemy_turds = board.turds_player if board.is_as_turn else board.turds_enemy
        
        cur = chicken.get_location()
        size = board.game_map.MAP_SIZE
        legal = []
        
        for idx, (dir_str, move_str) in enumerate(ACTION_SPACE):
            direction = DIR_MAP[dir_str]
            move_type = MOVE_MAP[move_str]
            
            dx, dy = DIR_DELTA[dir_str]
            nx, ny = cur[0] + dx, cur[1] + dy
            
            # Check bounds
            if not (0 <= nx < size and 0 <= ny < size):
                continue
            
            # Check obstacles
            if (nx, ny) in turds or (nx, ny) in enemy_turds:
                continue
            
            # Check move type validity
            if move_type == MoveType.EGG:
                # Can lay egg if we haven't laid more eggs than we have moves
                # Use board's can_lay_egg method
                if not board.can_lay_egg():
                    continue
            elif move_type == MoveType.TURD:
                if chicken.turds_left <= 0:
                    continue
            
            legal.append(idx)
        
        return legal if legal else [0]  # Fallback
    
    def select_action(self, board: Board) -> Tuple[Direction, MoveType]:
        """Select action using policy network."""
        self.turn_count += 1
        
        # Featurize
        state_board, state_scalar = self.featurize(board)
        
        # Get legal moves
        legal_indices = self.get_legal_moves(board)
        
        # Forward pass
        board_tensor = torch.from_numpy(state_board).unsqueeze(0).to(self.device)
        scalar_tensor = torch.from_numpy(state_scalar).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            policy_logits, value = self.model(board_tensor, scalar_tensor)
        
        # Mask illegal moves
        mask = torch.full((12,), float('-inf'), device=self.device)
        for idx in legal_indices:
            mask[idx] = 0.0
        
        masked_logits = policy_logits[0] + mask
        
        # Apply temperature
        if self.temperature != 1.0:
            masked_logits = masked_logits / self.temperature
        
        # Sample action
        probs = F.softmax(masked_logits, dim=0)
        dist = torch.distributions.Categorical(probs)
        action_idx = dist.sample().item()
        log_prob = dist.log_prob(torch.tensor(action_idx, device=self.device)).item()
        
        # Collect experience
        if self.collect_experiences:
            exp = Experience(
                state_board=state_board,
                state_scalar=state_scalar,
                action_idx=action_idx,
                log_prob=log_prob,
                value=value[0].item(),
            )
            self.experiences.append(exp)
        
        # Convert to game action
        dir_str, move_str = IDX_TO_ACTION[action_idx]
        return DIR_MAP[dir_str], MOVE_MAP[move_str]


def play_self_play_game(
    model: nn.Module,
    device: torch.device,
    temperature: float = 1.0,
) -> GameResult:
    """Play a single self-play game and collect experiences."""
    
    # Create agents - both see themselves as "player" from their perspective
    agent_a = SelfPlayAgent(model, device, is_player_a=True, temperature=temperature)
    agent_b = SelfPlayAgent(model, device, is_player_a=True, temperature=temperature)  # Also True - sees self as player
    
    # Setup game
    game_map = GameMap()
    trapdoor_manager = TrapdoorManager(game_map)
    board = Board(game_map, time_to_play=360, build_history=True)
    
    spawns = trapdoor_manager.choose_spawns()
    trapdoor_manager.choose_trapdoors()
    board.chicken_player.start(spawns[0], 0)
    board.chicken_enemy.start(spawns[1], 1)
    
    # Track original A's eggs for final scoring
    eggs_a_final = 0
    eggs_b_final = 0
    
    # Play game
    max_turns = 80
    for turn in range(max_turns):
        # Get current player's view
        is_a_turn = board.is_as_turn
        current_agent = agent_a if is_a_turn else agent_b
        
        # If it's B's turn, reverse perspective so B sees itself as player
        if not is_a_turn:
            board.reverse_perspective()
        
        try:
            direction, move_type = current_agent.select_action(board)
        except Exception as e:
            direction, move_type = Direction.UP, MoveType.PLAIN
        
        # Execute move
        success = board.apply_move(direction, move_type, timer=0.0, check_ok=True)
        
        # If move failed, try a fallback
        if not success:
            for fallback_dir in [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]:
                if board.apply_move(fallback_dir, MoveType.PLAIN, timer=0.0, check_ok=True):
                    break
        
        # Reverse back if we reversed for B
        if not is_a_turn:
            board.reverse_perspective()
        
        # Check for game end
        if board.is_game_over():
            break
    
    # Determine winner from A's perspective (A is original chicken_player)
    # After game, board might be in either perspective depending on last turn
    # Reset to A's perspective
    if not board.is_as_turn:
        # Currently in B's turn state, so board is from A's view
        eggs_a_final = board.chicken_player.get_eggs_laid()
        eggs_b_final = board.chicken_enemy.get_eggs_laid()
    else:
        # Currently in A's turn state
        eggs_a_final = board.chicken_player.get_eggs_laid()
        eggs_b_final = board.chicken_enemy.get_eggs_laid()
    
    if eggs_a_final > eggs_b_final:
        winner = 1
    elif eggs_b_final > eggs_a_final:
        winner = -1
    else:
        winner = 0
    
    # Assign rewards
    # Winner gets +1, loser gets -1, tie gets 0
    # Also add small reward shaping based on egg differential
    eggs_a = eggs_a_final
    eggs_b = eggs_b_final
    egg_diff_a = (eggs_a - eggs_b) / 40.0  # Normalized
    egg_diff_b = -egg_diff_a
    
    for exp in agent_a.experiences:
        if winner == 1:
            exp.reward = 1.0 + 0.2 * egg_diff_a
        elif winner == -1:
            exp.reward = -1.0 + 0.2 * egg_diff_a
        else:
            exp.reward = 0.1 * egg_diff_a
    
    for exp in agent_b.experiences:
        if winner == -1:
            exp.reward = 1.0 + 0.2 * egg_diff_b
        elif winner == 1:
            exp.reward = -1.0 + 0.2 * egg_diff_b
        else:
            exp.reward = 0.1 * egg_diff_b
    
    return GameResult(
        experiences_a=agent_a.experiences,
        experiences_b=agent_b.experiences,
        winner=winner,
        eggs_a=eggs_a,
        eggs_b=eggs_b,
        turns=board.turn_count,
        reason=str(board.win_reason) if board.win_reason else "CONTINUE",
    )


class ExperienceBuffer:
    """Replay buffer for experiences."""
    
    def __init__(self, max_size: int = 100000):
        self.buffer: deque = deque(maxlen=max_size)
    
    def add_game(self, result: GameResult):
        """Add all experiences from a game."""
        for exp in result.experiences_a:
            self.buffer.append(exp)
        for exp in result.experiences_b:
            self.buffer.append(exp)
    
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample a batch of experiences."""
        return random.sample(list(self.buffer), min(batch_size, len(self.buffer)))
    
    def get_all(self) -> List[Experience]:
        """Get all experiences."""
        return list(self.buffer)
    
    def clear(self):
        """Clear the buffer."""
        self.buffer.clear()
    
    def __len__(self):
        return len(self.buffer)


def compute_returns(experiences: List[Experience], gamma: float = 0.99) -> List[float]:
    """Compute discounted returns for a sequence of experiences."""
    returns = []
    G = 0.0
    for exp in reversed(experiences):
        G = exp.reward + gamma * G
        returns.insert(0, G)
    return returns


def train_policy_gradient(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    experiences: List[Experience],
    device: torch.device,
    gamma: float = 0.99,
    entropy_coef: float = 0.01,
    value_coef: float = 0.5,
) -> Dict[str, float]:
    """Train using policy gradient (REINFORCE with baseline)."""
    
    if len(experiences) == 0:
        return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
    
    model.train()
    
    # Prepare batches
    boards = torch.stack([torch.from_numpy(e.state_board) for e in experiences]).to(device)
    scalars = torch.stack([torch.from_numpy(e.state_scalar) for e in experiences]).to(device)
    actions = torch.tensor([e.action_idx for e in experiences], device=device)
    old_log_probs = torch.tensor([e.log_prob for e in experiences], device=device)
    returns = torch.tensor([e.reward for e in experiences], dtype=torch.float32, device=device)
    
    # Normalize returns
    if len(returns) > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    
    # Forward pass
    policy_logits, values = model(boards, scalars)
    
    # Policy loss (REINFORCE)
    log_probs = F.log_softmax(policy_logits, dim=1)
    action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
    
    # Advantage = return - baseline (value estimate)
    advantages = returns - values.detach()
    policy_loss = -(action_log_probs * advantages).mean()
    
    # Value loss
    value_loss = F.mse_loss(values, returns)
    
    # Entropy bonus (encourages exploration)
    probs = F.softmax(policy_logits, dim=1)
    entropy = -(probs * log_probs).sum(dim=1).mean()
    
    # Total loss
    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    return {
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "entropy": entropy.item(),
        "total_loss": loss.item(),
    }


def evaluate_against_baseline(
    model: nn.Module,
    device: torch.device,
    num_games: int = 20,
) -> Dict[str, float]:
    """Evaluate current model in self-play (as proxy for strength)."""
    
    wins = 0
    losses = 0
    ties = 0
    total_eggs = 0
    
    model.eval()
    
    for _ in range(num_games):
        result = play_self_play_game(model, device, temperature=0.5)
        
        if result.winner == 1:
            wins += 1
        elif result.winner == -1:
            losses += 1
        else:
            ties += 1
        
        total_eggs += result.eggs_a + result.eggs_b
    
    return {
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "avg_eggs": total_eggs / (2 * num_games),
        "win_rate": wins / num_games,
    }


def self_play_training(
    model: nn.Module,
    device: torch.device,
    num_iterations: int = 1000,
    games_per_iteration: int = 50,
    batch_size: int = 256,
    lr: float = 1e-4,
    gamma: float = 0.99,
    entropy_coef: float = 0.01,
    value_coef: float = 0.5,
    temperature_start: float = 1.0,
    temperature_end: float = 0.3,
    eval_interval: int = 10,
    save_interval: int = 50,
    output_dir: str = "agents/Ronaldo",
    verbose: bool = True,
):
    """Main self-play training loop."""
    
    optimizer = Adam(model.parameters(), lr=lr)
    buffer = ExperienceBuffer(max_size=200000)
    
    best_avg_eggs = 0.0
    
    # Training history
    history = {
        "iterations": [],
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
        "avg_eggs": [],
        "temperature": [],
    }
    
    for iteration in range(num_iterations):
        iter_start = time.time()
        
        # Anneal temperature
        progress = iteration / num_iterations
        temperature = temperature_start + (temperature_end - temperature_start) * progress
        
        # Generate self-play games
        model.eval()
        iteration_experiences = []
        total_eggs = 0
        wins_a = 0
        
        for game_idx in range(games_per_iteration):
            result = play_self_play_game(model, device, temperature=temperature)
            buffer.add_game(result)
            iteration_experiences.extend(result.experiences_a)
            iteration_experiences.extend(result.experiences_b)
            total_eggs += result.eggs_a + result.eggs_b
            if result.winner == 1:
                wins_a += 1
        
        avg_eggs = total_eggs / (2 * games_per_iteration)
        
        # Train on collected experiences
        model.train()
        
        # Multiple training passes on the buffer
        train_metrics = {"policy_loss": 0, "value_loss": 0, "entropy": 0, "total_loss": 0}
        num_updates = max(1, len(iteration_experiences) // batch_size)
        
        for _ in range(num_updates):
            batch = buffer.sample(batch_size)
            metrics = train_policy_gradient(
                model, optimizer, batch, device,
                gamma=gamma, entropy_coef=entropy_coef, value_coef=value_coef,
            )
            for k, v in metrics.items():
                train_metrics[k] += v / num_updates
        
        # Record history
        history["iterations"].append(iteration)
        history["policy_loss"].append(train_metrics["policy_loss"])
        history["value_loss"].append(train_metrics["value_loss"])
        history["entropy"].append(train_metrics["entropy"])
        history["avg_eggs"].append(avg_eggs)
        history["temperature"].append(temperature)
        
        iter_time = time.time() - iter_start
        
        if verbose:
            print(
                f"Iter {iteration+1}/{num_iterations} | "
                f"Games: {games_per_iteration} | "
                f"Eggs: {avg_eggs:.1f} | "
                f"P_Loss: {train_metrics['policy_loss']:.4f} | "
                f"V_Loss: {train_metrics['value_loss']:.4f} | "
                f"Entropy: {train_metrics['entropy']:.4f} | "
                f"Temp: {temperature:.2f} | "
                f"Buffer: {len(buffer)} | "
                f"Time: {iter_time:.1f}s"
            )
        
        # Periodic evaluation
        if (iteration + 1) % eval_interval == 0:
            eval_metrics = evaluate_against_baseline(model, device, num_games=20)
            if verbose:
                print(f"  -> Eval: Avg eggs={eval_metrics['avg_eggs']:.1f}")
            
            # Save if improved
            if eval_metrics['avg_eggs'] > best_avg_eggs:
                best_avg_eggs = eval_metrics['avg_eggs']
                save_path = os.path.join(output_dir, "ronaldo_selfplay_best.pt")
                torch.save({
                    'iteration': iteration,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'avg_eggs': best_avg_eggs,
                }, save_path)
                if verbose:
                    print(f"  -> Saved best model (avg_eggs: {best_avg_eggs:.1f})")
        
        # Periodic checkpoint
        if (iteration + 1) % save_interval == 0:
            save_path = os.path.join(output_dir, f"ronaldo_selfplay_iter{iteration+1}.pt")
            torch.save({
                'iteration': iteration,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history,
            }, save_path)
            if verbose:
                print(f"  -> Checkpoint saved")
    
    # Final save
    save_path = os.path.join(output_dir, "ronaldo_selfplay_final.pt")
    torch.save({
        'iteration': num_iterations,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
    }, save_path)
    
    return history


def main():
    parser = argparse.ArgumentParser(description="Self-play RL training for Ronaldo")
    parser.add_argument("--weights", type=str, default="agents/Ronaldo/ronaldo_weights.pt",
                        help="Initial weights to start from")
    parser.add_argument("--output", type=str, default="agents/Ronaldo",
                        help="Output directory")
    parser.add_argument("--iterations", type=int, default=500,
                        help="Number of training iterations")
    parser.add_argument("--games", type=int, default=50,
                        help="Games per iteration")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--entropy_coef", type=float, default=0.01,
                        help="Entropy coefficient")
    parser.add_argument("--value_coef", type=float, default=0.5,
                        help="Value loss coefficient")
    parser.add_argument("--temp_start", type=float, default=1.0,
                        help="Starting temperature")
    parser.add_argument("--temp_end", type=float, default=0.3,
                        help="Ending temperature")
    parser.add_argument("--eval_interval", type=int, default=10,
                        help="Evaluation interval")
    parser.add_argument("--save_interval", type=int, default=50,
                        help="Checkpoint save interval")
    parser.add_argument("--small", action="store_true",
                        help="Use smaller network")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use")
    
    args = parser.parse_args()
    
    # Device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load or create model
    if os.path.exists(args.weights):
        print(f"Loading weights from {args.weights}")
        checkpoint = torch.load(args.weights, map_location=device)
        use_small = checkpoint.get('use_small_net', args.small)
        
        if use_small:
            model = RonaldoNetSmall(board_size=8, in_channels=10, channels=64)
        else:
            model = RonaldoNet(board_size=8, in_channels=10, channels=128, num_res_blocks=4)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model (val_acc: {checkpoint.get('val_acc', 'N/A')})")
    else:
        print("No weights found, starting from scratch")
        if args.small:
            model = RonaldoNetSmall(board_size=8, in_channels=10, channels=64)
        else:
            model = RonaldoNet(board_size=8, in_channels=10, channels=128, num_res_blocks=4)
    
    model = model.to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    # Run self-play training
    print("\n" + "="*50)
    print("Starting Self-Play Reinforcement Learning")
    print("="*50 + "\n")
    
    history = self_play_training(
        model=model,
        device=device,
        num_iterations=args.iterations,
        games_per_iteration=args.games,
        batch_size=args.batch_size,
        lr=args.lr,
        gamma=args.gamma,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        temperature_start=args.temp_start,
        temperature_end=args.temp_end,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        output_dir=args.output,
    )
    
    print("\n" + "="*50)
    print("Self-Play Training Complete!")
    print("="*50)


if __name__ == "__main__":
    main()

