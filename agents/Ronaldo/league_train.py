"""
League Training for Ronaldo Agent.

Trains against a pool of strong opponents (Messi, Pele, Fluffy) + self-play.
This prevents mode collapse and learns robust strategies.
"""
from __future__ import annotations

import argparse
import copy
import importlib
import os
import random
import sys
import time
from collections import deque
from dataclasses import dataclass
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

# Import game engine
from game.board import Board
from game.game_map import GameMap
from game.trapdoor_manager import TrapdoorManager
from game.enums import Direction, MoveType, Result

# Direction/Move mappings
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
    """Single experience tuple."""
    state_board: np.ndarray
    state_scalar: np.ndarray
    action_idx: int
    log_prob: float
    reward: float = 0.0
    value: float = 0.0


@dataclass 
class GameResult:
    """Result of a training game."""
    experiences: List[Experience]  # Only Ronaldo's experiences
    won: bool
    eggs_ronaldo: int
    eggs_opponent: int
    opponent_name: str
    turns: int


class RonaldoPlayer:
    """Ronaldo agent wrapper for league training."""
    
    def __init__(self, model: nn.Module, device: torch.device, temperature: float = 1.0):
        self.model = model
        self.device = device
        self.temperature = temperature
        self.experiences: List[Experience] = []
        self.turn_count = 0
        self.known_traps: set = set()
    
    def reset(self):
        self.experiences = []
        self.turn_count = 0
        self.known_traps = set()
    
    def featurize(self, board: Board) -> Tuple[np.ndarray, np.ndarray]:
        """Convert board to neural network input."""
        size = board.game_map.MAP_SIZE
        state = np.zeros((10, size, size), dtype=np.float32)
        
        state[0, :, :] = 1.0  # Board mask
        
        px, py = board.chicken_player.get_location()
        ex, ey = board.chicken_enemy.get_location()
        state[1, py, px] = 1.0
        state[2, ey, ex] = 1.0
        
        for (x, y) in board.eggs_player:
            state[3, y, x] = 1.0
        for (x, y) in board.eggs_enemy:
            state[4, y, x] = 1.0
        for (x, y) in board.turds_player:
            state[5, y, x] = 1.0
        for (x, y) in board.turds_enemy:
            state[6, y, x] = 1.0
        
        for loc in getattr(board, "found_trapdoors", set()):
            self.known_traps.add(loc)
        for (x, y) in self.known_traps:
            state[7, y, x] = 1.0
        
        for y in range(size):
            for x in range(size):
                state[8, y, x] = 1.0 - (abs(x - px) + abs(y - py)) / (2 * size)
                state[9, y, x] = 1.0 - (abs(x - ex) + abs(y - ey)) / (2 * size)
        
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
        
        return state, scalars
    
    def get_legal_moves(self, board: Board) -> List[int]:
        """Get legal move indices."""
        chicken = board.chicken_player
        cur = chicken.get_location()
        size = board.game_map.MAP_SIZE
        legal = []
        
        for idx, (dir_str, move_str) in enumerate(ACTION_SPACE):
            dx, dy = DIR_DELTA[dir_str]
            nx, ny = cur[0] + dx, cur[1] + dy
            
            if not (0 <= nx < size and 0 <= ny < size):
                continue
            if (nx, ny) in board.turds_player or (nx, ny) in board.turds_enemy:
                continue
            
            move_type = MOVE_MAP[move_str]
            if move_type == MoveType.EGG and not board.can_lay_egg():
                continue
            if move_type == MoveType.TURD and chicken.turds_left <= 0:
                continue
            
            legal.append(idx)
        
        return legal if legal else [0]
    
    def select_action(self, board: Board) -> Tuple[Direction, MoveType]:
        """Select action and record experience."""
        self.turn_count += 1
        
        state_board, state_scalar = self.featurize(board)
        legal_indices = self.get_legal_moves(board)
        
        board_tensor = torch.from_numpy(state_board).unsqueeze(0).to(self.device)
        scalar_tensor = torch.from_numpy(state_scalar).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            policy_logits, value = self.model(board_tensor, scalar_tensor)
        
        # Mask illegal moves
        mask = torch.full((12,), float('-inf'), device=self.device)
        for idx in legal_indices:
            mask[idx] = 0.0
        
        masked_logits = policy_logits[0] + mask
        
        if self.temperature != 1.0:
            masked_logits = masked_logits / self.temperature
        
        probs = F.softmax(masked_logits, dim=0)
        dist = torch.distributions.Categorical(probs)
        action_idx = dist.sample().item()
        log_prob = dist.log_prob(torch.tensor(action_idx, device=self.device)).item()
        
        # Record experience
        exp = Experience(
            state_board=state_board,
            state_scalar=state_scalar,
            action_idx=action_idx,
            log_prob=log_prob,
            value=value[0].item(),
        )
        self.experiences.append(exp)
        
        dir_str, move_str = IDX_TO_ACTION[action_idx]
        return DIR_MAP[dir_str], MOVE_MAP[move_str]


def load_opponent_agent(agent_name: str, board: Board):
    """Load an opponent agent."""
    try:
        module = importlib.import_module(f"agents.{agent_name}.agent")
        def time_left():
            return 360.0
        agent = module.PlayerAgent(board, time_left)
        return agent
    except Exception as e:
        print(f"Failed to load {agent_name}: {e}")
        return None


def play_self_play_game(
    ronaldo_model: nn.Module,
    device: torch.device,
    temperature: float = 1.0,
) -> Optional[GameResult]:
    """Play a game against itself (self-play).
    
    Both players use the same model. We collect experiences from player A's perspective.
    """
    
    # Setup game
    game_map = GameMap()
    trapdoor_manager = TrapdoorManager(game_map)
    board = Board(game_map, time_to_play=360, build_history=True)
    
    spawns = trapdoor_manager.choose_spawns()
    trapdoor_manager.choose_trapdoors()
    board.chicken_player.start(spawns[0], 0)
    board.chicken_enemy.start(spawns[1], 1)
    
    # Both players use same model
    player_a = RonaldoPlayer(ronaldo_model, device, temperature)
    player_b = RonaldoPlayer(ronaldo_model, device, temperature)
    
    current_is_a = True
    
    max_turns = 80
    for turn in range(max_turns):
        if current_is_a:
            direction, move_type = player_a.select_action(board)
        else:
            direction, move_type = player_b.select_action(board)
        
        success = board.apply_move(direction, move_type, timer=0.0, check_ok=True)
        if not success:
            for fallback_dir in [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]:
                if board.apply_move(fallback_dir, MoveType.PLAIN, timer=0.0, check_ok=True):
                    break
        
        if board.is_game_over():
            break
        
        board.reverse_perspective()
        current_is_a = not current_is_a
    
    # Get final eggs
    eggs_current_player = board.chicken_player.get_eggs_laid()
    eggs_current_enemy = board.chicken_enemy.get_eggs_laid()
    
    if current_is_a:
        eggs_a = eggs_current_player
        eggs_b = eggs_current_enemy
    else:
        eggs_a = eggs_current_enemy
        eggs_b = eggs_current_player
    
    # We track from player A's perspective
    won = eggs_a > eggs_b
    
    if won:
        reward = 1.0 + 0.1 * (eggs_a - eggs_b) / 40.0
    elif eggs_a == eggs_b:
        reward = 0.0
    else:
        reward = -1.0 + 0.1 * (eggs_a - eggs_b) / 40.0
    
    for exp in player_a.experiences:
        exp.reward = reward
    
    return GameResult(
        experiences=player_a.experiences,
        won=won,
        eggs_ronaldo=eggs_a,
        eggs_opponent=eggs_b,
        opponent_name="Self",
        turns=board.turn_count,
    )


def play_against_opponent(
    ronaldo_model: nn.Module,
    device: torch.device,
    opponent_name: str,
    ronaldo_is_player_a: bool,
    temperature: float = 1.0,
) -> Optional[GameResult]:
    """Play a game against a specific opponent.
    
    The game engine works as follows:
    - Board perspective is swapped after EACH turn (via reverse_perspective)
    - apply_move always operates on chicken_player (the current player's view)
    - Each player always sees themselves as chicken_player when it's their turn
    
    So we need to:
    1. Give the current player the board (they see themselves as chicken_player)
    2. Get their move
    3. Apply the move
    4. Reverse perspective for the next player
    """
    
    # Handle self-play specially
    if opponent_name == "Self":
        return play_self_play_game(ronaldo_model, device, temperature)
    
    # Setup game
    game_map = GameMap()
    trapdoor_manager = TrapdoorManager(game_map)
    board = Board(game_map, time_to_play=360, build_history=True)
    
    spawns = trapdoor_manager.choose_spawns()
    trapdoor_manager.choose_trapdoors()
    board.chicken_player.start(spawns[0], 0)
    board.chicken_enemy.start(spawns[1], 1)
    
    # Create Ronaldo player
    ronaldo = RonaldoPlayer(ronaldo_model, device, temperature)
    
    # Track who is currently "player" (chicken_player) from board's perspective
    # Initially, A is chicken_player
    current_is_a = True
    
    # Load opponent - they need their own board copy for initialization
    # When opponent is B, they'll be initialized with reversed board
    if ronaldo_is_player_a:
        # Opponent is B - give them a reversed board for init
        init_board = board.get_copy()
        init_board.reverse_perspective()
        opponent = load_opponent_agent(opponent_name, init_board)
    else:
        # Opponent is A - give them the normal board for init
        opponent = load_opponent_agent(opponent_name, board.get_copy())
    
    if opponent is None:
        return None
    
    # Track eggs laid by each side (A and B)
    # We'll track this independently since perspective swaps
    
    # Play game
    max_turns = 80
    for turn in range(max_turns):
        # current_is_a tells us who is currently chicken_player
        ronaldo_turn = (ronaldo_is_player_a == current_is_a)
        
        if ronaldo_turn:
            # Ronaldo's turn - board already shows Ronaldo as chicken_player
            try:
                direction, move_type = ronaldo.select_action(board)
            except:
                direction, move_type = Direction.UP, MoveType.PLAIN
        else:
            # Opponent's turn - board already shows opponent as chicken_player
            try:
                sensor_data = [(False, False), (False, False)]
                def time_left():
                    return 360.0
                direction, move_type = opponent.play(board, sensor_data, time_left)
            except Exception as e:
                direction, move_type = Direction.UP, MoveType.PLAIN
        
        # Apply move
        success = board.apply_move(direction, move_type, timer=0.0, check_ok=True)
        if not success:
            for fallback_dir in [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]:
                if board.apply_move(fallback_dir, MoveType.PLAIN, timer=0.0, check_ok=True):
                    break
        
        if board.is_game_over():
            break
        
        # Swap perspective for next turn (like the real engine does)
        board.reverse_perspective()
        current_is_a = not current_is_a
    
    # Determine result
    # After the game, we need to figure out the final egg counts
    # The board's current perspective depends on how many turns were played
    # chicken_player is whoever would play next
    
    # Get eggs from current perspective
    eggs_current_player = board.chicken_player.get_eggs_laid()
    eggs_current_enemy = board.chicken_enemy.get_eggs_laid()
    
    # Map to A and B
    if current_is_a:
        eggs_a = eggs_current_player
        eggs_b = eggs_current_enemy
    else:
        eggs_a = eggs_current_enemy
        eggs_b = eggs_current_player
    
    if ronaldo_is_player_a:
        eggs_ronaldo = eggs_a
        eggs_opponent = eggs_b
    else:
        eggs_ronaldo = eggs_b
        eggs_opponent = eggs_a
    
    won = eggs_ronaldo > eggs_opponent
    
    # Assign rewards
    if won:
        reward = 1.0 + 0.1 * (eggs_ronaldo - eggs_opponent) / 40.0
    elif eggs_ronaldo == eggs_opponent:
        reward = 0.0
    else:
        reward = -1.0 + 0.1 * (eggs_ronaldo - eggs_opponent) / 40.0
    
    for exp in ronaldo.experiences:
        exp.reward = reward
    
    return GameResult(
        experiences=ronaldo.experiences,
        won=won,
        eggs_ronaldo=eggs_ronaldo,
        eggs_opponent=eggs_opponent,
        opponent_name=opponent_name,
        turns=board.turn_count,
    )


class ExperienceBuffer:
    """Replay buffer."""
    
    def __init__(self, max_size: int = 100000):
        self.buffer: deque = deque(maxlen=max_size)
    
    def add_game(self, result: GameResult):
        for exp in result.experiences:
            self.buffer.append(exp)
    
    def sample(self, batch_size: int) -> List[Experience]:
        return random.sample(list(self.buffer), min(batch_size, len(self.buffer)))
    
    def clear(self):
        self.buffer.clear()
    
    def __len__(self):
        return len(self.buffer)


def train_policy_gradient(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    experiences: List[Experience],
    device: torch.device,
    entropy_coef: float = 0.05,
    value_coef: float = 0.5,
) -> Dict[str, float]:
    """Train using policy gradient with higher entropy."""
    
    if len(experiences) == 0:
        return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
    
    model.train()
    
    boards = torch.stack([torch.from_numpy(e.state_board) for e in experiences]).to(device)
    scalars = torch.stack([torch.from_numpy(e.state_scalar) for e in experiences]).to(device)
    actions = torch.tensor([e.action_idx for e in experiences], device=device)
    returns = torch.tensor([e.reward for e in experiences], dtype=torch.float32, device=device)
    
    # Normalize returns
    if len(returns) > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    
    policy_logits, values = model(boards, scalars)
    
    log_probs = F.log_softmax(policy_logits, dim=1)
    action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
    
    advantages = returns - values.detach()
    policy_loss = -(action_log_probs * advantages).mean()
    
    value_loss = F.mse_loss(values, returns)
    
    # Higher entropy bonus
    probs = F.softmax(policy_logits, dim=1)
    entropy = -(probs * log_probs).sum(dim=1).mean()
    
    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # Smaller clip
    optimizer.step()
    
    return {
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "entropy": entropy.item(),
    }


def league_training(
    model: nn.Module,
    device: torch.device,
    opponents: List[str],
    num_iterations: int = 500,
    games_per_opponent: int = 10,
    batch_size: int = 256,
    lr: float = 5e-5,  # Lower LR
    entropy_coef: float = 0.05,  # Higher entropy
    value_coef: float = 0.5,
    temperature: float = 0.8,  # Fixed temperature, no annealing
    eval_interval: int = 20,
    save_interval: int = 50,
    output_dir: str = "agents/Ronaldo",
    verbose: bool = True,
):
    """Main league training loop."""
    
    optimizer = Adam(model.parameters(), lr=lr)
    buffer = ExperienceBuffer(max_size=100000)
    
    best_win_rate = 0.0
    best_model_state = None
    consecutive_bad = 0
    
    # Stats tracking
    history = {
        "iterations": [],
        "win_rates": [],
        "avg_eggs": [],
        "entropy": [],
    }
    
    for iteration in range(num_iterations):
        iter_start = time.time()
        
        # Play games against each opponent
        model.eval()
        wins = {opp: 0 for opp in opponents}
        total_eggs = 0
        games_played = 0
        
        for opponent in opponents:
            for game_idx in range(games_per_opponent):
                # Alternate sides
                ronaldo_is_a = (game_idx % 2 == 0)
                
                result = play_against_opponent(
                    model, device, opponent, ronaldo_is_a, temperature
                )
                
                if result is not None:
                    buffer.add_game(result)
                    if result.won:
                        wins[opponent] += 1
                    total_eggs += result.eggs_ronaldo
                    games_played += 1
        
        # Calculate stats
        total_wins = sum(wins.values())
        win_rate = total_wins / max(games_played, 1)
        avg_eggs = total_eggs / max(games_played, 1)
        
        # Train - but only do a FEW updates to prevent catastrophic forgetting
        model.train()
        train_metrics = {"policy_loss": 0, "value_loss": 0, "entropy": 0}
        
        if len(buffer) >= batch_size:
            # Only 2-3 gradient updates per iteration to be conservative
            num_updates = 3
            for _ in range(num_updates):
                batch = buffer.sample(batch_size)
                metrics = train_policy_gradient(
                    model, optimizer, batch, device,
                    entropy_coef=entropy_coef, value_coef=value_coef,
                )
                for k, v in metrics.items():
                    train_metrics[k] += v / num_updates
        
        # Record history
        history["iterations"].append(iteration)
        history["win_rates"].append(win_rate)
        history["avg_eggs"].append(avg_eggs)
        history["entropy"].append(train_metrics["entropy"])
        
        iter_time = time.time() - iter_start
        
        if verbose:
            wins_str = " ".join([f"{opp[:3]}:{wins[opp]}/{games_per_opponent}" for opp in opponents])
            print(
                f"Iter {iteration+1}/{num_iterations} | "
                f"WinRate: {win_rate:.1%} | "
                f"Eggs: {avg_eggs:.1f} | "
                f"Wins: [{wins_str}] | "
                f"Ent: {train_metrics['entropy']:.4f} | "
                f"Buf: {len(buffer)} | "
                f"Time: {iter_time:.1f}s"
            )
        
        # Save best model whenever we improve
        if win_rate > best_win_rate:
            best_win_rate = win_rate
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            save_path = os.path.join(output_dir, "ronaldo_league_best.pt")
            torch.save({
                'iteration': iteration,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'win_rate': best_win_rate,
                'avg_eggs': avg_eggs,
            }, save_path)
            if verbose:
                print(f"  -> Saved best model (win_rate: {best_win_rate:.1%})")
            consecutive_bad = 0
        else:
            consecutive_bad += 1
        
        # CRITICAL: Reload best model if performance collapses
        # This prevents catastrophic forgetting
        if consecutive_bad >= 10 and best_model_state is not None:
            if verbose:
                print(f"  -> Performance collapsed! Reloading best model (was {best_win_rate:.1%})")
            model.load_state_dict(best_model_state)
            consecutive_bad = 0
            # Clear buffer to remove bad experiences
            buffer = ExperienceBuffer(max_size=100000)
        
        # Periodic checkpoint
        if (iteration + 1) % save_interval == 0:
            save_path = os.path.join(output_dir, f"ronaldo_league_iter{iteration+1}.pt")
            torch.save({
                'iteration': iteration,
                'model_state_dict': model.state_dict(),
                'history': history,
            }, save_path)
            if verbose:
                print(f"  -> Checkpoint saved")
    
    # Final save
    save_path = os.path.join(output_dir, "ronaldo_league_final.pt")
    torch.save({
        'iteration': num_iterations,
        'model_state_dict': model.state_dict(),
        'history': history,
    }, save_path)
    
    return history


def main():
    parser = argparse.ArgumentParser(description="League training for Ronaldo")
    parser.add_argument("--weights", type=str, default="agents/Ronaldo/ronaldo_weights.pt")
    parser.add_argument("--output", type=str, default="agents/Ronaldo")
    parser.add_argument("--opponents", type=str, nargs="+", default=["Messi", "Fluffy", "Pele"])
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--games_per_opp", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--entropy_coef", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--eval_interval", type=int, default=20)
    parser.add_argument("--save_interval", type=int, default=50)
    parser.add_argument("--device", type=str, default=None)
    
    args = parser.parse_args()
    
    # Device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load model
    if os.path.exists(args.weights):
        print(f"Loading weights from {args.weights}")
        checkpoint = torch.load(args.weights, map_location=device)
        model = RonaldoNet(board_size=8, in_channels=10, channels=128, num_res_blocks=4)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("No weights found, starting from scratch")
        model = RonaldoNet(board_size=8, in_channels=10, channels=128, num_res_blocks=4)
    
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Opponents: {args.opponents}")
    
    # Run training
    print("\n" + "="*60)
    print("Starting League Training")
    print("="*60 + "\n")
    
    history = league_training(
        model=model,
        device=device,
        opponents=args.opponents,
        num_iterations=args.iterations,
        games_per_opponent=args.games_per_opp,
        batch_size=args.batch_size,
        lr=args.lr,
        entropy_coef=args.entropy_coef,
        temperature=args.temperature,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        output_dir=args.output,
    )
    
    print("\n" + "="*60)
    print("League Training Complete!")
    print("="*60)


if __name__ == "__main__":
    main()

