from __future__ import annotations

import math
import os
import random
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENGINE_ROOT = PROJECT_ROOT / "engine"
if str(ENGINE_ROOT) not in sys.path:
    sys.path.append(str(ENGINE_ROOT))

from game.board import Board
from game.enums import Direction, MoveType, Result
from game.game_map import GameMap

from .trapdoor_belief import TrapdoorBelief
from .policy_utils import (
    ActionBiasParams,
    sample_action_with_bias,
    temperature_for_turn,
)
from .policy_utils import (
    ActionBiasParams,
    sample_action_with_bias,
    temperature_for_turn,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ALL_DIRECTIONS = [d for d in Direction]
ALL_MOVETYPES = [m for m in MoveType]
ACTION_SPACE = [(d, m) for d in ALL_DIRECTIONS for m in ALL_MOVETYPES]
ACTION_INDEX: Dict[Tuple[int, int], int] = {
    (int(d), int(m)): i for i, (d, m) in enumerate(ACTION_SPACE)
}
NUM_ACTIONS = len(ACTION_SPACE)

def encode_action(action: Tuple[Direction, MoveType]) -> int:
    d, m = action
    return ACTION_INDEX[(int(d), int(m))]


def decode_action(index: int) -> Tuple[Direction, MoveType]:
    d, m = ACTION_SPACE[index]
    return Direction(int(d)), MoveType(int(m))


class ChickenNet(nn.Module):
    """Shared convolutional policy + value network."""

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

        self.scalar_fc = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )

        flat_size = channels * board_size * board_size
        self.policy_head = nn.Sequential(
            nn.Linear(flat_size + 64, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, NUM_ACTIONS),
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


class MCTSNode:
    def __init__(self, parent: "MCTSNode | None", prior: float):
        self.parent = parent
        self.prior = prior
        self.children: Dict[int, "MCTSNode"] = {}
        self.visit_count = 0
        self.value_sum = 0.0

    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def expand(self, action_priors: Dict[int, float]) -> None:
        for a_idx, prob in action_priors.items():
            if a_idx not in self.children:
                self.children[a_idx] = MCTSNode(self, prob)

    def is_leaf(self) -> bool:
        return len(self.children) == 0


class MCTS:
    def __init__(self, net: ChickenNet, c_puct: float = 1.5):
        self.net = net
        self.c_puct = c_puct

    @staticmethod
    def _result_to_value(winner: Result) -> float:
        if winner == Result.PLAYER:
            return 1.0
        if winner == Result.ENEMY:
            return -1.0
        return 0.0

    def run(
        self,
        root_board: Board,
        root_belief: TrapdoorBelief,
        max_sims: int,
        root_dirichlet: Tuple[float, float] | None = None,
    ) -> np.ndarray:
        root = MCTSNode(parent=None, prior=1.0)
        root_terminal = self._evaluate_and_expand(root, root_board, root_belief)

        if root_terminal:
            return self._visit_distribution(root)

        if max_sims <= 0:
            return self._visit_distribution(root)

        # Add root Dirichlet noise for exploration during self-play if requested
        if root.children and root_dirichlet is not None:
            alpha, frac = root_dirichlet
            keys = list(root.children.keys())
            noise = np.random.dirichlet([alpha] * len(keys))
            for k, n in zip(keys, noise):
                child = root.children[k]
                child.prior = (1.0 - frac) * child.prior + frac * float(n)

        for _ in range(max_sims):
            board = root_board.get_copy(False, True)
            belief = root_belief.clone()
            node = root
            path = [node]

            while not node.is_leaf() and node.children:
                action_idx, node = self._select_child(node)
                action = decode_action(action_idx)
                board = self._next_state(board, action)
                belief = belief  # belief is static through simulations
                path.append(node)

            value, action_priors, terminal = self._evaluate(board, belief)
            if not terminal and action_priors:
                node.expand(action_priors)

            for visited in reversed(path):
                visited.visit_count += 1
                visited.value_sum += value
                value = -value

        return self._visit_distribution(root)

    def _select_child(self, node: MCTSNode) -> Tuple[int, MCTSNode]:
        total_visits = max(node.visit_count, 1)
        best_score = -1e9
        best = None
        best_child = None

        for a_idx, child in node.children.items():
            exploitation = child.value
            exploration = (
                self.c_puct
                * child.prior
                * math.sqrt(total_visits)
                / (1 + child.visit_count)
            )
            score = exploitation + exploration
            if score > best_score:
                best_score = score
                best = a_idx
                best_child = child

        assert best is not None and best_child is not None
        return best, best_child

    def _evaluate(
        self,
        board: Board,
        belief: TrapdoorBelief,
    ) -> Tuple[float, Dict[int, float] | None, bool]:
        winner = board.get_winner()
        if winner is not None:
            return self._result_to_value(winner), None, True

        board_tensor, scalar_tensor = make_state_tensors(board, belief)
        self.net.eval()
        with torch.no_grad():
            logits, value = self.net(
                board_tensor.to(DEVICE),
                scalar_tensor.to(DEVICE),
            )
        logits = logits.cpu().numpy()[0]
        value = float(value.cpu().numpy()[0])

        priors = self._legal_priors(board, logits)
        return value, priors, False

    def _evaluate_and_expand(
        self,
        node: MCTSNode,
        board: Board,
        belief: TrapdoorBelief,
    ) -> bool:
        value, priors, terminal = self._evaluate(board, belief)
        if not terminal and priors:
            node.expand(priors)
        node.value_sum += value
        node.visit_count += 1
        return terminal

    def _legal_priors(self, board: Board, logits: np.ndarray) -> Dict[int, float]:
        legal_moves = board.get_valid_moves()
        if not legal_moves:
            return {}

        priors = np.exp(logits - logits.max())
        priors /= np.sum(priors)

        action_priors: Dict[int, float] = {}
        total = 0.0
        legal_indices = [encode_action(move) for move in legal_moves]
        for idx in legal_indices:
            prob = float(priors[idx])
            action_priors[idx] = prob
            total += prob

        if total > 0:
            for idx in action_priors:
                action_priors[idx] /= total
        else:
            uniform = 1.0 / len(legal_indices)
            action_priors = {idx: uniform for idx in legal_indices}

        return action_priors

    def _next_state(self, board: Board, action: Tuple[Direction, MoveType]) -> Board:
        next_board = board.get_copy(False, True)
        valid = next_board.apply_move(action[0], action[1], check_ok=True)
        if not valid:
            return board
        next_board.reverse_perspective()
        return next_board

    def _visit_distribution(self, root: MCTSNode) -> np.ndarray:
        visits = np.zeros(NUM_ACTIONS, dtype=np.float32)
        for a_idx, child in root.children.items():
            visits[a_idx] = child.visit_count
        total = float(visits.sum())
        if total > 0:
            visits /= total
        else:
            visits.fill(1.0 / NUM_ACTIONS)
        return visits


def make_state_tensors(board: Board, belief: TrapdoorBelief) -> Tuple[torch.Tensor, torch.Tensor]:
    game_map: GameMap = board.game_map
    size = game_map.MAP_SIZE
    channels = 9
    state = np.zeros((channels, size, size), dtype=np.float32)

    for y in range(size):
        for x in range(size):
            state[0, y, x] = 1.0

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

    state[7:9, :, :] = belief.as_tensor().astype(np.float32)

    scalar = np.array(
        [
            board.turns_left_player / 40.0,
            board.turns_left_enemy / 40.0,
            len(board.eggs_player) / 40.0,
            len(board.eggs_enemy) / 40.0,
            board.chicken_player.turds_left / 5.0,
            board.chicken_enemy.turds_left / 5.0,
        ],
        dtype=np.float32,
    )

    board_tensor = torch.from_numpy(state).unsqueeze(0)
    scalar_tensor = torch.from_numpy(scalar).unsqueeze(0)
    return board_tensor, scalar_tensor


class PlayerAgent:
    """Production S-tier agent that combines belief tracking with CNN-guided MCTS."""

    def __init__(self, board: Board, time_left: Callable):
        del time_left  # Not used in initialization but part of required signature.
        self.game_map = board.game_map
        self.size = self.game_map.MAP_SIZE
        self.belief = TrapdoorBelief(self.game_map)
        self.net = ChickenNet(board_size=self.size).to(DEVICE)
        self._load_weights()
        self.net.eval()
        self.mcts = MCTS(self.net)
        self.safety_margin = 8.0
        self.estimate_per_sim = 0.003  # tuned empirically
        self.rng = np.random.default_rng()
        self.bias_params = ActionBiasParams()
        self.last_dir: Direction | None = None
        self.second_last_dir: Direction | None = None
        self.moves_since_last_egg: int = 0

    def _load_weights(self) -> None:
        weights_path = os.path.join(os.path.dirname(__file__), "s_agent_weights.pt")
        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location=DEVICE)
            self.net.load_state_dict(state_dict)
        else:
            raise FileNotFoundError(
                f"Expected trained weights at {weights_path}. "
                "Run train_s_agent.py or copy your s_agent_weights.pt here."
            )

    def _estimate_simulations(self, time_left: Callable[[], float]) -> int:
        try:
            remaining = time_left() - self.safety_margin
        except Exception:
            remaining = 5.0
        remaining = max(0.0, remaining)
        if remaining <= 0.0:
            return 1
        sims = max(1, int(remaining / self.estimate_per_sim))
        return min(sims, 160)

    def _ingest_found_trapdoors(self, board: Board) -> None:
        for loc in getattr(board, "found_trapdoors", ()):
            self.belief.register_known_trapdoor(loc)

    def play(
        self,
        board: Board,
        sensor_data: List[Tuple[bool, bool]],
        time_left: Callable[[], float],
    ) -> Tuple[Direction, MoveType]:
        try:
            self._ingest_found_trapdoors(board)
            self.belief.update(board.chicken_player, sensor_data)
            sims = self._estimate_simulations(time_left)
            visit_dist = self.mcts.run(board, self.belief, sims)
            legal_moves = board.get_valid_moves()
            if not legal_moves:
                return Direction.UP, MoveType.PLAIN

            legal_indices = [encode_action(move) for move in legal_moves]
            temperature = temperature_for_turn(board.turn_count)
            chosen_idx = sample_action_with_bias(
                board,
                legal_indices,
                visit_dist,
                self.last_dir,
                self.second_last_dir,
                self.moves_since_last_egg,
                self.belief,
                self.rng,
                ACTION_SPACE,
                self.bias_params,
                temperature=temperature,
            )
            action = decode_action(chosen_idx)
            if action[1] == MoveType.PLAIN:
                self.second_last_dir = self.last_dir
                self.last_dir = action[0]
                self.moves_since_last_egg += 1
            elif action[1] == MoveType.EGG:
                self.moves_since_last_egg = 0
                self.second_last_dir = self.last_dir
                self.last_dir = None
            else:
                self.second_last_dir = self.last_dir
                self.last_dir = None
                self.moves_since_last_egg += 1
            return action
        except Exception:
            legal_moves = board.get_valid_moves()
            if not legal_moves:
                return Direction.UP, MoveType.PLAIN
            return random.choice(legal_moves)

__all__ = [
    "PlayerAgent",
    "ChickenNet",
    "MCTS",
    "encode_action",
    "decode_action",
    "make_state_tensors",
    "TrapdoorBelief",
    "NUM_ACTIONS",
]

