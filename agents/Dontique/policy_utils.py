from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from game.board import Board
from game.enums import Direction, MoveType, loc_after_direction
from .trapdoor_belief import TrapdoorBelief


OPPOSITE_DIRECTION = {
    Direction.UP: Direction.DOWN,
    Direction.DOWN: Direction.UP,
    Direction.LEFT: Direction.RIGHT,
    Direction.RIGHT: Direction.LEFT,
}


@dataclass(frozen=True)
class ActionBiasParams:
    egg_bonus: float = 0.8
    backtrack_penalty: float = 0.25
    deficit_push_bonus: float = 0.45
    same_dir_bonus: float = 1.1
    aba_penalty: float = 0.4
    farm_trapdoor_threshold: float = 0.05
    farm_hold_threshold: int = 2
    farm_move_bonus: float = 0.35
    farm_move_penalty: float = 0.85
    farm_danger_distance: int = 3
    force_egg_below: int = 4
    force_egg_prob_cap: float = 0.35
    force_egg_deficit: int = 1
    deficit_egg_multiplier: float = 4.0
    deficit_egg_force_start: int = 1
    egg_cooldown_bias_start: int = 4
    egg_cooldown_force: int = 7
    egg_cooldown_bonus: float = 1.6
    deficit_temp_zero: int = 3
    seek_enemy_deficit: int = 2
    enemy_egg_seek_bonus: float = 0.4
    enemy_egg_avoid_penalty: float = 0.88
    turd_combat_distance: int = 2
    defend_cluster_radius: int = 1
    defend_cluster_size: int = 2
    defend_lead_for_turd: int = 1
    turd_attack_bonus: float = 1.4
    turd_defense_bonus: float = 1.15
    trapdoor_risk_scale: float = 2.0
    trapdoor_zero_threshold: float = 0.20
    trapdoor_zero_penalty: float = 1e-4
    trapdoor_floor: float = 0.05
    min_prob: float = 1e-6
    endgame_push_turns: int = 10
    endgame_egg_multiplier: float = 2.0


def temperature_for_turn(turn_idx: int) -> float:
    if turn_idx < 20:
        return 1.0
    if turn_idx < 50:
        return 0.6
    return 0.25


def jitter_simulations(base: int, rng: np.random.Generator) -> int:
    noise = int(rng.normal(0.0, max(4.0, base * 0.15)))
    sims = base + noise
    return max(32, sims)


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _count_nearby_eggs(loc: tuple[int, int], eggs: Sequence[tuple[int, int]], radius: int) -> int:
    return sum(1 for egg in eggs if _manhattan(loc, egg) <= radius)


def _min_distance_to(loc: tuple[int, int], targets: Sequence[tuple[int, int]]) -> float:
    best = float("inf")
    for tgt in targets:
        dist = _manhattan(loc, tgt)
        if dist < best:
            best = dist
    return best


def sample_action_with_bias(
    board: Board,
    legal_indices: Sequence[int],
    visit_dist: np.ndarray,
    last_dir: Direction | None,
    second_last_dir: Direction | None,
    since_last_egg: int,
    belief: TrapdoorBelief,
    rng: np.random.Generator,
    action_space: Sequence[tuple[Direction, MoveType]],
    params: ActionBiasParams,
    temperature: float = 1.0,
) -> int:
    if not legal_indices:
        raise ValueError("No legal moves supplied for sampling")

    probs = np.array([visit_dist[idx] for idx in legal_indices], dtype=np.float64)

    my_eggs = board.chicken_player.get_eggs_laid()
    opp_eggs = board.chicken_enemy.get_eggs_laid()
    egg_deficit = opp_eggs - my_eggs
    egg_lead = my_eggs - opp_eggs
    cur_loc = board.chicken_player.get_location()
    cur_trap_prob = belief.trapdoor_prob_at(cur_loc)
    enemy_loc = board.chicken_enemy.get_location()
    enemy_dist_current = _manhattan(cur_loc, enemy_loc)
    enemy_eggs = tuple(board.eggs_enemy)
    current_enemy_egg_dist = (
        _min_distance_to(cur_loc, enemy_eggs) if enemy_eggs else float("inf")
    )
    effective_temp = temperature
    if egg_deficit >= params.deficit_temp_zero:
        effective_temp = 0.0

    egg_moves = [
        idx for idx in legal_indices if action_space[idx][1] == MoveType.EGG
    ]
    if (
        egg_moves
        and board.can_lay_egg()
        and cur_trap_prob < 0.95  # avoid confirmed trapdoors
        and (
            (my_eggs < params.force_egg_below and cur_trap_prob <= params.force_egg_prob_cap)
            or (
                egg_deficit >= params.force_egg_deficit
                and cur_trap_prob <= params.force_egg_prob_cap * 1.5
            )
        )
    ):
        best_egg = max(egg_moves, key=lambda idx: visit_dist[idx])
        return best_egg
    # Periodic egg forcing if it's been too long and safe
    if (
        egg_moves
        and board.can_lay_egg()
        and since_last_egg >= params.egg_cooldown_force
        and cur_trap_prob <= params.force_egg_prob_cap
    ):
        best_egg = max(egg_moves, key=lambda idx: visit_dist[idx])
        return best_egg

    if effective_temp <= 0:
        best = int(np.argmax(probs))
        return legal_indices[best]

    probs = np.maximum(probs, params.min_prob) ** (1.0 / effective_temp)

    for i, idx in enumerate(legal_indices):
        direction, move_type = action_space[idx]

        if move_type == MoveType.EGG and board.can_lay_egg():
            probs[i] *= 1.0 + params.egg_bonus
            probs[i] *= max(
                params.trapdoor_floor,
                1.0 - params.trapdoor_risk_scale * cur_trap_prob,
            )
            if egg_deficit >= params.deficit_egg_force_start:
                probs[i] *= params.deficit_egg_multiplier
            if since_last_egg >= params.egg_cooldown_bias_start:
                probs[i] *= params.egg_cooldown_bonus
            # Endgame preference for banking eggs
            if getattr(board, "turns_left_player", 40) <= params.endgame_push_turns:
                probs[i] *= params.endgame_egg_multiplier
            continue

        if move_type == MoveType.TURD:
            if not board.can_lay_turd():
                probs[i] = params.min_prob
                continue
            enemy_dist = _manhattan(cur_loc, enemy_loc)
            cluster = _count_nearby_eggs(
                cur_loc, board.eggs_player, params.defend_cluster_radius
            )
            defend_cluster = (
                cluster >= params.defend_cluster_size
                and egg_lead >= params.defend_lead_for_turd
            )
            allowed = (
                enemy_dist <= params.turd_combat_distance
                or defend_cluster
            )
            if not allowed:
                probs[i] = params.min_prob
                continue
            boost = (
                params.turd_attack_bonus
                if enemy_dist <= params.turd_combat_distance
                else params.turd_defense_bonus
            )
            probs[i] *= boost
            continue

        if egg_deficit >= 2 and move_type == MoveType.PLAIN:
            probs[i] *= 1.0 + params.deficit_push_bonus

        if (
            move_type == MoveType.PLAIN
            and last_dir is not None
            and direction == OPPOSITE_DIRECTION[last_dir]
        ):
            probs[i] *= params.backtrack_penalty

        # Anti A-B-A oscillation
        if (
            move_type == MoveType.PLAIN
            and second_last_dir is not None
            and last_dir is not None
            and last_dir == OPPOSITE_DIRECTION[second_last_dir]
            and direction == second_last_dir
        ):
            probs[i] *= params.aba_penalty

        # Directional persistence
        if move_type == MoveType.PLAIN and last_dir is not None and direction == last_dir:
            probs[i] *= params.same_dir_bonus

        next_loc = None
        if move_type == MoveType.PLAIN:
            next_loc = loc_after_direction(cur_loc, direction)

        # Only bias "hold away from enemy" if enemy is actually close
        if (
            move_type == MoveType.PLAIN
            and my_eggs >= params.farm_hold_threshold
            and enemy_dist_current <= params.farm_danger_distance
        ):
            next_dist = _manhattan(next_loc, enemy_loc)
            if next_dist > enemy_dist_current:
                probs[i] *= 1.0 + params.farm_move_bonus
            elif next_dist < enemy_dist_current:
                probs[i] *= params.farm_move_penalty

        if (
            move_type == MoveType.PLAIN
            and enemy_eggs
            and egg_deficit >= params.seek_enemy_deficit
        ):
            next_enemy_egg_dist = _min_distance_to(next_loc, enemy_eggs)
            if next_enemy_egg_dist < current_enemy_egg_dist:
                probs[i] *= 1.0 + params.enemy_egg_seek_bonus
            elif next_enemy_egg_dist > current_enemy_egg_dist:
                probs[i] *= params.enemy_egg_avoid_penalty

        if move_type == MoveType.PLAIN:
            next_prob = belief.trapdoor_prob_at(next_loc)
            if next_prob >= params.trapdoor_zero_threshold:
                probs[i] = max(params.min_prob, params.trapdoor_zero_penalty)
                continue
            risk_penalty = max(
                params.trapdoor_floor,
                1.0 - params.trapdoor_risk_scale * next_prob,
            )
            probs[i] *= risk_penalty

    probs_sum = probs.sum()
    if probs_sum <= params.min_prob * len(probs):
        probs[:] = 1.0 / len(probs)
    else:
        probs /= probs_sum

    choice = int(rng.choice(len(legal_indices), p=probs))
    return legal_indices[choice]

