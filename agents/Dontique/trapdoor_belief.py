from __future__ import annotations

from typing import List, Tuple

import numpy as np

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENGINE_ROOT = PROJECT_ROOT / "engine"
if str(ENGINE_ROOT) not in sys.path:
    sys.path.append(str(ENGINE_ROOT))

from game.chicken import Chicken
from game.game_map import GameMap


class TrapdoorBelief:
    """Bayesian belief tracker for even and odd trapdoor locations."""

    def __init__(self, game_map: GameMap, min_likelihood: float = 1e-6) -> None:
        self.game_map = game_map
        self.size = game_map.MAP_SIZE
        self.belief_even = np.zeros((self.size, self.size), dtype=np.float64)
        self.belief_odd = np.zeros((self.size, self.size), dtype=np.float64)
        self._initialized = False
        self._min_likelihood = min_likelihood

    def reset(self) -> None:
        """Clear cached beliefs so the next update re-initializes them."""
        self._initialized = False
        self.belief_even.fill(0.0)
        self.belief_odd.fill(0.0)

    def clone(self) -> "TrapdoorBelief":
        """Return a deep copy of this belief state."""
        new = TrapdoorBelief(self.game_map, self._min_likelihood)
        new._initialized = self._initialized
        new.belief_even = self.belief_even.copy()
        new.belief_odd = self.belief_odd.copy()
        return new

    def as_tensor(self) -> np.ndarray:
        """Return stacked (2, H, W) numpy array for NN consumption."""
        if not self._initialized:
            self._init_uniform()
        return np.stack([self.belief_even, self.belief_odd], axis=0)

    def update(
        self,
        chicken: Chicken,
        sensory_data: List[Tuple[bool, bool]],
    ) -> None:
        """Bayes update the belief grids given the latest sensory data."""
        if len(sensory_data) < 2:
            return

        if not self._initialized:
            self._init_uniform()

        observations = {
            "even": sensory_data[0],
            "odd": sensory_data[1],
        }

        for parity_name, belief in [
            ("even", self.belief_even),
            ("odd", self.belief_odd),
        ]:
            did_hear, did_feel = observations[parity_name]
            target_parity = 0 if parity_name == "even" else 1

            for y in range(self.size):
                for x in range(self.size):
                    if not self._is_valid_cell(x, y):
                        belief[y, x] = 0.0
                        continue

                    if (x + y) % 2 != target_parity:
                        belief[y, x] = 0.0
                        continue

                    prior = belief[y, x]
                    if prior <= 0.0:
                        continue

                    prob_hear, prob_feel = chicken.prob_senses_if_trapdoor_were_at(
                        did_hear,
                        did_feel,
                        x,
                        y,
                    )
                    likelihood = max(prob_hear * prob_feel, self._min_likelihood)
                    belief[y, x] = prior * likelihood

            total = float(belief.sum())
            if total > 0.0:
                belief /= total
            else:
                # Numerical underflow -> reset priors to stay sane.
                self._init_uniform()
                break

    def _init_uniform(self) -> None:
        """Initialize a uniform prior over valid cells of matching parity."""
        even_mask = np.zeros_like(self.belief_even, dtype=bool)
        odd_mask = np.zeros_like(self.belief_odd, dtype=bool)

        for y in range(self.size):
            for x in range(self.size):
                if not self._is_valid_cell(x, y):
                    continue
                if (x + y) % 2 == 0:
                    even_mask[y, x] = True
                else:
                    odd_mask[y, x] = True

        even_count = int(even_mask.sum())
        odd_count = int(odd_mask.sum())

        if even_count == 0 or odd_count == 0:
            valid_mask = even_mask | odd_mask
            count = max(int(valid_mask.sum()), 1)
            self.belief_even[valid_mask] = 1.0 / count
            self.belief_odd[valid_mask] = 1.0 / count
        else:
            self.belief_even[even_mask] = 1.0 / even_count
            self.belief_odd[odd_mask] = 1.0 / odd_count

        self._initialized = True

    def _is_valid_cell(self, x: int, y: int) -> bool:
        return 0 <= x < self.size and 0 <= y < self.size


