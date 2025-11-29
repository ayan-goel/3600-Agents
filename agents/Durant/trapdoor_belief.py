from __future__ import annotations

from typing import List, Optional, Tuple

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
        self._known_even: Optional[Tuple[int, int]] = None
        self._known_odd: Optional[Tuple[int, int]] = None

    def reset(self) -> None:
        """Clear cached beliefs so the next update re-initializes them."""
        self._initialized = False
        self.belief_even.fill(0.0)
        self.belief_odd.fill(0.0)
        self._known_even = None
        self._known_odd = None

    def clone(self) -> "TrapdoorBelief":
        """Return a deep copy of this belief state."""
        new = TrapdoorBelief(self.game_map, self._min_likelihood)
        new._initialized = self._initialized
        new.belief_even = self.belief_even.copy()
        new.belief_odd = self.belief_odd.copy()
        new._known_even = self._known_even
        new._known_odd = self._known_odd
        return new

    def as_tensor(self) -> np.ndarray:
        """Return stacked (2, H, W) numpy array for compatibility."""
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

        if self._known_even is not None:
            self.belief_even.fill(0.0)
            ex, ey = self._known_even
            self.belief_even[ey, ex] = 1.0
        else:
            even_count = int(even_mask.sum())
            if even_count == 0:
                valid_mask = even_mask | odd_mask
                count = max(int(valid_mask.sum()), 1)
                self.belief_even[valid_mask] = 1.0 / count
            else:
                self.belief_even[even_mask] = 1.0 / even_count

        if self._known_odd is not None:
            self.belief_odd.fill(0.0)
            ox, oy = self._known_odd
            self.belief_odd[oy, ox] = 1.0
        else:
            odd_count = int(odd_mask.sum())
            if odd_count == 0:
                valid_mask = even_mask | odd_mask
                count = max(int(valid_mask.sum()), 1)
                self.belief_odd[valid_mask] = 1.0 / count
            else:
                self.belief_odd[odd_mask] = 1.0 / odd_count

        self._initialized = True

    def _is_valid_cell(self, x: int, y: int) -> bool:
        return 0 <= x < self.size and 0 <= y < self.size

    def register_known_trapdoor(self, loc: Tuple[int, int]) -> None:
        """Collapse parity belief when a trapdoor is definitively located."""
        x, y = loc
        if not self._is_valid_cell(x, y):
            return
        parity = (x + y) % 2
        target = self.belief_even if parity == 0 else self.belief_odd
        if target[y, x] == 1.0 and target.sum() == 1.0:
            return
        target.fill(0.0)
        target[y, x] = 1.0
        if parity == 0:
            self._known_even = (x, y)
        else:
            self._known_odd = (x, y)
        self._initialized = True

    def trapdoor_prob_at(self, loc: Tuple[int, int]) -> float:
        """Return belief probability that a trapdoor is at loc (using matching parity grid)."""
        if not self._initialized:
            self._init_uniform()
        x, y = loc
        if not self._is_valid_cell(x, y):
            return 0.0
        if (x + y) % 2 == 0:
            return float(self.belief_even[y, x])
        return float(self.belief_odd[y, x])