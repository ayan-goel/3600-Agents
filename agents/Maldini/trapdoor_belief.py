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
        # Cache center-weighted prior (trapdoors are more likely near center)
        self._center_weights = self._build_center_weights()
        # Track tiles we've safely visited (no trapdoor there)
        self._safe_tiles: set = set()
        # Track strong signal history for better inference
        self._signal_history: List[Tuple[Tuple[int, int], Tuple[bool, bool], Tuple[bool, bool]]] = []
    
    def _build_center_weights(self) -> np.ndarray:
        """Build center-weighted prior matching trapdoor placement logic."""
        dim = self.size
        weights = np.zeros((dim, dim), dtype=np.float64)
        # Edges have weight 0 (rings 0 and 1)
        # Ring 2 (2 from edge) has weight 1
        # Ring 3+ (center) has weight 2
        weights[2:dim-2, 2:dim-2] = 1.0
        weights[3:dim-3, 3:dim-3] = 2.0
        return weights

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
        """Bayes update the belief grids given the latest sensory data.
        
        Also marks current location as safe and records signal history.
        """
        if len(sensory_data) < 2:
            return
        
        # Mark current location as safe (we're here and didn't trigger trapdoor)
        current_loc = chicken.get_location()
        self.mark_safe(current_loc)
        
        # Record signal history for potential multi-observation inference
        self._signal_history.append((current_loc, sensory_data[0], sensory_data[1]))
        if len(self._signal_history) > 50:
            self._signal_history.pop(0)

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
                    
                    # Safe tiles have zero probability
                    if (x, y) in self._safe_tiles:
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
        """Initialize a CENTER-WEIGHTED prior over valid cells of matching parity.
        
        Trapdoors are placed with higher probability near the center:
        - Edge tiles (rings 0-1): weight 0 (impossible)
        - Ring 2: weight 1
        - Center (ring 3+): weight 2
        """
        even_weights = np.zeros_like(self.belief_even, dtype=np.float64)
        odd_weights = np.zeros_like(self.belief_odd, dtype=np.float64)

        for y in range(self.size):
            for x in range(self.size):
                if not self._is_valid_cell(x, y):
                    continue
                # Skip tiles we've safely visited
                if (x, y) in self._safe_tiles:
                    continue
                weight = self._center_weights[y, x]
                if weight <= 0:
                    continue
                if (x + y) % 2 == 0:
                    even_weights[y, x] = weight
                else:
                    odd_weights[y, x] = weight

        if self._known_even is not None:
            self.belief_even.fill(0.0)
            ex, ey = self._known_even
            self.belief_even[ey, ex] = 1.0
        else:
            total = even_weights.sum()
            if total > 0:
                self.belief_even = even_weights / total
            else:
                # Fallback to uniform if all weights are zero
                even_mask = np.zeros_like(self.belief_even, dtype=bool)
                for y in range(self.size):
                    for x in range(self.size):
                        if (x + y) % 2 == 0 and self._is_valid_cell(x, y):
                            if (x, y) not in self._safe_tiles:
                                even_mask[y, x] = True
                count = max(int(even_mask.sum()), 1)
                self.belief_even[even_mask] = 1.0 / count

        if self._known_odd is not None:
            self.belief_odd.fill(0.0)
            ox, oy = self._known_odd
            self.belief_odd[oy, ox] = 1.0
        else:
            total = odd_weights.sum()
            if total > 0:
                self.belief_odd = odd_weights / total
            else:
                odd_mask = np.zeros_like(self.belief_odd, dtype=bool)
                for y in range(self.size):
                    for x in range(self.size):
                        if (x + y) % 2 == 1 and self._is_valid_cell(x, y):
                            if (x, y) not in self._safe_tiles:
                                odd_mask[y, x] = True
                count = max(int(odd_mask.sum()), 1)
                self.belief_odd[odd_mask] = 1.0 / count

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
        # If we've visited this tile safely, probability is 0
        if loc in self._safe_tiles:
            return 0.0
        if (x + y) % 2 == 0:
            return float(self.belief_even[y, x])
        return float(self.belief_odd[y, x])
    
    def mark_safe(self, loc: Tuple[int, int]) -> None:
        """Mark a tile as safe (we visited it without triggering a trapdoor)."""
        x, y = loc
        if not self._is_valid_cell(x, y):
            return
        self._safe_tiles.add(loc)
        # Zero out belief at this location
        if (x + y) % 2 == 0:
            if self.belief_even[y, x] > 0:
                self.belief_even[y, x] = 0.0
                total = self.belief_even.sum()
                if total > 0:
                    self.belief_even /= total
        else:
            if self.belief_odd[y, x] > 0:
                self.belief_odd[y, x] = 0.0
                total = self.belief_odd.sum()
                if total > 0:
                    self.belief_odd /= total
    
    def get_danger_zone(self, threshold: float = 0.15) -> set:
        """Return set of tiles with trapdoor probability above threshold."""
        if not self._initialized:
            self._init_uniform()
        danger = set()
        for y in range(self.size):
            for x in range(self.size):
                if (x, y) in self._safe_tiles:
                    continue
                prob = self.trapdoor_prob_at((x, y))
                if prob >= threshold:
                    danger.add((x, y))
        return danger
    
    def get_high_risk_tiles(self, top_n: int = 4) -> List[Tuple[Tuple[int, int], float]]:
        """Return the top N highest-probability trapdoor locations."""
        if not self._initialized:
            self._init_uniform()
        candidates = []
        for y in range(self.size):
            for x in range(self.size):
                if (x, y) in self._safe_tiles:
                    continue
                prob = self.trapdoor_prob_at((x, y))
                if prob > 0:
                    candidates.append(((x, y), prob))
        candidates.sort(key=lambda t: t[1], reverse=True)
        return candidates[:top_n]

