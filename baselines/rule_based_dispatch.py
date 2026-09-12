"""
NexusDispatch Baseline: Rule-Based Greedy Dispatch
Uses shortest-path routing with a conservative static safety stop zone.
Prone to the infamous 'Freezing Robot' deadlock problem when two opposing vehicles meet.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


class RuleBasedDispatch:
    """
    Greedy shortest-path dispatch with conservative static halt bubble.
    Demonstrates deadlock / freeze behavior at corridor chokepoints.
    """
    def __init__(self, stop_distance: float = 2.0, v_max: float = 2.0):
        self.stop_distance = stop_distance
        self.v_max = v_max
        self.deadlocks_detected = 0

    def compute_action(
        self,
        agent_id: str,
        pos: np.ndarray,
        target_pos: np.ndarray,
        neighbors: List[Tuple[str, np.ndarray, np.ndarray]]
    ) -> Tuple[np.ndarray, bool]:
        """
        If any opposing agent is within stop_distance, completely halts velocity (Freezing Robot).
        Returns: (velocity_action, is_frozen)
        """
        pos = np.asarray(pos, dtype=float)
        target_pos = np.asarray(target_pos, dtype=float)

        # Check halt condition
        is_frozen = False
        for _, n_pos, n_vel in neighbors:
            dist = float(np.linalg.norm(pos - np.asarray(n_pos, dtype=float)))
            if dist < self.stop_distance:
                # Opposing movement or stationary obstacle
                is_frozen = True
                self.deadlocks_detected += 1
                return np.zeros(2), True

        diff = target_pos - pos
        dist = float(np.linalg.norm(diff))
        if dist > 1e-3:
            vel = (diff / dist) * min(self.v_max, dist)
        else:
            vel = np.zeros(2)

        return vel, False
