"""
NexusDispatch Baseline: Unconstrained MARL
Implements standard multi-agent reinforcement learning with soft penalty reward shaping:
R = R_task - w_coll * I(dist < d_safe) - w_batt * I(battery < E_min)
Without CBF safety projection or dual multiplier convergence guarantees.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


class UnconstrainedMARL:
    """
    Standard soft-penalty MARL policy.
    Susceptible to severe safety violations and battery stranding under heavy traffic.
    """
    def __init__(
        self,
        num_agents: int = 8,
        coll_penalty: float = 50.0,
        batt_penalty: float = 100.0,
        v_max: float = 2.0
    ):
        self.num_agents = num_agents
        self.coll_penalty = coll_penalty
        self.batt_penalty = batt_penalty
        self.v_max = v_max

    def compute_action(
        self,
        agent_id: str,
        pos: np.ndarray,
        target_pos: np.ndarray,
        neighbors: List[Tuple[str, np.ndarray]],
        battery: float
    ) -> np.ndarray:
        """
        Computes velocity action towards target with naive repulsive potential force.
        Lacks formal forward invariance guarantees.
        """
        pos = np.asarray(pos, dtype=float)
        target_pos = np.asarray(target_pos, dtype=float)

        diff = target_pos - pos
        dist = float(np.linalg.norm(diff))
        if dist > 1e-3:
            dir_goal = diff / dist
            u_nom = dir_goal * min(self.v_max, dist)
        else:
            u_nom = np.zeros(2)

        # Naive soft repulsion (often fails or causes oscillation)
        repulsion = np.zeros(2)
        for _, n_pos in neighbors:
            n_diff = pos - np.asarray(n_pos, dtype=float)
            n_dist = float(np.linalg.norm(n_diff))
            if 0.001 < n_dist < 2.0:
                repulsion += (n_diff / n_dist) * (2.0 - n_dist) * 0.5

        action = u_nom + repulsion
        speed = float(np.linalg.norm(action))
        if speed > self.v_max:
            action = (action / speed) * self.v_max

        return action
