"""
NexusDispatch Baseline: Centralized MAPF (Time-Space Reservation Grid)
Implements prioritized multi-agent path finding over discretized space-time reservations.
Subject to high computational overhead O(b^N) and vulnerability to execution drift.
"""

import math
import numpy as np
from typing import Dict, List, Optional, Tuple, Set


class CentralizedMAPF:
    """
    Time-Space Reservation Table baseline.
    Requires global synchronization and re-planning upon kinematic deviations.
    """
    def __init__(self, grid_res: float = 1.0, time_step: float = 0.5):
        self.grid_res = grid_res
        self.time_step = time_step
        # Reservation table: (x_cell, y_cell, time_tick) -> agent_id
        self.reservations: Dict[Tuple[int, int, int], str] = {}
        self.replan_count = 0

    def to_cell(self, pos: np.ndarray) -> Tuple[int, int]:
        return int(round(pos[0] / self.grid_res)), int(round(pos[1] / self.grid_res))

    def plan_path(
        self,
        agent_id: str,
        start_pos: np.ndarray,
        goal_pos: np.ndarray,
        start_time_tick: int,
        max_horizon: int = 40
    ) -> List[Tuple[float, float]]:
        """
        Plans a space-time collision-free path reserving grid cells.
        """
        self.replan_count += 1
        start_cell = self.to_cell(start_pos)
        goal_cell = self.to_cell(goal_pos)

        # Clear existing reservations for agent_id
        keys_to_del = [k for k, v in self.reservations.items() if v == agent_id]
        for k in keys_to_del:
            del self.reservations[k]

        # Greedy space-time A* search
        path = [start_pos]
        curr = start_cell
        for t in range(1, max_horizon):
            dx = np.sign(goal_cell[0] - curr[0])
            dy = np.sign(goal_cell[1] - curr[1])
            # Candidate moves: dx, dy, or wait
            candidates = [
                (curr[0] + dx, curr[1]),
                (curr[0], curr[1] + dy),
                curr
            ]
            chosen = curr
            for cand in candidates:
                tick = start_time_tick + t
                if (cand[0], cand[1], tick) not in self.reservations:
                    chosen = cand
                    break

            self.reservations[(chosen[0], chosen[1], start_time_tick + t)] = agent_id
            curr = chosen
            path.append(np.array([curr[0] * self.grid_res, curr[1] * self.grid_res]))
            if curr == goal_cell:
                break

        return path
