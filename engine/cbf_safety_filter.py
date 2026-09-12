"""
NexusDispatch: Control Barrier Function (CBF) Safety Filter
Implements the Two-Mode Safety Architecture:
Mode 1: Hard CBF-QP (Certified Safe Control Projection)
Mode 2: Emergency Safe-Stop Controller (Deceleration & Safe Retreat)
"""

import math
import numpy as np
from typing import Dict, List, Optional, Tuple


class CBFSafetyFilter:
    """
    Kinematic Micro-Timescale Safety Filter for Multi-Agent Fleets.
    Guarantees conditional forward invariance of the safe set C:
    { x | ||p_i - p_j|| >= d_safe,  p_i in corridor_safe }.
    """
    def __init__(
        self,
        d_safe: float = 1.0,
        v_max: float = 2.0,
        a_max: float = 3.0,
        alpha: float = 2.0,
        r_sense: float = 4.0,
        dt: float = 0.05
    ):
        self.d_safe = d_safe
        self.v_max = v_max
        self.a_max = a_max
        self.alpha = alpha
        self.r_sense = r_sense
        self.dt = dt

        # Audit & Diagnostics
        self.total_evaluations = 0
        self.hard_qp_successes = 0
        self.emergency_stops = 0

    def compute_safe_control(
        self,
        agent_id: str,
        pos_i: np.ndarray,
        vel_i: np.ndarray,
        u_rl: np.ndarray,
        neighbors: List[Tuple[str, np.ndarray, np.ndarray]],  # (id, pos_j, vel_j)
        obstacles: List[Tuple[np.ndarray, np.ndarray, float]] = None # (center, normal, safe_dist)
    ) -> Tuple[np.ndarray, str, Dict]:
        """
        Projects nominal RL action u_rl onto the safe control polytope.
        Returns: (u_safe, mode: 'HARD_CBF' | 'EMERGENCY_STOP', telemetry_dict)
        """
        self.total_evaluations += 1
        pos_i = np.asarray(pos_i, dtype=float)
        vel_i = np.asarray(vel_i, dtype=float)
        u_rl = np.asarray(u_rl, dtype=float)

        # Acceleration bound on step: ||u - vel_i|| <= a_max * dt
        delta_v_max = self.a_max * self.dt
        u_box_min = np.maximum(-self.v_max, vel_i - delta_v_max)
        u_box_max = np.minimum(self.v_max, vel_i + delta_v_max)

        # 1. Construct Linear CBF Constraints: A u <= b
        # For each neighbor j in sensing radius:
        A_rows = []
        b_rows = []

        active_neighbors = []
        for n_id, pos_j, vel_j in neighbors:
            pos_j = np.asarray(pos_j, dtype=float)
            vel_j = np.asarray(vel_j, dtype=float)
            diff = pos_i - pos_j
            dist_sq = float(np.dot(diff, diff))
            dist = math.sqrt(max(1e-6, dist_sq))

            if dist <= self.r_sense and dist > 1e-4:
                active_neighbors.append((n_id, pos_j, vel_j, dist))
                # Barrier: h_ij = ||p_i - p_j||^2 - d_safe^2 >= 0
                h_ij = dist_sq - (self.d_safe ** 2)

                # Time derivative: h_dot = 2 (p_i - p_j)^T (u_i - v_j)
                # Reciprocal CBF condition: 2 diff^T u_i - 2 diff^T v_j + alpha * h_ij >= 0
                # In standard form a^T u_i <= b:
                # -2 diff^T u_i <= -2 diff^T v_j + alpha * h_ij
                a_ij = -2.0 * diff
                b_ij = -2.0 * float(np.dot(diff, vel_j)) + self.alpha * h_ij
                A_rows.append(a_ij)
                b_rows.append(b_ij)

        # Static obstacle barriers if any
        if obstacles:
            for obs_center, obs_normal, obs_margin in obstacles:
                diff_obs = pos_i - obs_center
                dist_obs = float(np.dot(diff_obs, obs_normal))
                h_obs = dist_obs - obs_margin
                # Normal points into safe half-space: normal^T u_i + alpha * h_obs >= 0
                # -normal^T u_i <= alpha * h_obs
                A_rows.append(-obs_normal)
                b_rows.append(self.alpha * h_obs)

        # 2. Mode 1: Attempt Hard CBF-QP Solve
        is_feasible, u_hard = self._solve_2d_cbf_qp(
            u_rl=u_rl,
            A=np.array(A_rows) if A_rows else np.zeros((0, 2)),
            b=np.array(b_rows) if b_rows else np.zeros(0),
            u_min=u_box_min,
            u_max=u_box_max
        )

        if is_feasible:
            self.hard_qp_successes += 1
            return u_hard, "HARD_CBF", {
                "active_constraints": len(A_rows),
                "slack": 0.0,
                "deviation_from_rl": float(np.linalg.norm(u_hard - u_rl))
            }

        # 3. Mode 2: Emergency Safe-Stop Controller
        # If the hard polytope is empty (extreme bottleneck congestion),
        # decelerate along current velocity while adding repulsive gradient
        self.emergency_stops += 1
        speed = float(np.linalg.norm(vel_i))
        if speed > 1e-4:
            # Maximum deceleration towards 0
            decay = max(0.0, 1.0 - (delta_v_max / speed))
            u_brake = vel_i * decay
        else:
            u_brake = np.zeros(2)

        # Repulsive push away from critically close neighbors
        for _, pos_j, _, dist in active_neighbors:
            if dist < self.d_safe:
                repulse_dir = (pos_i - pos_j) / dist
                # Push gently away along barrier normal
                u_brake += 0.3 * repulse_dir

        u_brake = np.clip(u_brake, u_box_min, u_box_max)

        return u_brake, "EMERGENCY_STOP", {
            "active_constraints": len(A_rows),
            "emergency_triggered": True,
            "speed": speed
        }

    def _solve_2d_cbf_qp(
        self,
        u_rl: np.ndarray,
        A: np.ndarray,
        b: np.ndarray,
        u_min: np.ndarray,
        u_max: np.ndarray
    ) -> Tuple[bool, np.ndarray]:
        """
        Fast, deterministic 2D Quadratic Program solver using active-set projection.
        Solves: min 0.5 ||u - u_rl||^2 s.t. A u <= b, u_min <= u <= u_max.
        """
        # First check unconstrained projection on box
        u_cand = np.clip(u_rl, u_min, u_max)
        if len(b) == 0 or np.all(A @ u_cand <= b + 1e-6):
            return True, u_cand

        # Discretized boundary search & 1D line intersections for robust 2D QP
        # Test candidate vertices from intersections of active constraints and box limits
        candidates = []

        # 1. Unconstrained candidate
        if np.all(A @ u_cand <= b + 1e-6):
            candidates.append(u_cand)

        # 2. Project u_rl onto each constraint line a_k^T u = b_k
        m = len(b)
        for k in range(m):
            a_k = A[k]
            b_k = b[k]
            a_norm_sq = float(np.dot(a_k, a_k))
            if a_norm_sq > 1e-8:
                # Projection of u_rl onto line: u_proj = u_rl - ((a^T u_rl - b) / ||a||^2) a
                dist = (float(np.dot(a_k, u_rl)) - b_k) / a_norm_sq
                u_proj = u_rl - dist * a_k
                u_proj_clipped = np.clip(u_proj, u_min, u_max)
                if np.all(A @ u_proj_clipped <= b + 1e-5):
                    candidates.append(u_proj_clipped)

        # 3. Intersections between pairs of constraints
        for i in range(min(m, 8)):
            for j in range(i + 1, min(m, 8)):
                mat = np.array([A[i], A[j]])
                det = mat[0, 0] * mat[1, 1] - mat[0, 1] * mat[1, 0]
                if abs(det) > 1e-6:
                    u_inter = np.linalg.solve(mat, np.array([b[i], b[j]]))
                    u_inter_clipped = np.clip(u_inter, u_min, u_max)
                    if np.all(A @ u_inter_clipped <= b + 1e-5):
                        candidates.append(u_inter_clipped)

        # 4. Box corners
        corners = [
            np.array([u_min[0], u_min[1]]),
            np.array([u_min[0], u_max[1]]),
            np.array([u_max[0], u_min[1]]),
            np.array([u_max[0], u_max[1]]),
            np.array([0.0, 0.0]) # Zero velocity candidate
        ]
        for c in corners:
            c_clipped = np.clip(c, u_min, u_max)
            if np.all(A @ c_clipped <= b + 1e-5):
                candidates.append(c_clipped)

        if not candidates:
            # Polytope is empty or contradictory: infeasible
            return False, np.clip(u_rl, u_min, u_max)

        # Select feasible candidate minimizing distance to u_rl
        best_cand = None
        best_cost = float('inf')
        for cand in candidates:
            cost = float(np.sum((cand - u_rl) ** 2))
            if cost < best_cost:
                best_cost = cost
                best_cand = cand

        return True, best_cand
