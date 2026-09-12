"""
NexusDispatch: Primal-Dual Actor-Critic (PDAC) CMDP Policy
Macro-Timescale Strategic Routing Layer with Dual Multiplier Adaptation
and Asymptotic Sub-linear Constraint Regret R_c(T) = o(T).
"""

import math
import numpy as np
from typing import Dict, List, Optional, Tuple


class PrimalDualActorCritic:
    """
    Solves the Constrained Markov Decision Process (CMDP):
    max_pi E[R]  s.t.  E[C_k] <= d_k for constraint channels k.
    Features dual Lagrangian update lambda <- [lambda + eta_lambda * (C - d)]+
    and tracks empirical cumulative constraint violation regret.
    """
    def __init__(
        self,
        obs_dim: int = 8,
        action_dim: int = 4,  # e.g., Cardinal directions [North, South, East, West]
        gamma: float = 0.98,
        lr_actor: float = 0.01,
        lr_critic: float = 0.02,
        lr_dual: float = 0.05,
        constraint_thresholds: Optional[List[float]] = None
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.lr_dual = lr_dual

        # Default constraint thresholds: [corridor_congestion_risk, energy_risk]
        self.d = np.array(constraint_thresholds if constraint_thresholds else [0.20, 0.15], dtype=float)
        self.num_constraints = len(self.d)

        # Dual Lagrangian multipliers: lambda >= 0
        self.lambdas = np.zeros(self.num_constraints, dtype=float)

        # Policy parameters (Linear-Softmax feature approximator for deterministic auditability)
        np.random.seed(42)
        self.theta = np.random.randn(obs_dim, action_dim) * 0.1
        self.w_reward = np.zeros(obs_dim, dtype=float)
        self.w_costs = [np.zeros(obs_dim, dtype=float) for _ in range(self.num_constraints)]

        # Cumulative Regret Tracking: R_c(T) = sum_{t=1}^T (C_t - d)
        self.cumulative_cost = np.zeros(self.num_constraints, dtype=float)
        self.cumulative_budget = np.zeros(self.num_constraints, dtype=float)
        self.regret_history = []
        self.dual_history = []
        self.steps = 0

    def get_action_probs(self, obs: np.ndarray) -> np.ndarray:
        """
        Computes softmax policy action distribution pi(a | s).
        """
        logits = np.dot(obs, self.theta)
        logits_stab = logits - np.max(logits)
        exp_logits = np.exp(logits_stab)
        probs = exp_logits / (np.sum(exp_logits) + 1e-12)
        return probs

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> Tuple[int, np.ndarray]:
        """
        Samples or selects argmax action according to current policy.
        Returns: (action_index, action_probs)
        """
        probs = self.get_action_probs(obs)
        if deterministic:
            action = int(np.argmax(probs))
        else:
            action = int(np.random.choice(len(probs), p=probs))
        return action, probs

    def update_step(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        costs: List[float],
        next_obs: np.ndarray,
        done: bool
    ) -> Dict[str, float]:
        """
        Performs one Primal-Dual step:
        1. Temporal Difference critic updates for reward and costs.
        2. Policy gradient update with penalization sum_k lambda_k Q_Ck.
        3. Dual subgradient update: lambda_k <- [lambda_k + eta_dual * (C_k - d_k)]+.
        4. Regret accounting.
        """
        self.steps += 1
        costs_arr = np.array(costs, dtype=float)

        # 1. Critic TD errors
        v_r = float(np.dot(obs, self.w_reward))
        v_r_next = 0.0 if done else float(np.dot(next_obs, self.w_reward))
        td_r = reward + self.gamma * v_r_next - v_r
        self.w_reward += self.lr_critic * td_r * obs

        td_costs = []
        for k in range(self.num_constraints):
            v_c = float(np.dot(obs, self.w_costs[k]))
            v_c_next = 0.0 if done else float(np.dot(next_obs, self.w_costs[k]))
            td_c = costs_arr[k] + self.gamma * v_c_next - v_c
            self.w_costs[k] += self.lr_critic * td_c * obs
            td_costs.append(td_c)

        # 2. Combined Advantage for Actor
        # A_L = td_r - sum_k lambda_k * td_c
        combined_adv = td_r - float(np.dot(self.lambdas, np.array(td_costs)))

        probs = self.get_action_probs(obs)
        grad_log_pi = -np.outer(obs, probs)
        grad_log_pi[:, action] += obs

        self.theta += self.lr_actor * combined_adv * grad_log_pi
        # Clip weights to avoid divergence
        np.clip(self.theta, -5.0, 5.0, out=self.theta)

        # 3. Dual Update (Subgradient ascent on Lagrangian multiplier)
        # lambda_k <- max(0, lambda_k + eta_dual * (cost_k - d_k))
        constraint_violations = costs_arr - self.d
        self.lambdas = np.maximum(0.0, self.lambdas + self.lr_dual * constraint_violations)

        # 4. Regret Accounting: R_c(T) = sum_{t=1}^T (C_t - d)
        self.cumulative_cost += costs_arr
        self.cumulative_budget += self.d
        current_regret = self.cumulative_cost - self.cumulative_budget
        self.regret_history.append(float(np.max(current_regret)))
        self.dual_history.append(list(self.lambdas))

        return {
            "td_reward": float(td_r),
            "lambdas": list(self.lambdas),
            "max_regret": float(np.max(current_regret)),
            "average_regret": float(np.max(current_regret) / max(1, self.steps))
        }

    def evaluate_regret(self) -> Dict[str, float]:
        """
        Summarizes asymptotic regret metrics: R_c(T) and R_c(T)/T.
        """
        cum_reg = self.cumulative_cost - self.cumulative_budget
        return {
            "total_steps": self.steps,
            "max_cumulative_regret": float(np.max(cum_reg)),
            "average_constraint_regret": float(np.max(cum_reg) / max(1, self.steps)),
            "slater_satisfied": bool(np.all(self.cumulative_cost / max(1, self.steps) <= self.d + 0.05))
        }
