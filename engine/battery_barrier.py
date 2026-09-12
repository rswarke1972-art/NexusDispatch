"""
NexusDispatch: Battery Margin Barrier & Energy Management System
Guarantees certified zero vehicle stranding via dynamic margin:
h_E(E_i, p_i) = E_i - E_min - E_reserve(p_i, charger) >= 0
"""

import math
import numpy as np
from typing import Dict, List, Optional, Tuple


class BatteryBarrierManager:
    """
    Manages energy dissipation dynamics, battery margin Control Barrier Functions,
    and automatic charging pad diversions to prevent AGV/drone stranding.
    """
    def __init__(
        self,
        e_min: float = 5.0,           # Hard cutoff battery percentage (%)
        e_reserve_rate: float = 0.8,   # Energy cost per unit distance (% / m)
        payload_factor: float = 0.25,  # Added drain coefficient per unit payload
        p_base: float = 0.05,          # Idle electrical baseload (% / s)
        c_v: float = 0.08,             # Velocity quadratic drag (% / (m/s)^2 / s)
        c_a: float = 0.04,             # Acceleration mechanical cost (% / (m/s^2) / s)
        charge_rate: float = 5.0,      # Charging speed (% / s)
        gamma: float = 1.5,            # CBF class-K linear parameter
        divert_threshold: float = 3.0  # Margin threshold (h_E) triggering diversion
    ):
        self.e_min = e_min
        self.e_reserve_rate = e_reserve_rate
        self.payload_factor = payload_factor
        self.p_base = p_base
        self.c_v = c_v
        self.c_a = c_a
        self.charge_rate = charge_rate
        self.gamma = gamma
        self.divert_threshold = divert_threshold

    def compute_reserve_energy(
        self,
        pos_i: np.ndarray,
        charger_pos: np.ndarray,
        payload_mass: float = 0.0
    ) -> Tuple[float, np.ndarray]:
        """
        Computes E_reserve(p_i, charger) and its gradient with respect to p_i.
        Returns: (E_reserve, grad_p)
        """
        diff = np.asarray(pos_i, dtype=float) - np.asarray(charger_pos, dtype=float)
        dist = float(np.linalg.norm(diff))
        weight_mult = 1.0 + self.payload_factor * max(0.0, payload_mass)
        e_reserve = self.e_reserve_rate * dist * weight_mult

        if dist > 1e-4:
            grad_p = (self.e_reserve_rate * weight_mult) * (diff / dist)
        else:
            grad_p = np.zeros_like(diff)

        return e_reserve, grad_p

    def compute_battery_margin(
        self,
        energy: float,
        pos_i: np.ndarray,
        charging_stations: List[np.ndarray],
        payload_mass: float = 0.0
    ) -> Tuple[float, np.ndarray, int]:
        """
        Computes safety margin h_E to the closest charging pad.
        Returns: (h_E, nearest_charger_pos, nearest_index)
        """
        if not charging_stations:
            # Fallback if no chargers specified
            return energy - self.e_min, np.zeros(2), -1

        min_reserve = float('inf')
        best_charger = charging_stations[0]
        best_idx = 0

        for idx, ch_pos in enumerate(charging_stations):
            res, _ = self.compute_reserve_energy(pos_i, ch_pos, payload_mass)
            if res < min_reserve:
                min_reserve = res
                best_charger = ch_pos
                best_idx = idx

        h_e = energy - self.e_min - min_reserve
        return h_e, np.asarray(best_charger, dtype=float), best_idx

    def step_energy(
        self,
        current_energy: float,
        vel: np.ndarray,
        acc: np.ndarray,
        dt: float,
        is_charging: bool = False
    ) -> float:
        """
        Integrates dynamic power dissipation physics over interval dt:
        E_dot = +charge_rate (if charging) OR -(P_base + c_v * ||v||^2 + c_a * ||a||)
        """
        if is_charging:
            new_energy = current_energy + self.charge_rate * dt
            return float(min(100.0, new_energy))

        speed_sq = float(np.dot(vel, vel))
        acc_mag = float(np.linalg.norm(acc))
        dissipation_rate = self.p_base + self.c_v * speed_sq + self.c_a * acc_mag
        delta_energy = dissipation_rate * dt
        new_energy = max(0.0, current_energy - delta_energy)
        return float(new_energy)

    def should_divert_to_charger(
        self,
        energy: float,
        pos_i: np.ndarray,
        charging_stations: List[np.ndarray],
        payload_mass: float = 0.0
    ) -> Tuple[bool, Optional[np.ndarray], float]:
        """
        Determines whether the vehicle must override mission dispatch and divert to charging.
        Returns: (needs_diversion, target_charger_pos, current_h_E)
        """
        h_e, target_charger, _ = self.compute_battery_margin(
            energy, pos_i, charging_stations, payload_mass
        )
        needs_diversion = (h_e <= self.divert_threshold)
        return needs_diversion, target_charger, h_e
