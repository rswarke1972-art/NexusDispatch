"""
NexusDispatch: Logistics Manager
High-level order generation, queue allocation, SLA tracking,
and fleet throughput analysis for autonomous multi-agent logistics.
"""

import time
import uuid
import numpy as np
from typing import Dict, List, Optional, Tuple


class Order:
    def __init__(self, order_id: str, pickup_pos: np.ndarray, delivery_pos: np.ndarray, creation_time: float):
        self.id = order_id
        self.pickup_pos = np.array(pickup_pos, dtype=float)
        self.delivery_pos = np.array(delivery_pos, dtype=float)
        self.creation_time = creation_time
        self.assigned_agent: Optional[str] = None
        self.pickup_time: Optional[float] = None
        self.completion_time: Optional[float] = None
        self.weight: float = 1.0  # kg


class LogisticsManager:
    """
    Coordinates high-level task allocation, order assignment queues,
    and fleet performance analytics.
    """
    def __init__(self, arrival_rate: float = 0.5):
        self.arrival_rate = arrival_rate  # Poisson arrival lambda (orders / sec)
        self.pending_orders: List[Order] = []
        self.active_orders: Dict[str, Order] = {}
        self.completed_orders: List[Order] = []

        self.sim_time = 0.0
        self.total_generated = 0

    def generate_orders(
        self,
        dt: float,
        pickup_stations: List[np.ndarray],
        delivery_stations: List[np.ndarray],
        rng: Optional[np.random.RandomState] = None
    ) -> List[Order]:
        """
        Samples new customer orders according to Poisson distribution.
        """
        self.sim_time += dt
        rng = rng or np.random.RandomState()

        # Expected arrivals in dt
        num_new = rng.poisson(self.arrival_rate * dt)
        new_orders = []

        for _ in range(num_new):
            self.total_generated += 1
            o_id = f"ord_{self.total_generated:04d}"
            p_pos = pickup_stations[rng.choice(len(pickup_stations))]
            d_pos = delivery_stations[rng.choice(len(delivery_stations))]

            order = Order(o_id, p_pos, d_pos, self.sim_time)
            self.pending_orders.append(order)
            new_orders.append(order)

        return new_orders

    def dispatch_pending(self, available_agents: List[Tuple[str, np.ndarray]]) -> List[Tuple[str, Order]]:
        """
        Assigns pending orders to idle vehicles using greedy distance heuristic.
        """
        assignments = []
        if not self.pending_orders or not available_agents:
            return assignments

        unassigned_agents = list(available_agents)

        while self.pending_orders and unassigned_agents:
            order = self.pending_orders.pop(0)
            # Find nearest idle agent to pickup position
            best_agent_idx = -1
            min_dist = float('inf')

            for idx, (ag_id, ag_pos) in enumerate(unassigned_agents):
                dist = float(np.linalg.norm(ag_pos - order.pickup_pos))
                if dist < min_dist:
                    min_dist = dist
                    best_agent_idx = idx

            if best_agent_idx >= 0:
                agent_id, _ = unassigned_agents.pop(best_agent_idx)
                order.assigned_agent = agent_id
                self.active_orders[order.id] = order
                assignments.append((agent_id, order))
            else:
                # Put back if no agent available
                self.pending_orders.insert(0, order)
                break

        return assignments

    def complete_order(self, order_id: str) -> Optional[Order]:
        """
        Marks an order as delivered and logs completion latency.
        """
        if order_id in self.active_orders:
            order = self.active_orders.pop(order_id)
            order.completion_time = self.sim_time
            self.completed_orders.append(order)
            return order
        return None

    def get_kpi_summary(self) -> Dict[str, float]:
        """
        Computes fleet-wide performance indicators.
        """
        num_completed = len(self.completed_orders)
        if num_completed > 0:
            latencies = [o.completion_time - o.creation_time for o in self.completed_orders if o.completion_time]
            avg_latency = float(np.mean(latencies)) if latencies else 0.0
            p95_latency = float(np.percentile(latencies, 95)) if latencies else 0.0
        else:
            avg_latency = 0.0
            p95_latency = 0.0

        sim_hours = max(1e-4, self.sim_time / 3600.0)
        throughput_per_hr = num_completed / sim_hours

        return {
            "total_orders_generated": self.total_generated,
            "total_orders_completed": num_completed,
            "pending_in_queue": len(self.pending_orders),
            "throughput_orders_per_hr": float(throughput_per_hr),
            "average_latency_sec": float(avg_latency),
            "p95_latency_sec": float(p95_latency)
        }
