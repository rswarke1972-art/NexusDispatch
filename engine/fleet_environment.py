"""
NexusDispatch: Fleet Logistics Environment
Simulates 2D multi-agent kinematic fleet coordination in warehouse layouts,
cross-docking hubs, chokepoint corridors, and drone airspace topologies.
Integrates continuous-time kinematics, battery dissipation, and CBF safety filtering.
"""

import math
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

from cbf_safety_filter import CBFSafetyFilter
from battery_barrier import BatteryBarrierManager


class AgentState:
    def __init__(self, agent_id: str, pos: np.ndarray, battery: float = 100.0):
        self.id = agent_id
        self.pos = np.array(pos, dtype=float)
        self.vel = np.zeros(2, dtype=float)
        self.battery = battery
        self.payload = 0.0  # kg
        self.target_pos: Optional[np.ndarray] = None
        self.task_type: str = "IDLE"  # IDLE, PICKUP, DELIVER, CHARGING
        self.charging: bool = False
        self.assigned_order_id: Optional[str] = None
        self.completed_deliveries = 0
        self.emergency_stops = 0
        self.mode_history = []


class FleetEnvironment:
    """
    Multi-Agent Logistics Environment with continuous kinematics,
    static obstacle barriers, dynamic battery management, and chokepoints.
    """
    def __init__(
        self,
        num_agents: int = 8,
        width: float = 40.0,
        height: float = 30.0,
        dt: float = 0.05,
        topology: str = "warehouse_grid",
        enable_cbf: bool = True,
        enable_battery_barrier: bool = True
    ):
        self.num_agents = num_agents
        self.width = width
        self.height = height
        self.dt = dt
        self.topology = topology
        self.enable_cbf = enable_cbf
        self.enable_battery_barrier = enable_battery_barrier

        # Initialize Safety Filter and Battery Manager
        self.cbf_filter = CBFSafetyFilter(d_safe=1.2, v_max=2.0, a_max=3.0, dt=dt)
        self.battery_manager = BatteryBarrierManager(e_min=5.0, divert_threshold=4.0)

        # Environment elements
        self.charging_stations: List[np.ndarray] = []
        self.pickup_stations: List[np.ndarray] = []
        self.delivery_stations: List[np.ndarray] = []
        self.obstacles: List[Tuple[np.ndarray, np.ndarray, float]] = []  # (center, normal, margin)
        self.obstacles_rects: List[Tuple[float, float, float, float]] = []  # (x, y, w, h) for collision check

        self._setup_topology(topology)

        # Spawn agents
        self.agents: Dict[str, AgentState] = {}
        self._spawn_agents()

        # Telemetry & Diagnostics
        self.steps = 0
        self.total_orders_completed = 0
        self.total_collisions = 0
        self.total_battery_exhaustions = 0
        self.total_emergency_stops = 0
        self.min_agent_distance = float('inf')

    def _setup_topology(self, topology: str):
        """
        Builds warehouse racks, narrow chokepoints, or open airspace.
        """
        self.charging_stations = [
            np.array([4.0, 4.0]),
            np.array([4.0, self.height - 4.0]),
            np.array([self.width - 4.0, 4.0]),
            np.array([self.width - 4.0, self.height - 4.0])
        ]
        self.pickup_stations = [
            np.array([8.0, 8.0]),
            np.array([8.0, self.height - 8.0])
        ]
        self.delivery_stations = [
            np.array([self.width - 8.0, 8.0]),
            np.array([self.width - 8.0, self.height - 8.0])
        ]

        if topology == "bidirectional_chokepoint":
            # Two large walls creating a 3.0-meter central corridor bottleneck
            corridor_y_center = self.height / 2.0
            corridor_gap = 2.0  # narrow gap
            wall_thickness = 4.0
            wall_x = self.width / 2.0

            # Upper wall
            self.obstacles_rects.append((wall_x - 1.0, corridor_y_center + corridor_gap, 2.0, self.height))
            # Lower wall
            self.obstacles_rects.append((wall_x - 1.0, 0.0, 2.0, corridor_y_center - corridor_gap))

            # Add barrier normals
            self.obstacles.append((np.array([wall_x, corridor_y_center + corridor_gap]), np.array([0.0, -1.0]), 0.6))
            self.obstacles.append((np.array([wall_x, corridor_y_center - corridor_gap]), np.array([0.0, 1.0]), 0.6))

        elif topology == "warehouse_grid":
            # Regular grid of 4 storage racks
            for rx in [14.0, 26.0]:
                for ry in [7.0, 19.0]:
                    self.obstacles_rects.append((rx, ry, 3.0, 7.0))
                    center = np.array([rx + 1.5, ry + 3.5])
                    # 4 half-spaces for each box
                    self.obstacles.append((np.array([rx, ry + 3.5]), np.array([-1.0, 0.0]), 0.5))
                    self.obstacles.append((np.array([rx + 3.0, ry + 3.5]), np.array([1.0, 0.0]), 0.5))

    def _spawn_agents(self):
        self.agents.clear()
        rng = np.random.RandomState(42)
        for i in range(self.num_agents):
            agent_id = f"agv_{i:02d}"
            # Even agents start on left, odd agents start on right
            if i % 2 == 0:
                pos = np.array([5.0 + rng.uniform(-1.0, 1.0), 6.0 + (i * 2.5) % (self.height - 12.0)])
                initial_target = self.delivery_stations[rng.choice(len(self.delivery_stations))]
            else:
                pos = np.array([self.width - 5.0 + rng.uniform(-1.0, 1.0), 6.0 + (i * 2.5) % (self.height - 12.0)])
                initial_target = self.pickup_stations[rng.choice(len(self.pickup_stations))]

            agent = AgentState(agent_id, pos, battery=rng.uniform(70.0, 95.0))
            agent.target_pos = np.array(initial_target, dtype=float)
            agent.task_type = "DELIVER" if i % 2 == 0 else "PICKUP"
            self.agents[agent_id] = agent

    def step(self, nominal_actions: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, Any]:
        """
        Executes one micro-timescale simulation step (dt = 0.05s).
        Applies CBF safety filter and battery dynamics to all agents.
        """
        self.steps += 1
        nominal_actions = nominal_actions or {}
        step_telemetry = {}

        # 1. Generate nominal control if not supplied
        for agent_id, agent in self.agents.items():
            if agent_id not in nominal_actions:
                if agent.target_pos is not None:
                    diff = agent.target_pos - agent.pos
                    dist = float(np.linalg.norm(diff))
                    if dist > 0.1:
                        nominal_v = (diff / dist) * min(self.cbf_filter.v_max, dist)
                    else:
                        nominal_v = np.zeros(2)
                else:
                    nominal_v = np.zeros(2)
                nominal_actions[agent_id] = nominal_v

        # 2. Check Battery Barrier & Override destination if battery low
        if self.enable_battery_barrier:
            for agent_id, agent in self.agents.items():
                if not agent.charging:
                    needs_div, ch_pos, h_e = self.battery_manager.should_divert_to_charger(
                        agent.battery, agent.pos, self.charging_stations, agent.payload
                    )
                    if needs_div and ch_pos is not None:
                        agent.target_pos = ch_pos
                        agent.task_type = "CHARGING"
                        # Recompute nominal heading towards charger
                        diff_ch = ch_pos - agent.pos
                        dist_ch = float(np.linalg.norm(diff_ch))
                        if dist_ch > 0.2:
                            nominal_actions[agent_id] = (diff_ch / dist_ch) * min(self.cbf_filter.v_max, dist_ch)

        # 3. Apply CBF Safety Filter to each agent
        actual_controls = {}
        for agent_id, agent in self.agents.items():
            u_rl = nominal_actions[agent_id]

            if self.enable_cbf:
                # Assemble neighbor list
                neighbors = []
                for other_id, other in self.agents.items():
                    if other_id != agent_id:
                        neighbors.append((other_id, other.pos, other.vel))

                u_safe, mode, meta = self.cbf_filter.compute_safe_control(
                    agent_id=agent_id,
                    pos_i=agent.pos,
                    vel_i=agent.vel,
                    u_rl=u_rl,
                    neighbors=neighbors,
                    obstacles=self.obstacles
                )
                actual_controls[agent_id] = u_safe
                agent.mode_history.append(mode)
                if mode == "EMERGENCY_STOP":
                    agent.emergency_stops += 1
                    self.total_emergency_stops += 1
                step_telemetry[agent_id] = meta
            else:
                # Unconstrained Baseline
                u_unconstrained = np.clip(u_rl, -self.cbf_filter.v_max, self.cbf_filter.v_max)
                actual_controls[agent_id] = u_unconstrained
                step_telemetry[agent_id] = {"mode": "UNCONSTRAINED", "slack": 0.0}

        # 4. Integrate continuous kinematics & battery dissipation
        for agent_id, agent in self.agents.items():
            u = actual_controls[agent_id]
            # Kinematics: pos += u * dt, vel = u
            acc = (u - agent.vel) / self.dt
            agent.pos += u * self.dt
            agent.vel = u

            # Boundary clipping
            agent.pos[0] = np.clip(agent.pos[0], 1.0, self.width - 1.0)
            agent.pos[1] = np.clip(agent.pos[1], 1.0, self.height - 1.0)

            # Check if arrived at charging station
            is_at_charger = False
            for ch in self.charging_stations:
                if np.linalg.norm(agent.pos - ch) < 1.0:
                    is_at_charger = True
                    break

            if agent.task_type == "CHARGING" and is_at_charger:
                agent.charging = True
            else:
                agent.charging = False

            # Energy step
            agent.battery = self.battery_manager.step_energy(
                agent.battery, agent.vel, acc, self.dt, is_charging=agent.charging
            )
            if agent.battery <= 0.1:
                self.total_battery_exhaustions += 1

            if agent.charging and agent.battery >= 90.0:
                agent.charging = False
                agent.task_type = "PICKUP"
                agent.target_pos = self.pickup_stations[np.random.choice(len(self.pickup_stations))]

            # Check delivery / pickup completion
            if agent.target_pos is not None:
                dist_target = float(np.linalg.norm(agent.pos - agent.target_pos))
                if dist_target < 1.2:
                    if agent.task_type == "PICKUP":
                        agent.payload = 1.0
                        agent.task_type = "DELIVER"
                        agent.target_pos = self.delivery_stations[np.random.choice(len(self.delivery_stations))]
                    elif agent.task_type == "DELIVER":
                        agent.payload = 0.0
                        agent.completed_deliveries += 1
                        self.total_orders_completed += 1
                        agent.task_type = "PICKUP"
                        agent.target_pos = self.pickup_stations[np.random.choice(len(self.pickup_stations))]

        # 5. Collision & Safety Checking
        agents_list = list(self.agents.values())
        for i in range(len(agents_list)):
            for j in range(i + 1, len(agents_list)):
                dist = float(np.linalg.norm(agents_list[i].pos - agents_list[j].pos))
                if dist < self.min_agent_distance:
                    self.min_agent_distance = dist
                if dist < self.cbf_filter.d_safe:
                    self.total_collisions += 1

        return {
            "step": self.steps,
            "orders_completed": self.total_orders_completed,
            "collisions": self.total_collisions,
            "battery_exhaustions": self.total_battery_exhaustions,
            "emergency_stops": self.total_emergency_stops,
            "min_distance": self.min_agent_distance,
            "telemetry": step_telemetry
        }
