"""
NexusDispatch: 4-Way Ablation Study
Model A: Unconstrained MARL (CBF=Off, Battery=Off)
Model B: Collision CBF Only (CBF=On, Battery=Off)
Model C: Battery Barrier Only (CBF=Off, Battery=On)
Model D: Full NexusDispatch (CBF=On, Battery=On)
"""

import os
import sys
import json
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "engine")))
from fleet_environment import FleetEnvironment


def run_ablation_study():
    configs = [
        {"model": "Model A (Unconstrained)", "cbf": False, "battery": False},
        {"model": "Model B (Collision CBF Only)", "cbf": True, "battery": False},
        {"model": "Model C (Battery Barrier Only)", "cbf": False, "battery": True},
        {"model": "Model D (Full NexusDispatch)", "cbf": True, "battery": True}
    ]

    results = []

    for cfg in configs:
        np.random.seed(42)
        env = FleetEnvironment(
            num_agents=10,
            width=40.0,
            height=30.0,
            dt=0.05,
            topology="warehouse_grid",
            enable_cbf=cfg["cbf"],
            enable_battery_barrier=cfg["battery"]
        )

        t0 = time.perf_counter()
        for _ in range(150):
            env.step()
        runtime_ms = (time.perf_counter() - t0) * 1000.0

        sim_hours = (150 * 0.05) / 3600.0
        throughput = env.total_orders_completed / sim_hours

        results.append({
            "model": cfg["model"],
            "cbf_enabled": cfg["cbf"],
            "battery_barrier_enabled": cfg["battery"],
            "orders_delivered": env.total_orders_completed,
            "throughput_per_hr": round(throughput, 1),
            "collisions": env.total_collisions,
            "emergency_stops": env.total_emergency_stops,
            "battery_exhaustions": env.total_battery_exhaustions,
            "min_distance_m": round(float(env.min_agent_distance), 3),
            "runtime_ms": round(runtime_ms, 2)
        })

    output_path = os.path.join(os.path.dirname(__file__), "ablation_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Ablation study complete (saved to {output_path}).")
    return results


if __name__ == "__main__":
    run_ablation_study()
