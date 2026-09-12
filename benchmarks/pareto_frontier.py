"""
NexusDispatch: Safety-Throughput-Deadlock Pareto Frontier Benchmark
Sweeps fleet sizes N in {4, 8, 12, 16, 20} and safety margins d_safe in {0.8, 1.0, 1.2, 1.5}
to map the empirical trade-off between delivery throughput and safety guarantees.
"""

import os
import sys
import json
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "engine")))
from fleet_environment import FleetEnvironment


def run_pareto_sweep():
    fleet_sizes = [4, 8, 12, 16, 20]
    safety_margins = [0.8, 1.0, 1.2, 1.5]

    results = []

    for n_agents in fleet_sizes:
        for d_safe in safety_margins:
            env = FleetEnvironment(
                num_agents=n_agents,
                width=40.0,
                height=30.0,
                dt=0.05,
                topology="warehouse_grid",
                enable_cbf=True,
                enable_battery_barrier=True
            )
            env.cbf_filter.d_safe = d_safe

            t0 = time.perf_counter()
            for _ in range(120):
                env.step()
            sim_time = time.perf_counter() - t0

            # Simulation time in hours (120 steps * 0.05s = 6.0s = 0.001666 hrs)
            sim_hours = (120 * 0.05) / 3600.0
            throughput = env.total_orders_completed / sim_hours

            results.append({
                "fleet_size": n_agents,
                "d_safe_m": d_safe,
                "orders_delivered": env.total_orders_completed,
                "throughput_per_hr": round(throughput, 1),
                "collisions": env.total_collisions,
                "emergency_stops": env.total_emergency_stops,
                "battery_exhaustions": env.total_battery_exhaustions,
                "min_distance_m": round(float(env.min_agent_distance), 3),
                "sim_runtime_ms": round(sim_time * 1000.0, 2)
            })

    output_path = os.path.join(os.path.dirname(__file__), "pareto_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Pareto frontier benchmark complete ({len(results)} points saved to {output_path}).")
    return results


if __name__ == "__main__":
    run_pareto_sweep()
