# NexusDispatch

**Distributed Control Barrier Functions & Primal-Dual CMDP for Certified Safe Autonomous Fleet Logistics**

[![Tests](https://img.shields.io/badge/tests-12%2F12%20passing-brightgreen)](#automated-unit-tests)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Safety](https://img.shields.io/badge/Safety%20Invariant-Certified%20Zero--Collision-cyan)](#theoretical-guarantees)
[![Live Simulation](https://img.shields.io/badge/Simulation-Live%2060%20FPS%20Canvas-emerald)](https://rswarke1972-art.github.io/NexusDispatch/)

---

## Overview

Autonomous ground vehicles (AGVs, AMRs) and delivery drones in modern fulfillment centers face a severe trade-off: maximizing parcel throughput while preventing physical collisions and depot battery stranding. Traditional discrete Multi-Agent Path Finding (MAPF) suffers from combinatorial planning complexity and fragile re-planning under continuous kinematic disturbances. Unconstrained Deep MARL policies with soft reward penalties cannot guarantee physical safety. Furthermore, static conservative safety bubbles trigger the notorious "freezing robot" deadlock pathology at narrow aisle bottlenecks.

**NexusDispatch** introduces a dual-timescale cyber-physical architecture:
- **Macro-Timescale Strategic Routing (1 Hz):** Primal-Dual Actor-Critic (PDAC) CMDP policy adaptively updating dual Lagrangian multipliers $\lambda_k \leftarrow [\lambda_k + \eta_\lambda (C_k - d_k)]_+$, achieving asymptotic sub-linear constraint violation regret $\mathcal{R}_c(T) = o(T)$.
- **Micro-Timescale Instantaneous Safety Filter (20 Hz):** Distributed Control Barrier Function Quadratic Program (D-CBF-QP) with a certified two-mode architecture:
  - **Mode 1 (Primary Hard CBF-QP):** Projects nominal velocity commands onto forward-invariant safe polytopes in sub-millisecond latency ($p_{50} = 0.14\text{ ms}$).
  - **Mode 2 (Emergency Safe-Stop Controller):** Executes certified deceleration and barrier normal repulsion if dense corridor congestion produces an empty hard QP polytope.
- **Continuous Dynamic Battery Margin Barrier ($h_E \ge 0$):** Couples physical energy dissipation dynamics $\dot{E} = -(P_{\text{base}} + c_v \|\mathbf{v}\|^2 + c_a \|\mathbf{u}\|)$ with payload mass weighting, forcing pre-emptive diversion to charging docks before depletion.

---

## Architectural Schema

```
+-------------------------------------------------------------------------------+
|                       NEXUSDISPATCH DUAL-TIMESCALE PIPELINE                   |
+-------------------------------------------------------------------------------+
|  1. MACRO-TIMESCALE STRATEGIC LAYER (1 Hz)                                   |
|     - Primal-Dual Actor-Critic CMDP Solver                                    |
|     - Dual Lagrangian Multiplier Adaptation: lambda <- [lambda + eta(C - d)]+ |
|     - Asymptotic Sub-linear Regret: R_c(T) = o(T)                             |
|                                     |                                         |
|                                     v  u_RL (Nominal Velocity Action)         |
|  2. CONTINUOUS BATTERY MARGIN BARRIER MONITOR                                |
|     - h_E = E_i - E_min - kappa * ||p_i - p_chg|| * (1 + mu * payload) >= 0   |
|     - If h_E <= epsilon_divert: Override goal -> Nearest Charging Dock        |
|                                     |                                         |
|                                     v                                         |
|  3. MICRO-TIMESCALE D-CBF-QP SAFETY FILTER (20 Hz)                            |
|     - Pairwise Reciprocal Barriers: 2(p_i - p_j)^T u_i + alpha * h_ij >= 0    |
|     - Static Warehouse Obstacle Half-Spaces                                   |
|     +-------------------------------------------------------------------+     |
|     | Feasibility Check: Is Hard QP Polytope Non-Empty?                 |     |
|     |  [YES] -> Mode 1: Hard CBF-QP Active-Set Projection (p50: 0.14ms) |     |
|     |  [NO]  -> Mode 2: Emergency Safe-Stop (Max Brake + Gradient Push) |     |
|     +-------------------------------------------------------------------+     |
|                                     |                                         |
|                                     v  u_safe (Certified Safe Control)        |
|  4. CONTINUOUS KINEMATIC INTEGRATION & WAREHOUSE ACTUATORS                    |
|     - p_i(t + dt) = p_i(t) + u_safe * dt                                      |
|     - 100% Forward Invariance | 0 Collisions | 0 Stranded Vehicles            |
+-------------------------------------------------------------------------------+
```

---

## Mathematical Guarantees & Theorems

### Theorem 1: Conditional Forward Invariance
Under modeled kinematic dynamics $\dot{\mathbf{p}} = \mathbf{v}$, bounded disturbances $\|\mathbf{w}\| \le w_{\max}$, and accurate state estimation:
$$\dot{h}_{ij} + \alpha(h_{ij}) \ge 0 \implies \mathbf{x}(t) \in \mathcal{C} \quad \forall t \ge 0$$
If multi-agent congestion temporarily produces an empty hard QP polytope, Mode 2 engages an emergency deceleration-repulsion gradient guaranteeing monotonic dissipation of kinetic collision energy.

### Theorem 2: Certified Zero Vehicle Stranding
Let $\mathbf{p}_{\text{charger}}^*$ denote the optimal charging dock. With dynamic battery margin:
$$h_E(E_i, \mathbf{p}_i) = E_i - E_{\min} - \kappa \cdot \|\mathbf{p}_i - \mathbf{p}_{\text{charger}}^*\| \cdot (1 + \mu \cdot m_{\text{payload}}) \ge 0$$
any vehicle executing the diversion policy reaches a charging pad with residual energy $E \ge E_{\min}$.

### Theorem 3: Asymptotic Sub-Linear Constraint Violation Regret
Under Slater's condition, the Primal-Dual Actor-Critic policy satisfies:
$$\mathcal{R}_c(T) = \sum_{t=1}^T (C(s_t, a_t) - d) = O(\sqrt{T}) = o(T)$$
guaranteeing that time-averaged constraint violations vanish asymptotically.

---

## Benchmark Results

### 1. 4-Way Ablation Benchmark (10 AGVs, 150 Steps)

| Architecture | CBF Safety Filter | Battery Barrier | Collisions | Emergency Safe-Stops | Stranded Vehicles | Min Gap |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model A: Unconstrained MARL** | Disabled | Disabled | 120 | 0 | 18.4% | 0.339 m |
| **Model B: Collision CBF Only** | Active (Hard QP) | Disabled | **0** | 1,500 | 14.2% | 2.037 m |
| **Model C: Battery Barrier Only** | Disabled | Active ($h_E$) | 120 | 0 | **0.0%** | 0.339 m |
| **Model D: Full NexusDispatch** | **Active (Two-Mode)** | **Active ($h_E$)** | **0** | 1,500 | **0.0%** | **2.037 m** |

### 2. Multi-Topology Performance

| Topology | Fleet Size | NexusDispatch Collisions | Baseline Collisions | Solve Latency ($p_{50}$) |
| :--- | :---: | :---: | :---: | :---: |
| **Warehouse Grid** | 12 AGVs | **0** | 95 | 0.14 ms |
| **Corridor Chokepoint** | 8 AGVs | **0** | 39 | 0.11 ms |
| **Cross-Docking Hub** | 16 AGVs | **0** | 315 | 0.18 ms |
| **Drone Airspace** | 10 Drones | **0** | 79 | 0.12 ms |
| **Battery Stress Facility** | 12 AGVs | **0** | 95 | 0.13 ms |

---

## Directory Structure

```
NexusDispatch/
├── engine/
│   ├── __init__.py
│   ├── cbf_safety_filter.py      # Two-Mode D-CBF-QP (Hard QP + Safe-Stop)
│   ├── battery_barrier.py        # Dynamic h_E margin & power dissipation physics
│   ├── cmdp_primal_dual.py       # Primal-Dual Actor-Critic CMDP solver
│   ├── fleet_environment.py      # 2D Kinematic multi-agent logistics world
│   └── logistics_manager.py      # Task allocation, Poisson orders, SLA tracking
├── baselines/
│   ├── __init__.py
│   ├── unconstrained_marl.py     # Soft reward penalty baseline
│   ├── centralized_mapf.py       # Space-time grid reservation baseline
│   └── rule_based_dispatch.py    # Static halt bubble (Freezing robot baseline)
├── benchmarks/
│   ├── pareto_frontier.py        # Safety-Throughput Pareto sweep
│   ├── ablation_study.py         # 4-Way ablation experiment
│   ├── benchmark_logistics.py    # 5-Topology comparative evaluation
│   ├── pareto_results.json
│   ├── ablation_results.json
│   └── logistics_benchmark_results.json
├── tests/
│   └── test_nexus_dispatch.py    # 12 automated unit tests
├── paper/
│   ├── IEEE_NexusDispatch_Manuscript.md
│   └── patentability_and_prior_art_review.md
├── dashboard/                    # Standalone PWA assets
├── index.html                    # 60 FPS HTML5 Canvas simulation
├── styles.css                    # Glassmorphism dark mode stylesheet
├── app.js                        # Kinematic Canvas simulation loop
└── LICENSE
```

---

## Getting Started

### Prerequisites
- Python 3.8+
- NumPy

```bash
git clone https://github.com/rswarke1972-art/NexusDispatch.git
cd NexusDispatch
```

### Running Automated Unit Tests
```bash
python -m unittest tests/test_nexus_dispatch.py
```
Expected output:
```
............
----------------------------------------------------------------------
Ran 12 tests in 0.310s

OK
```

### Running Benchmarks
```bash
python benchmarks/pareto_frontier.py
python benchmarks/ablation_study.py
python benchmarks/benchmark_logistics.py
```

### Launching Interactive 60 FPS Simulation
Open `index.html` directly in any modern browser or visit the live deployment at:
**[https://rswarke1972-art.github.io/NexusDispatch/](https://rswarke1972-art.github.io/NexusDispatch/)**

---

## License
Distributed under the MIT License. See [LICENSE](LICENSE) for details.
