# NexusDispatch: Formal Patentability Analysis & Prior Art Specifications

**Filing Entity:** Autonomous Systems and Robotics Research Group  
**Lead Inventor:** Sahil Warke  
**Technical Classification:** CPC G05D 1/0289, G06Q 10/08, G06N 3/092, B65G 1/137  

---

## Part I: 12 Formal Patent Claims Specifications

### Independent Claim 1 (Method):
A computer-implemented method for certified collision-free and energy-resilient coordination of a multi-agent autonomous vehicle fleet within a physical operating environment, the method comprising:
1. Receiving, at a macro-timescale planning layer, mission task orders and vehicle operational telemetry for a plurality of autonomous vehicles operating within the physical environment;
2. Generating, via a Constrained Markov Decision Process (CMDP) solver utilizing a Primal-Dual Actor-Critic architecture, nominal routing velocity commands for each autonomous vehicle, wherein dual Lagrangian multipliers are adaptively updated based on cumulative constraint violations to enforce an asymptotic sub-linear constraint violation regret $\mathcal{R}_c(T) = o(T)$;
3. Monitoring, for each autonomous vehicle, a dynamic battery-to-charger margin barrier $h_E(E_i, \mathbf{p}_i) \ge 0$ as a function of remaining battery state-of-charge $E_i$, physical vehicle position $\mathbf{p}_i$, cargo payload mass $m_i$, and quadratic velocity power dissipation dynamics;
4. Pre-empting nominal mission dispatch commands and routing the vehicle to a target charging pad upon detecting that the dynamic battery margin barrier satisfies $h_E \le \epsilon_{\text{divert}}$;
5. Evaluating, at a micro-timescale safety layer executing at a higher frequency than the macro-timescale planning layer, a distributed Control Barrier Function Quadratic Program (D-CBF-QP) based on local neighbor positions and velocities within a sensing radius;
6. In a primary operational mode (Mode 1), projecting the nominal routing velocity command onto a safe forward-invariant control polytope defined by inter-vehicle and static obstacle barrier constraints to output a certified safe control command; and
7. In an emergency safe-stop mode (Mode 2), upon determining that the safe forward-invariant control polytope is empty due to conflicting multi-agent boundary constraints, commanding an emergency deceleration along the current velocity vector combined with a repulsive gradient directed along barrier normals away from adjacent vehicles.

### Dependent Claim 2 (Mode 2 Repulsion Gradient):
The method of claim 1, wherein the emergency deceleration in Mode 2 comprises a closed-form velocity update:
$$\mathbf{u}_{\text{brake}} = \mathbf{v}_i \cdot \max\left(0, 1 - \frac{a_{\max} \Delta t}{\|\mathbf{v}_i\|}\right) + \sum_{j \in \mathcal{N}_i, \|\mathbf{p}_i - \mathbf{p}_j\| < d_{\text{safe}}} \beta \cdot \frac{\mathbf{p}_i - \mathbf{p}_j}{\|\mathbf{p}_i - \mathbf{p}_j\|}$$
wherein $\beta$ is a strictly positive gradient gain coefficient preventing physical collision during corridor bottleneck congestion.

### Dependent Claim 3 (Dynamic Battery Margin Barrier):
The method of claim 1, wherein the dynamic battery margin barrier $h_E(E_i, \mathbf{p}_i)$ is formulated as:
$$h_E(E_i, \mathbf{p}_i) = E_i - E_{\min} - \kappa \cdot \|\mathbf{p}_i - \mathbf{p}_{\text{charger}}^*\| \cdot (1 + \mu \cdot m_{\text{payload}})$$
wherein $E_{\min}$ is a certified non-depletable cutoff reserve, $\kappa$ is a distance-to-energy consumption coefficient, $\mu$ is a payload mass penalty scalar, and $\mathbf{p}_{\text{charger}}^*$ is an optimal charging station coordinate selected based on minimum traversal distance and queue vacancy.

### Dependent Claim 4 (Power Dissipation Dynamics):
The method of claim 3, wherein the vehicle state-of-charge dissipation over time $\dot{E}_i(t)$ comprises:
$$\dot{E}_i(t) = -(P_{\text{base}} + c_v \|\mathbf{v}_i\|^2 + c_a \|\mathbf{u}_i\|)$$
incorporating idle electronic load $P_{\text{base}}$, aerodynamic velocity drag loss $c_v \|\mathbf{v}_i\|^2$, and mechanical acceleration torque expenditure $c_a \|\mathbf{u}_i\|$.

### Dependent Claim 5 (Reciprocal CBF Formulation):
The method of claim 1, wherein the inter-vehicle barrier function $h_{ij}(\mathbf{p}_i, \mathbf{p}_j) = \|\mathbf{p}_i - \mathbf{p}_j\|^2 - d_{\text{safe}}^2$ satisfies the reciprocal barrier inequality:
$$2(\mathbf{p}_i - \mathbf{p}_j)^T \mathbf{u}_i - 2(\mathbf{p}_i - \mathbf{p}_j)^T \mathbf{v}_j + \alpha h_{ij} \ge 0$$
wherein $\alpha > 0$ denotes a linear class-$\mathcal{K}$ parameter, ensuring symmetric velocity sharing without centralized arbitration.

### Dependent Claim 6 (Anti-Freezing Robot Guarantee):
The method of claim 1, wherein the micro-timescale safety filter resolves opposing head-on vehicle trajectories within a bidirectional aisle by continuous transverse velocity deflection, thereby avoiding static zero-velocity deadlocks characteristic of threshold halt bubbles.

---

### Independent Claim 7 (System):
A cyber-physical multi-agent fleet logistics management system, comprising:
1. A plurality of autonomous mobile vehicles, each equipped with wheel actuators, power storage units, and localization sensors;
2. A wireless communications transceiver configured to exchange local velocity and position state estimates with neighboring vehicles within a sensing radius;
3. A centralized fleet management server communicatively coupled to the plurality of autonomous mobile vehicles; and
4. An on-board electronic control unit (ECU) disposed on each autonomous mobile vehicle, the ECU comprising a hardware processor and non-transitory memory storing instructions that, when executed, cause the ECU to:
   - Receive nominal strategic dispatch waypoints from the centralized fleet management server;
   - Compute instantaneous velocity actions utilizing a continuous Control Barrier Function Quadratic Program (CBF-QP) executed at a cycle frequency of at least 20 Hz;
   - Project the nominal waypoints onto a safe forward-invariant polytope during normal operational conditions;
   - Trigger an emergency safe-stop braking mode with normal gradient repulsion when multiple conflicting vehicle constraints produce an infeasible QP polytope; and
   - Override task navigation and steer to a designated charging dock when a dynamic battery-to-charger margin barrier $h_E$ drops below a certified diversion threshold.

### Dependent Claim 8 (Active-Set Microsecond QP Solver):
The system of claim 7, wherein the ECU executes an active-set boundary projection algorithm solving the two-dimensional quadratic program in a deterministic latency of less than 0.50 milliseconds ($p_{95} \le 0.50\text{ ms}$).

### Dependent Claim 9 (Dual Subgradient Adaptation Engine):
The system of claim 7, wherein the fleet management server updates Lagrangian multipliers $\boldsymbol{\lambda}^{(t+1)} = [\boldsymbol{\lambda}^{(t)} + \eta_\lambda (\mathbf{C} - \mathbf{d})]_+$ using projected subgradient ascent, continuously penalizing route selections through congested warehouse aisles.

### Dependent Claim 10 (Multi-Topology Corridor Adaptation):
The system of claim 7, wherein the ECU dynamically reconfigures obstacle half-space barrier normals according to warehouse storage rack geometries, bidirectional corridor chokepoints, and cross-docking sorting hub barriers.

---

### Independent Claim 11 (Non-Transitory Computer-Readable Medium):
A non-transitory computer-readable storage medium storing instructions that, when executed by one or more processors of an autonomous vehicle coordination system, cause the system to:
1. Formulate a multi-agent routing objective as a Constrained Markov Decision Process with dual multiplier penalty terms;
2. Enforce forward invariance of a collision-free safe set $\mathcal{C}$ using continuous distributed Control Barrier Functions;
3. Track an energy depletion barrier $h_E = E - E_{\min} - E_{\text{reserve}} \ge 0$;
4. Automatically switch between a primary hard QP projection mode and an emergency braking deflection mode based on polytope feasibility; and
5. Deliver packages to destination stations with certified zero inter-vehicle collisions and certified zero depot energy exhaustion.

### Dependent Claim 12 (Aerial Drone Logistics Airspace):
The computer-readable medium of claim 11, wherein the autonomous vehicles comprise aerial delivery drones, and the Control Barrier Function Quadratic Program enforces three-dimensional ellipsoidal separation hulls with vertical altitude tiering.

---

## Part II: Prior Art Comparative Matrix & Non-Obviousness Review

| Technology / Patent | Architecture | Safety Guarantee | Energy Coupling | Deadlock / Freezing Robot Handling |
| :--- | :--- | :--- | :--- | :--- |
| **Amazon Kiva Systems** (US 8,606,392) | Centralized grid reservation table | Discrete space-time cell exclusion | Static threshold trigger | Vehicles halt at conflicting intersections; prone to gridlock |
| **Ocado AMR Grid** (US 10,661,987) | Centralized overhead 2D grid planner | Discretized coordinate reservation | Fixed schedule charging slots | Central planner must re-solve entire grid if an AMR slips or delays |
| **Safe Deep MARL** (OpenAI / DeepMind) | Neural policy with soft Lagrangian reward penalty | **None** (Statistical soft penalties only; non-zero failure rate) | Heuristic reward shaping | Neural policy oscillates or crashes under distribution shift |
| **Classical CBF-QP** (Ames et al., Caltech) | Continuous control barrier function | Continuous forward invariance | Uncoupled from battery dynamics | Introduces soft slacks $\xi > 0$ when infeasible, destroying forward invariance |
| **NexusDispatch (Present Invention)** | **Dual-timescale CMDP + Two-Mode D-CBF-QP** | **Certified Forward Invariance (Mode 1) + Guaranteed Safe-Stop (Mode 2)** | **Coupled Dynamic Margin Barrier $h_E(E_i, \mathbf{p}_i) \ge 0$** | **Reciprocal continuous velocity projection; 0 freezing deadlocks** |

### Patentability Arguments (35 U.S.C. §§ 101, 102, 103):
1. **Eligible Subject Matter (§ 101):** NexusDispatch directly commands physical actuators (motors, steering wheels, rotors) on cyber-physical machines to prevent physical destruction and stranding. It is not an abstract mathematical concept.
2. **Novelty (§ 102):** No prior art integrates a Two-Mode D-CBF-QP (hard projection without soft slack + gradient-guided safe-stop) with a continuous dynamic battery margin barrier $h_E$ in a dual-timescale CMDP logistics pipeline.
3. **Non-Obviousness (§ 103):** Classical CBF literature uniformly resolves QP infeasibility by relaxing constraints with slack variables $\xi > 0$. However, slacks forfeit formal safety. NexusDispatch's rejection of soft slacks in favor of the Mode 2 emergency braking-repulsion gradient provides a non-obvious solution to multi-agent chokepoint congestion.
