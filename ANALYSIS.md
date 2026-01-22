# Supermarket Crowd Simulation Analysis

This document explains the analysis plots generated from the supermarket agent-based model simulation. The analysis follows PyCX-style principles, focusing on educational clarity and complex systems concepts.

## Simulation Configuration

- **Shop dimensions**: 50 x 30 cells
- **Total agents**: 200 customers
- **Spawn rate**: 1 agent per step (rush hour scenario)
- **Max products per agent**: 5 items
- **Simulation duration**: 523 steps (until all agents exit)

---

## Plot 1: Population Dynamics

**File**: `analysis_population_dynamics.png`

### Panel 1.1: Total Agents Over Time

This shows the number of agents present in the supermarket at each time step.

**What to observe**:
- **Buildup phase** (steps 0-200): Linear increase as agents spawn at constant rate
- **Peak phase** (steps ~150-200): System reaches maximum capacity (~157 agents)
- **Clearing phase** (steps 200+): Exponential-like decay as agents complete shopping

**Complex systems insight**: The peak value represents the system's **carrying capacity** - the maximum number of agents the shop can support given the layout and agent behavior. This emerges from the interaction between arrival rate and service rate.

### Panel 1.2: Agent State Distribution (Stacked)

Shows how agents are distributed across states: Shopping (blue), Dwelling (red), Exiting (green).

**What to observe**:
- Shopping dominates during buildup - agents spread through the store
- Dwelling spikes correspond to agents picking up products
- Exiting grows as agents complete their lists

**Complex systems insight**: The ratio between states reflects **flow balance**. When shopping >> dwelling, agents are mostly navigating. High dwelling indicates service activity. High exiting during peak shows potential **exit bottleneck**.

### Panel 1.3: Cumulative Completed

S-curve (sigmoid) of cumulative agent completions.

**What to observe**:
- Slow start: few completions as first agents still shopping
- Linear middle: steady throughput during balanced operation
- Tapering end: fewer agents remain to complete

**Complex systems insight**: This is a classic **growth curve** seen in many systems. The inflection point indicates when throughput is maximized.

### Panel 1.4: Agent Density

Percentage of walkable space occupied by agents.

**What to observe**:
- Peak density around 15% of walkable cells
- Congestion threshold marked at ~15%

**Complex systems insight**: In crowd dynamics, density strongly predicts flow behavior. Below ~15% density, agents move relatively freely. Above this threshold, **jamming** and **clogging** become more likely.

---

## Plot 2: Throughput Analysis

**File**: `analysis_throughput.png`

### Panel 2.1: Throughput Over Time

Number of agents completing shopping per 20-step window.

**What to observe**:
- Low throughput initially (no completions yet)
- Peak throughput around steps 150-250
- Gradual decline as population decreases

**Complex systems insight**: Throughput saturation occurs when adding more agents doesn't increase completion rate. This is analogous to **traffic flow saturation** where adding more cars doesn't increase road throughput.

### Panel 2.2: Fundamental Diagram

Throughput vs. density scatter plot (colored by time).

**What to observe**:
- Rising phase (cool colors): increasing density leads to increasing throughput
- Peak region: maximum throughput at moderate density
- Falling phase (warm colors): high density actually reduces throughput

**Complex systems insight**: This is the **fundamental diagram** from traffic flow theory. It reveals three regimes:
1. **Free flow**: low density, throughput scales with density
2. **Capacity**: optimal density, maximum throughput
3. **Congested**: high density, throughput decreases (jamming)

The hysteresis loop (different paths going up vs down) indicates **metastability** - the system doesn't instantly transition between regimes.

---

## Plot 3: Phase Transitions

**File**: `analysis_phase_transitions.png`

### Panel 3.1: Rate of Change in Population

First derivative of agent count (d(agents)/dt).

**What to observe**:
- Positive during buildup (agents entering > exiting)
- Zero crossing around step 100 (equilibrium attempt)
- Negative during clearing (agents exiting > entering)

**Complex systems insight**: Zero crossings indicate **equilibrium points**. Fluctuations around zero during peak phase show the system oscillating around a quasi-steady state.

### Panel 3.2: Phase Space Trajectory

2D plot of population vs. rate of change.

**What to observe**:
- Spiral/loop pattern characteristic of dynamical systems
- Trajectory moves clockwise from origin
- Converges back to origin (empty shop)

**Complex systems insight**: Phase portraits reveal system dynamics. The closed loop indicates a **transient response** to the "impulse" of spawning 200 agents. The shape shows how the system absorbs and processes this perturbation.

### Panel 3.3: Active Agent Ratio

Fraction of agents in active states (shopping + exiting) vs passive (dwelling).

**What to observe**:
- High ratio means agents are moving/seeking products
- Dips indicate more agents are at products (dwelling)

**Complex systems insight**: This ratio indicates **flow efficiency**. Sustained high values suggest agents spend more time navigating than shopping - a potential sign of congestion.

### Panel 3.4: Congestion Index

Heuristic measure of crowding effects.

**What to observe**:
- Higher values indicate more potential blocking
- Peaks correspond to periods of maximum crowding

**Complex systems insight**: Congestion is an **emergent property** - it arises from individual agent interactions, not explicit rules. The index captures collective blocking behavior.

---

## Plot 4: System Efficiency

**File**: `analysis_efficiency.png`

### Panel 4.1: Completion Milestones

Time required to reach each 10% completion increment.

**What to observe**:
- First milestones take longer (initial shopping time)
- Middle milestones are roughly evenly spaced (steady throughput)
- Later milestones may slow (fewer agents, but less congestion)

**Complex systems insight**: Nonlinear spacing indicates **non-constant processing rate**. Early delays are "warm-up" effects; late acceleration shows the benefit of reduced congestion.

### Panel 4.2: Efficiency Over Time

Completions per agent-step (normalized throughput).

**What to observe**:
- Low efficiency during buildup (agents shopping, none completing)
- Peak efficiency during steady operation
- Variable efficiency during clearing phase

**Complex systems insight**: Efficiency measures **system utilization**. Maximum efficiency corresponds to optimal operating conditions. The simulation reveals when the supermarket is "running well" vs "overwhelmed."

---

## Plot 5: Queuing Analysis

**File**: `analysis_queuing.png`

### Panel 5.1: Queue Analysis (Little's Law)

Cumulative arrivals, departures, and queue length.

**What to observe**:
- Gap between arrival and departure curves = agents in system
- Maximum gap at peak congestion
- Curves converge when all agents complete

**Complex systems insight**: **Little's Law** (L = λW) is fundamental to queuing theory:
- L = average number in system
- λ = arrival rate
- W = average time in system

The simulation validates this relationship in a complex multi-server system.

### Panel 5.2: Estimated Time in System

Derived from queue length / arrival rate.

**What to observe**:
- Peaks around 150-160 steps when queue is longest
- Represents average "residence time"

**Complex systems insight**: Time in system is a key **quality of service** metric. High values indicate customers waiting, potentially experiencing frustration or abandonment (not modeled here).

### Panel 5.3: Average State Distribution

Bar chart of mean agents in each state (during peak period).

**What to observe**:
- Shopping dominates - agents spend most time navigating
- Dwelling is relatively small - product pickup is fast
- Exiting queue indicates potential bottleneck

**Complex systems insight**: State distribution reveals **bottlenecks**. If exiting is large, the exit is congested. If shopping is large but dwelling is small, pathfinding/navigation is the limiting factor.

### Panel 5.4: Service vs Queue Ratio

Dwelling / (Shopping + Dwelling) over time.

**What to observe**:
- Low ratio means agents are queuing/navigating
- High ratio means agents are being served

**Complex systems insight**: Analogous to **utilization** in queuing models. Healthy systems maintain balance; extreme values indicate dysfunction.

---

## Plot 6: Summary Dashboard

**File**: `analysis_summary.png`

Consolidated view with key metrics and observations.

### Key Findings

1. **Capacity**: System supports up to ~157 simultaneous agents
2. **Peak timing**: Maximum congestion at step ~193
3. **Throughput**: Average 0.38 agents/step completion rate
4. **Duration**: 523 steps to clear all 200 agents

### Complex Systems Insights

1. **Emergence**: Congestion patterns emerge from simple agent rules
2. **Self-organization**: Agents naturally form queues and lanes
3. **Phase transitions**: System moves through distinct operational phases
4. **Nonlinearity**: Throughput doesn't scale linearly with agent count

---

## Running the Analysis

```bash
cd supermarket-mesa
python analysis_pycx.py
```

This will generate all 6 PNG files and display the plots interactively.

---

## Further Exploration

To study complex systems behavior more deeply:

1. **Parameter sensitivity**: Vary spawn rate, dwell time, shop layout
2. **Critical thresholds**: Find exact density where congestion onset occurs
3. **Layout optimization**: Test different shelf arrangements for throughput
4. **Deadlock detection**: Identify conditions that cause complete gridlock
5. **Spatial analysis**: Map position data to identify hotspots

---

## References

- Helbing, D. (2001). Traffic and related self-driven many-particle systems
- Schadschneider, A. (2002). Traffic flow: a statistical physics point of view
- Little, J.D.C. (1961). A Proof for the Queuing Formula: L = λW
- Sayama, H. (2013). PyCX: A Python-based simulation code repository
