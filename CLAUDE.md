# Supermarket Crowd Simulation

An agent-based simulation of customer movement in a supermarket environment using the Mesa framework. The goal is to observe crowd dynamics, particularly clogging behavior when many agents are spawned in a short period.


## Project Objective

Build a minimal viable product (MVP) that simulates agents navigating a supermarket to collect products from their shopping list, then exiting. The simulation should demonstrate clogging behavior when agent density is high.


## Architecture Overview

### Technology Stack

- **Framework**: Mesa (Python agent-based modeling framework)
- **Configuration**: YAML file for simulation parameters
- **Visualization**: ASCII text-based (default), with data export for post-processing

### Mesa Components

| Component | Implementation | Description |
|-----------|----------------|-------------|
| Model | `SupermarketModel` | Contains grid, scheduler, and simulation logic |
| Agent | `CustomerAgent` | Represents a shopper with a shopping list |
| Space | `SingleGrid` | 2D grid where each cell holds at most one agent |
| Scheduler | `RandomActivation` | Agents step in random order each iteration |
| DataCollector | Built-in | Collects model and agent-level data for export |

### Grid Coordinate System

- Origin `(0, 0)` is at bottom-left corner
- X increases to the right, Y increases upward
- Matches Mesa's default coordinate system

### Cell Types (Shop Map)

| Type | Symbol | Walkable | Description |
|------|--------|----------|-------------|
| Floor | `.` | Yes | Empty walkable space |
| Wall | `#` | No | Boundary, impassable |
| Shelf | `S` | No | Obstacle, impassable |
| Product | `P` | No | Product location on shelf edge (agents interact from adjacent floor) |
| Entrance | `E` | Yes | Agent spawn point (walkable floor) |
| Exit | `X` | Yes | Agent removal point (walkable floor) |

**Note**: Product cells are impassable. Agents "pick up" a product by standing on an adjacent floor cell within `interaction_distance`.


## Agent Behavior

### Agent States

```
ENTERING → SHOPPING → DWELLING → SHOPPING → ... → EXITING → REMOVED
```

| State | Description |
|-------|-------------|
| `ENTERING` | Agent just spawned, will transition to SHOPPING |
| `SHOPPING` | Navigating toward next product on list |
| `DWELLING` | Stationary at product location for `product_dwell_time` steps |
| `EXITING` | All products collected, navigating to exit |
| `REMOVED` | Agent has exited and is removed from simulation |

### Shopping List Generation

Each agent receives a randomized shopping list at spawn time:
- List size: random between 1 and `max_products_per_agent`
- Products selected randomly from available products in the shop
- Order of products in list determines collection sequence

### Movement and Pathfinding

- **Algorithm**: Breadth-First Search (BFS) to find shortest path on walkable cells
- **Collision handling**: `SingleGrid` prevents two agents occupying the same cell
- **Blocked movement**: If target cell is occupied, agent waits (no movement that step)
- **Recalculation**: Path is recalculated each step to handle dynamic obstacles (other agents)

### Product Interaction

- Agent must be on a floor cell adjacent to (within `interaction_distance` of) the product cell
- Upon reaching a product, agent enters `DWELLING` state
- After `product_dwell_time` steps, product is marked as collected
- Agent then targets next product or transitions to `EXITING` if list complete


## Configuration (YAML)

The simulation is configured via a YAML file with the following structure:

```yaml
# Simulation parameters
simulation:
  max_steps: 2000              # Maximum simulation steps before termination
  agent_spawn_rate: 5          # Spawn one agent every N steps
  agent_spawn_count: 20        # Total agents to spawn
  random_seed: null            # Optional: seed for reproducibility (null = random)
  stuck_threshold: 200         # Steps without progress before declaring gridlock

# Agent parameters
agent:
  interaction_distance: 1      # Manhattan distance to "reach" a product
  product_dwell_time: 5        # Steps to stay at each product
  max_products_per_agent: 5    # Maximum items on shopping list

# Shop layout (programmatic definition)
shop_layout:
  size:
    width: 50
    height: 30

  # Walls are auto-generated at boundaries (x=0, x=width-1, y=0, y=height-1)

  # Shelves: rectangles that act as obstacles
  # Products are placed at accessible edges of shelves
  shelves:
    - type: rectangle
      x: 5
      y: 8
      width: 8
      height: 2
      products: ["bread", "milk"]     # Products placed on accessible shelf edges
    - type: rectangle
      x: 5
      y: 14
      width: 8
      height: 2
      products: ["eggs", "cheese"]
    - type: rectangle
      x: 20
      y: 8
      width: 8
      height: 2
      products: ["butter", "juice"]
    - type: rectangle
      x: 20
      y: 14
      width: 8
      height: 2
      products: ["cereal", "yogurt"]

  # Entrance points (agents spawn here)
  entrances:
    - x: 1
      y: 1
      width: 2
      height: 1

  # Exit points (agents leave here)
  exits:
    - x: 48
      y: 28
      width: 2
      height: 1
```

### Product Placement on Shelves

Products are automatically placed on **accessible edges** of shelves:
- An edge cell is accessible if at least one adjacent cell is walkable floor
- Products from the shelf's `products` list are distributed among accessible edge cells
- If more products than edge cells, some cells may have multiple product types

### CLI Override Parameters

The following YAML parameters can be overridden via command-line arguments:

| Flag | YAML Path | Description |
|------|-----------|-------------|
| `--agent-count` | `simulation.agent_spawn_count` | Total agents to spawn |
| `--seed` | `simulation.random_seed` | Random seed for reproducibility |

Design the CLI to be extensible for additional overrides in the future.


## Data Collection and Export

### Model-Level Metrics (per step)

- `step`: Current simulation step
- `total_agents`: Number of agents currently in simulation
- `agents_shopping`: Agents in SHOPPING state
- `agents_dwelling`: Agents in DWELLING state
- `agents_exiting`: Agents in EXITING state
- `agents_completed`: Cumulative count of agents who have exited

### Agent-Level Data (per step)

- `agent_id`: Unique identifier
- `position`: `(x, y)` coordinates
- `state`: Current state
- `products_remaining`: Count of uncollected products

### Export Format

Data is exported to CSV files via Mesa's `DataCollector`:
- `model_data.csv`: Model-level metrics
- `agent_data.csv`: Agent-level data

This enables post-processing for animations or analysis.


## MVP Requirements

1. Single entrance and single exit
2. Agents spawn at entrance with randomized shopping lists
3. Agents navigate to each product in order using BFS pathfinding
4. Agents dwell at products, then proceed to next or exit
5. `SingleGrid` ensures no two agents occupy the same cell
6. ASCII visualization (console output) showing grid state
7. Data export for all steps (enables animation creation)
8. Configurable via YAML with CLI overrides


## Deferred Features (Post-MVP)

- Deadlock detection and resolution
- Multiple entrances/exits
- Agent priority/urgency levels
- A* pathfinding (optimization)
- Real-time visualization (Mesa's built-in server)
- Circular shelf shapes


## Project Structure

```
supermarket-mesa/
├── src/
│   ├── main.py              # CLI entry point
│   ├── model.py             # SupermarketModel class
│   ├── agent.py             # CustomerAgent class
│   ├── grid_builder.py      # Build grid from YAML config
│   ├── pathfinding.py       # BFS pathfinding implementation
│   └── visualization.py     # ASCII renderer
├── config.yaml              # Default configuration
├── CLAUDE.md                # This specification
└── requirements.txt         # Python dependencies
```


## Running the Simulation

```bash
# Install dependencies
pip install mesa pyyaml

# Basic run with config file
python src/main.py --config config.yaml

# Override agent count
python src/main.py --config config.yaml --agent-count 50

# Set random seed for reproducibility
python src/main.py --config config.yaml --seed 42
```


# Python Runtime Management
how to control python on debian or ubuntu?
you may need to install python3-venv.

create the python venv one layer above the git repository.

# source code management
commit at each logic change

do not commit __pycache__ folder and pyc files
if these files are not in .gitignore. put them inside root .gitignore.

