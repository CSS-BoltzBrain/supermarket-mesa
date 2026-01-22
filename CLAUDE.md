# Supermarket Crowd Simulation

An agent-based simulation of customer movement in a supermarket environment. The goal is to observe crowd dynamics, particularly clogging behavior when many agents are spawned in a short period.


## Project Objective

Build a minimal viable product (MVP) that simulates agents navigating a supermarket to collect products from their shopping list, then exiting. The simulation should demonstrate clogging behavior when agent density is high.


## Architecture Overview

- based on mesas, the agent-based modelling softwarer framework
- able to export state of the whole system and agents at any moment for users' postprocessing, including all moments so an animation or gif is possible.
- consuming all simulation parameters from a configuration file in yaml format.
- default visualization is ascii-text based.
- the simulation is a 2D supermarket.

### Cell Types (Shop Map)

| Type | Symbol | Description |
|------|--------|-------------|
| Floor | `.` | Walkable space |
| Wall | `#` | Boundary, impassable |
| Shelf | `S` | Obstacle, impassable |
| Product | `P` | Product location (on shelf edge), impassable |
| Entrance | `E` | Agent spawn point |
| Exit | `X` | Agent removal point |


## Configuration (YAML)

The simulation is configured via a YAML file with the following structure:

```yaml
# Simulation parameters
simulation:
  max_iterations: 1000
  agent_spawn_rate: 5        # Spawn one agent every N iterations
  agent_spawn_count: 20      # Total agents to spawn

# Agent parameters
agent:
  interaction_distance: 1    # Distance to "reach" a product (cells)
  product_dwell_time: 5      # Iterations to stay at each product

# Shop layout (programmatic definition)
# Shelf edges automatically become product locations
shop_layout:
  size:
    width: 50
    height: 30

  # Walls are auto-generated at boundaries

  # Shelves: rectangles or circles that act as obstacles
  # Products are placed at accessible edges of shelves
  shelves:
    - type: rectangle
      x: 5
      y: 8
      width: 8
      height: 2
      products: ["bread", "milk"]     # Products on this shelf's edges
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
    - type: circle
      x: 40
      y: 15
      radius: 3
      products: ["apples", "oranges"]  # Products around circular display

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

### CLI Override Parameters

The following YAML parameters can be overridden via command-line arguments:

- `--agent-count`: Override `simulation.agent_spawn_count`

Design the CLI to be extensible for additional overrides in the future.


## MVP Requirements

1. Single entrance and single exit
2. Agents spawn at entrance with randomized product lists
3. Agents navigate to each product in order, dwell, then proceed
4. Agents exit after completing their shopping list
5. Basic visualization (console output)
6. Configurable via YAML with CLI overrides

## Deferred Features (Post-MVP)

- `_potential_deadlock_detection()`: Detect and resolve gridlock patterns
- Multiple entrances/exits
- Agent priority/urgency levels
- More sophisticated pathfinding (A*)
- Analytics and metrics collection


## Running the Simulation

```bash
# Basic run with config file
python src/main.py --config config.yaml

# Override agent count
python src/main.py --config config.yaml --agent-count 50
```

