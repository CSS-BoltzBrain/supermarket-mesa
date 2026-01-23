"""Supermarket model for the agent-based simulation."""

import random
from mesa import Model, DataCollector
from mesa.space import SingleGrid

try:
    from .agent import CustomerAgent, AgentState
    from .grid_builder import build_grid
except ImportError:
    from agent import CustomerAgent, AgentState
    from grid_builder import build_grid


class SupermarketModel(Model):
    """
    The supermarket simulation model.

    Manages the grid, agents, and simulation logic.
    """

    def __init__(self, config):
        """
        Initialize the supermarket model.

        Args:
            config: Configuration dictionary loaded from YAML
        """
        super().__init__()

        # Store configuration
        self.config = config
        sim_config = config["simulation"]
        agent_config = config["agent"]

        # Simulation parameters
        self.max_steps = sim_config["max_steps"]
        self.agent_spawn_rate = sim_config["agent_spawn_rate"]
        self.agent_spawn_count = sim_config["agent_spawn_count"]

        # Agent parameters
        self.interaction_distance = agent_config["interaction_distance"]
        self.product_dwell_time = agent_config["product_dwell_time"]
        self.max_products_per_agent = agent_config["max_products_per_agent"]

        # Set random seed if specified
        seed = sim_config.get("random_seed")
        if seed is not None:
            random.seed(seed)
            self.random.seed(seed)

        # Build the shop grid
        self.shop_grid = build_grid(config)

        # Create Mesa grid (SingleGrid: one agent per cell)
        self.grid = SingleGrid(
            self.shop_grid.width,
            self.shop_grid.height,
            torus=False
        )

        # Tracking variables
        self.agents_spawned = 0
        self.agents_completed = 0
        self.current_step = 0

        # Stuck detection
        self._last_completed = 0
        self._stuck_steps = 0
        self._stuck_threshold = sim_config.get("stuck_threshold", 100)  # Steps without progress
        self.is_stuck = False

        # Get available product names
        self.available_products = list(self.shop_grid.product_positions.keys())

        # Setup data collector
        self.datacollector = DataCollector(
            model_reporters={
                "step": lambda m: m.current_step,
                "total_agents": lambda m: len(m.agents),
                "agents_shopping": lambda m: m._count_agents_in_state(AgentState.SHOPPING),
                "agents_dwelling": lambda m: m._count_agents_in_state(AgentState.DWELLING),
                "agents_exiting": lambda m: m._count_agents_in_state(AgentState.EXITING),
                "agents_completed": lambda m: m.agents_completed,
            },
            agent_reporters={
                "agent_id": lambda a: a.unique_id,
                "position": lambda a: a.pos,
                "state": lambda a: a.state.value,
                "products_remaining": lambda a: a.products_remaining,
            }
        )

    def step(self):
        """Execute one step of the simulation."""
        self.current_step += 1

        # Spawn new agents if needed
        self._maybe_spawn_agent()

        # Step all agents in random order
        agents_to_step = list(self.agents)
        self.random.shuffle(agents_to_step)
        for agent in agents_to_step:
            agent.step()

        # Check for stuck condition
        self._check_stuck()

        # Collect data
        self.datacollector.collect(self)

    def _check_stuck(self):
        """Check if simulation is stuck (no progress for too long)."""
        if self.agents_completed > self._last_completed:
            self._last_completed = self.agents_completed
            self._stuck_steps = 0
        elif len(self.agents) > 0:
            self._stuck_steps += 1

        if self._stuck_steps >= self._stuck_threshold:
            self.is_stuck = True

    def _maybe_spawn_agent(self):
        """Spawn a new agent if conditions are met."""
        if self.agents_spawned >= self.agent_spawn_count:
            return

        if self.current_step % self.agent_spawn_rate != 0:
            return

        # Find a free entrance cell
        spawn_pos = self._find_free_entrance()
        if spawn_pos is None:
            return  # All entrances blocked

        # Generate random shopping list
        shopping_list = self._generate_shopping_list()

        # Create and place agent
        agent = CustomerAgent(self, shopping_list)
        self.agents_spawned += 1

        self.grid.place_agent(agent, spawn_pos)

    def _find_free_entrance(self):
        """Find an unoccupied entrance cell."""
        for pos in self.shop_grid.entrance_cells:
            if self.grid.is_cell_empty(pos):
                return pos
        return None

    def _generate_shopping_list(self):
        """Generate a random shopping list for an agent."""
        if not self.available_products:
            return []

        num_products = random.randint(1, min(self.max_products_per_agent, len(self.available_products)))
        return random.sample(self.available_products, num_products)

    def remove_agent(self, agent):
        """Remove an agent from the simulation."""
        self.grid.remove_agent(agent)
        agent.remove()
        self.agents_completed += 1

    def get_occupied_cells(self):
        """Get set of cells currently occupied by agents."""
        occupied = set()
        for agent in self.agents:
            if agent.pos is not None:
                occupied.add(agent.pos)
        return occupied

    def _count_agents_in_state(self, state):
        """Count agents in a given state."""
        return sum(1 for agent in self.agents if agent.state == state)

    def is_running(self):
        """Check if simulation should continue running."""
        if self.current_step >= self.max_steps:
            return False

        if self.is_stuck:
            return False

        # Continue if there are agents or more to spawn
        if len(self.agents) > 0:
            return True
        if self.agents_spawned < self.agent_spawn_count:
            return True

        return False

    def export_data(self, model_file="model_data.csv", agent_file="agent_data.csv"):
        """Export collected data to CSV files."""
        model_df = self.datacollector.get_model_vars_dataframe()
        agent_df = self.datacollector.get_agent_vars_dataframe()

        model_df.to_csv(model_file)
        agent_df.to_csv(agent_file)

        return model_file, agent_file
