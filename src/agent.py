"""Customer agent for the supermarket simulation."""

from enum import Enum
from mesa import Agent

from .pathfinding import bfs_path, manhattan_distance


class AgentState(Enum):
    """States for the customer agent."""
    ENTERING = "entering"
    SHOPPING = "shopping"
    DWELLING = "dwelling"
    EXITING = "exiting"
    REMOVED = "removed"


class CustomerAgent(Agent):
    """
    A customer agent that navigates the supermarket to collect products.

    Attributes:
        state: Current state of the agent
        shopping_list: List of product names to collect
        current_target: Current target position (x, y)
        dwell_counter: Steps remaining at current product
        products_collected: List of collected product names
    """

    def __init__(self, model, shopping_list):
        """
        Initialize a customer agent.

        Args:
            model: The model instance
            shopping_list: List of product names to collect
        """
        super().__init__(model)
        self.state = AgentState.ENTERING
        self.shopping_list = list(shopping_list)
        self.products_collected = []
        self.current_target = None
        self.dwell_counter = 0
        self._cached_path = []  # Cached path to current target
        self._stuck_counter = 0  # Steps stuck at same position
        self._last_pos = None
        self._pathfind_cooldown = 0  # Steps to wait before retrying pathfinding

    def step(self):
        """Execute one step of the agent's behavior."""
        if self.state == AgentState.REMOVED:
            return

        if self.state == AgentState.ENTERING:
            self._handle_entering()
        elif self.state == AgentState.SHOPPING:
            self._handle_shopping()
        elif self.state == AgentState.DWELLING:
            self._handle_dwelling()
        elif self.state == AgentState.EXITING:
            self._handle_exiting()

    def _handle_entering(self):
        """Handle the ENTERING state - transition to SHOPPING."""
        self.state = AgentState.SHOPPING
        self._set_next_target()

    def _handle_shopping(self):
        """Handle the SHOPPING state - navigate to next product."""
        if self.current_target is None:
            # No more products, start exiting
            self.state = AgentState.EXITING
            self._set_exit_target()
            return

        # Check if we're close enough to the product
        if self._is_at_product():
            self.state = AgentState.DWELLING
            self.dwell_counter = self.model.product_dwell_time
            return

        # Move toward target
        self._move_toward_target()

    def _handle_dwelling(self):
        """Handle the DWELLING state - wait at product location."""
        self.dwell_counter -= 1

        if self.dwell_counter <= 0:
            # Collect the product and move on
            if self.shopping_list:
                collected = self.shopping_list.pop(0)
                self.products_collected.append(collected)

            self.state = AgentState.SHOPPING
            self._set_next_target()

    def _handle_exiting(self):
        """Handle the EXITING state - navigate to exit."""
        if self.current_target is None:
            self._set_exit_target()

        # Check if we've reached an exit
        if self.pos in self.model.shop_grid.exit_cells:
            self.state = AgentState.REMOVED
            self.model.remove_agent(self)
            return

        # Move toward exit
        self._move_toward_target()

    def _set_next_target(self):
        """Set the next product location as the target."""
        self._cached_path = []  # Clear cached path when target changes

        if not self.shopping_list:
            self.current_target = None
            return

        next_product = self.shopping_list[0]
        product_pos = self.model.shop_grid.product_positions.get(next_product)

        if product_pos is None:
            # Product not found, skip it
            self.shopping_list.pop(0)
            self._set_next_target()
            return

        # Find an adjacent walkable cell to the product
        self.current_target = self._find_adjacent_target(product_pos)

    def _set_exit_target(self):
        """Set the nearest exit as the target."""
        self._cached_path = []  # Clear cached path when target changes

        if not self.model.shop_grid.exit_cells:
            self.current_target = None
            return

        # Find nearest exit
        nearest_exit = min(
            self.model.shop_grid.exit_cells,
            key=lambda e: manhattan_distance(self.pos, e)
        )
        self.current_target = nearest_exit

    def _find_adjacent_target(self, product_pos):
        """Find a walkable cell adjacent to the product."""
        px, py = product_pos
        neighbors = [(px+1, py), (px-1, py), (px, py+1), (px, py-1)]

        walkable_neighbors = [
            n for n in neighbors
            if n in self.model.shop_grid.walkable_cells
        ]

        if walkable_neighbors:
            # Return nearest walkable neighbor
            return min(
                walkable_neighbors,
                key=lambda n: manhattan_distance(self.pos, n)
            )
        return None

    def _is_at_product(self):
        """Check if agent is adjacent to the current target product."""
        if not self.shopping_list:
            return False

        next_product = self.shopping_list[0]
        product_pos = self.model.shop_grid.product_positions.get(next_product)

        if product_pos is None:
            return False

        distance = manhattan_distance(self.pos, product_pos)
        return distance <= self.model.interaction_distance

    def _move_toward_target(self):
        """Move one step toward the current target."""
        if self.current_target is None:
            return

        if self.pos == self.current_target:
            self._cached_path = []
            self._stuck_counter = 0
            return

        # Decrement pathfinding cooldown
        if self._pathfind_cooldown > 0:
            self._pathfind_cooldown -= 1

        # Check if we need to recalculate path
        need_recalc = False

        if not self._cached_path:
            if self._pathfind_cooldown <= 0:
                need_recalc = True
        else:
            next_pos = self._cached_path[0]
            # Recalculate if next step is blocked
            if not self.model.grid.is_cell_empty(next_pos):
                if self._pathfind_cooldown <= 0:
                    need_recalc = True
                else:
                    self._cached_path = []  # Clear invalid path

        if need_recalc:
            occupied = self.model.get_occupied_cells()
            occupied.discard(self.pos)

            self._cached_path = bfs_path(
                self.pos,
                self.current_target,
                self.model.shop_grid.walkable_cells,
                occupied
            ) or []

            # If no path found, set cooldown to avoid repeated failed searches
            if not self._cached_path:
                self._pathfind_cooldown = 5

        # Try to move along cached path
        moved = False
        if self._cached_path:
            next_pos = self._cached_path[0]
            if self.model.grid.is_cell_empty(next_pos):
                self.model.grid.move_agent(self, next_pos)
                self._cached_path.pop(0)
                moved = True
                self._pathfind_cooldown = 0  # Reset cooldown on successful move

        # Track stuck status and try to wiggle if stuck too long
        if not moved:
            if self.pos == self._last_pos:
                self._stuck_counter += 1
            else:
                self._stuck_counter = 0

            # If stuck for too long, try to move to any adjacent empty cell
            if self._stuck_counter >= 10:
                self._try_wiggle()

        self._last_pos = self.pos

    def _try_wiggle(self):
        """Try to move to any adjacent empty cell to break deadlock."""
        x, y = self.pos
        neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]

        # Filter to walkable and empty cells
        available = [
            n for n in neighbors
            if n in self.model.shop_grid.walkable_cells
            and self.model.grid.is_cell_empty(n)
        ]

        if available:
            # Move to a random available cell
            next_pos = self.model.random.choice(available)
            self.model.grid.move_agent(self, next_pos)
            self._cached_path = []  # Clear cached path after wiggle
            self._stuck_counter = 0

    @property
    def products_remaining(self):
        """Number of products still to collect."""
        return len(self.shopping_list)
