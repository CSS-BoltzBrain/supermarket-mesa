"""ASCII visualization for the supermarket simulation."""

import os
from .grid_builder import CellType
from .agent import AgentState


# Agent display characters based on state
AGENT_CHARS = {
    AgentState.ENTERING: "e",
    AgentState.SHOPPING: "@",
    AgentState.DWELLING: "d",
    AgentState.EXITING: "x",
}


def render_grid(model):
    """
    Render the current state of the model as ASCII.

    Args:
        model: SupermarketModel instance

    Returns:
        String representation of the grid
    """
    grid = model.shop_grid
    lines = []

    # Build a map of agent positions
    agent_positions = {}
    for agent in model.agents:
        if agent.pos is not None:
            agent_positions[agent.pos] = agent

    # Render from top to bottom (high Y to low Y)
    for y in range(grid.height - 1, -1, -1):
        row = []
        for x in range(grid.width):
            pos = (x, y)

            # Check for agent first
            if pos in agent_positions:
                agent = agent_positions[pos]
                char = AGENT_CHARS.get(agent.state, "?")
                row.append(char)
            else:
                # Show cell type
                cell_type = grid.get_cell_type(x, y)
                row.append(cell_type.value)

        lines.append("".join(row))

    return "\n".join(lines)


def render_status(model):
    """
    Render status information about the simulation.

    Args:
        model: SupermarketModel instance

    Returns:
        String with status information
    """
    lines = [
        f"Step: {model.current_step}/{model.max_steps}",
        f"Agents: {len(model.agents)} active, "
        f"{model.agents_spawned} spawned, {model.agents_completed} completed",
    ]

    # Count agents by state
    state_counts = {}
    for agent in model.agents:
        state_name = agent.state.value
        state_counts[state_name] = state_counts.get(state_name, 0) + 1

    if state_counts:
        state_str = ", ".join(f"{k}: {v}" for k, v in sorted(state_counts.items()))
        lines.append(f"States: {state_str}")

    return "\n".join(lines)


def render_legend():
    """
    Render a legend explaining the symbols.

    Returns:
        String with legend information
    """
    lines = [
        "Legend:",
        "  . = Floor    # = Wall     S = Shelf",
        "  P = Product  E = Entrance X = Exit",
        "  @ = Shopping d = Dwelling x = Exiting e = Entering",
    ]
    return "\n".join(lines)


def clear_screen():
    """Clear the terminal screen."""
    os.system("cls" if os.name == "nt" else "clear")


def display(model, show_legend=True):
    """
    Display the current state of the simulation.

    Args:
        model: SupermarketModel instance
        show_legend: Whether to show the legend
    """
    output = []

    output.append(render_status(model))
    output.append("")
    output.append(render_grid(model))

    if show_legend:
        output.append("")
        output.append(render_legend())

    print("\n".join(output))
