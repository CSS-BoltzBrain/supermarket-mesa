"""CLI entry point for the supermarket simulation."""

import argparse
import time
import yaml

from .model import SupermarketModel
from .visualization import display, clear_screen


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def apply_overrides(config, args):
    """Apply command-line overrides to configuration."""
    if args.agent_count is not None:
        config["simulation"]["agent_spawn_count"] = args.agent_count

    if args.seed is not None:
        config["simulation"]["random_seed"] = args.seed

    return config


def run_simulation(config, visual=True, delay=0.1):
    """
    Run the simulation.

    Args:
        config: Configuration dictionary
        visual: Whether to show ASCII visualization
        delay: Delay between steps in seconds (for visualization)

    Returns:
        The completed model
    """
    model = SupermarketModel(config)

    if visual:
        clear_screen()
        display(model)
        time.sleep(delay)

    while model.is_running():
        model.step()

        if visual:
            clear_screen()
            display(model)
            time.sleep(delay)

    return model


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Supermarket crowd simulation using Mesa"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--agent-count",
        type=int,
        default=None,
        help="Override number of agents to spawn"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--no-visual",
        action="store_true",
        help="Disable ASCII visualization"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.1,
        help="Delay between steps in seconds (default: 0.1)"
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export data to CSV files after simulation"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory for output files (default: current directory)"
    )

    args = parser.parse_args()

    # Load and configure
    config = load_config(args.config)
    config = apply_overrides(config, args)

    # Run simulation
    visual = not args.no_visual
    model = run_simulation(config, visual=visual, delay=args.delay)

    # Final status
    print(f"\nSimulation complete!")
    print(f"  Total steps: {model.current_step}")
    print(f"  Agents spawned: {model.agents_spawned}")
    print(f"  Agents completed: {model.agents_completed}")

    # Export data if requested
    if args.export:
        import os
        model_file = os.path.join(args.output_dir, "model_data.csv")
        agent_file = os.path.join(args.output_dir, "agent_data.csv")
        model.export_data(model_file, agent_file)
        print(f"\nData exported to:")
        print(f"  {model_file}")
        print(f"  {agent_file}")


if __name__ == "__main__":
    main()
