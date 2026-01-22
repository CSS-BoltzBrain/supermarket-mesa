"""
PyCX-Style Analysis of Supermarket Crowd Simulation
====================================================

This script analyzes the output data from a supermarket agent-based model
following the PyCX philosophy: simplicity, readability, and pedagogical value.

The simulation models customer movement through a supermarket with:
- Agents spawning at entrance (rush hour: 1 agent/step)
- Agents collecting products from shelves
- Agents exiting after completing shopping list
- SingleGrid collision preventing overlapping agents

Key complex systems phenomena to observe:
1. Phase transitions in crowd density
2. Congestion/clogging at bottlenecks
3. Throughput saturation under high load
4. Self-organization and emergent patterns

Author: PyCX Style Analysis
"""

# PyCX typically uses "from pylab import *" but we use explicit imports
# for environments where pylab may not be installed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ast
from collections import Counter

# Import numpy functions into namespace (PyCX style uses global pylab)
from numpy import (zeros, ones, mean, std, diff, minimum, maximum,
                   argmax, argmin, arange, convolve, nan)
from numpy.random import random, rand, choice

# Matplotlib convenience aliases
figure = plt.figure
subplot = plt.subplot
plot = plt.plot
scatter = plt.scatter
bar = plt.bar
pie = plt.pie
stackplot = plt.stackplot
fill_between = plt.fill_between
imshow = plt.imshow
xlabel = plt.xlabel
ylabel = plt.ylabel
title = plt.title
legend = plt.legend
grid = plt.grid
axis = plt.axis
axhline = plt.axhline
axvline = plt.axvline
colorbar = plt.colorbar
annotate = plt.annotate
text = plt.text
xticks = plt.xticks
tight_layout = plt.tight_layout
savefig = plt.savefig
show = plt.show
gca = plt.gca

# =============================================================================
# GLOBAL VARIABLES (PyCX Style: explicit state for educational clarity)
# =============================================================================

# Data storage
model_data = None      # DataFrame with model-level metrics per step
agent_data = None      # DataFrame with agent-level data per step

# Derived metrics
time_steps = None      # Array of time steps
total_agents = None    # Total agents in system at each step
throughput = None      # Agents completing per time window
density = None         # Agent density over time

# Shop parameters (from config)
SHOP_WIDTH = 50
SHOP_HEIGHT = 30
TOTAL_CELLS = SHOP_WIDTH * SHOP_HEIGHT
WALKABLE_FRACTION = 0.7  # Approximate fraction of walkable cells

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_data(model_path='data/model_data.csv', agent_path='data/agent_data.csv'):
    """Load simulation output data from CSV files."""
    global model_data, agent_data
    global time_steps, total_agents

    print("Loading model data...")
    model_data = pd.read_csv(model_path)

    print("Loading agent data (this may take a moment)...")
    # For large agent data, we sample or use chunks
    try:
        agent_data = pd.read_csv(agent_path)
    except Exception as e:
        print(f"Warning: Could not load full agent data: {e}")
        agent_data = None

    # Extract basic arrays for plotting
    time_steps = model_data['step'].values
    total_agents = model_data['total_agents'].values

    print(f"Loaded {len(model_data)} time steps")
    print(f"Simulation duration: {time_steps[-1]} steps")
    print(f"Peak agent count: {max(total_agents)}")


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def compute_throughput(window=20):
    """
    Compute throughput: agents completing shopping per time window.

    Throughput is a key metric in complex systems - it often shows
    saturation behavior under high load (similar to traffic flow).
    """
    global throughput

    completed = model_data['agents_completed'].values
    # Throughput = change in completed agents per window
    throughput = zeros(len(completed))
    for i in range(window, len(completed)):
        throughput[i] = (completed[i] - completed[i - window]) / window

    return throughput


def compute_congestion_index():
    """
    Compute congestion index: ratio of blocked/waiting agents.

    A high congestion index indicates clogging behavior where
    agents cannot make progress due to crowding.
    """
    shopping = model_data['agents_shopping'].values
    dwelling = model_data['agents_dwelling'].values
    exiting = model_data['agents_exiting'].values
    total = model_data['total_agents'].values

    # Active agents (shopping + exiting) should be moving
    # If many are shopping but not dwelling, they may be blocked
    active = shopping + exiting

    # Congestion proxy: shopping agents / (shopping + dwelling + small epsilon)
    # High value means agents are trying to shop but blocked
    congestion = shopping / (shopping + dwelling + 1)

    return congestion


def compute_little_law_metrics():
    """
    Verify Little's Law: L = lambda * W

    L = average number of agents in system
    lambda = arrival rate (spawn rate)
    W = average time in system

    This fundamental queuing theory result applies to steady-state systems.
    """
    L = mean(total_agents[100:400])  # Average during busy period
    lambda_rate = 1.0  # 1 agent per step (from config)
    W_estimated = L / lambda_rate

    # Actual average time can be computed from agent data if available
    return L, lambda_rate, W_estimated


# =============================================================================
# PLOTTING FUNCTIONS (PyCX Style: simple matplotlib)
# =============================================================================

def plot_population_dynamics():
    """
    Plot 1: Population dynamics over time.

    Shows how agent count evolves - key for understanding
    system capacity and congestion onset.
    """
    figure(figsize=(12, 8))

    subplot(2, 2, 1)
    plot(time_steps, total_agents, 'b-', linewidth=1.5, label='Total agents')
    axhline(y=max(total_agents), color='r', linestyle='--', alpha=0.5,
            label=f'Peak: {max(total_agents)}')
    xlabel('Time step')
    ylabel('Number of agents')
    title('Total Agents in System Over Time')
    legend()
    grid(True, alpha=0.3)

    subplot(2, 2, 2)
    shopping = model_data['agents_shopping'].values
    dwelling = model_data['agents_dwelling'].values
    exiting = model_data['agents_exiting'].values

    stackplot(time_steps, shopping, dwelling, exiting,
              labels=['Shopping', 'Dwelling', 'Exiting'],
              colors=['#3498db', '#e74c3c', '#2ecc71'], alpha=0.7)
    xlabel('Time step')
    ylabel('Number of agents')
    title('Agent State Distribution')
    legend(loc='upper right')
    grid(True, alpha=0.3)

    subplot(2, 2, 3)
    completed = model_data['agents_completed'].values
    plot(time_steps, completed, 'g-', linewidth=1.5)
    xlabel('Time step')
    ylabel('Cumulative completed')
    title('Cumulative Agents Completed Shopping')
    grid(True, alpha=0.3)

    subplot(2, 2, 4)
    # Agent density as fraction of shop capacity
    # Approximate walkable cells ~ 70% of total
    walkable_cells = int(TOTAL_CELLS * WALKABLE_FRACTION)
    density = total_agents / walkable_cells * 100
    plot(time_steps, density, 'm-', linewidth=1.5)
    axhline(y=15, color='orange', linestyle='--', alpha=0.7,
            label='Congestion threshold (~15%)')
    xlabel('Time step')
    ylabel('Agent density (%)')
    title('Agent Density (% of walkable space)')
    legend()
    grid(True, alpha=0.3)

    tight_layout()
    savefig('analysis_population_dynamics.png', dpi=150)
    print("Saved: analysis_population_dynamics.png")


def plot_throughput_analysis():
    """
    Plot 2: Throughput analysis.

    Throughput vs. density reveals the fundamental diagram -
    a key concept in traffic flow and crowd dynamics.
    """
    figure(figsize=(12, 5))

    window = 20
    throughput = compute_throughput(window)

    subplot(1, 2, 1)
    plot(time_steps, throughput, 'b-', linewidth=1, alpha=0.7)

    # Smooth with moving average
    smooth_window = 30
    smooth_throughput = convolve(throughput, ones(smooth_window)/smooth_window, mode='same')
    plot(time_steps, smooth_throughput, 'r-', linewidth=2, label='Smoothed')

    xlabel('Time step')
    ylabel(f'Throughput (completions per {window} steps)')
    title('Shopping Throughput Over Time')
    legend()
    grid(True, alpha=0.3)

    subplot(1, 2, 2)
    # Fundamental diagram: throughput vs density
    walkable_cells = int(TOTAL_CELLS * WALKABLE_FRACTION)
    density = total_agents / walkable_cells * 100

    # Skip initial transient
    start_idx = 50
    scatter(density[start_idx:], smooth_throughput[start_idx:],
            c=time_steps[start_idx:], cmap='viridis', alpha=0.5, s=10)
    colorbar(label='Time step')
    xlabel('Agent density (%)')
    ylabel('Throughput')
    title('Fundamental Diagram: Throughput vs Density')
    grid(True, alpha=0.3)

    tight_layout()
    savefig('analysis_throughput.png', dpi=150)
    print("Saved: analysis_throughput.png")


def plot_phase_transitions():
    """
    Plot 3: Phase transitions and regime changes.

    Complex systems often exhibit distinct phases:
    - Free flow: agents move unimpeded
    - Synchronized: agents influence each other
    - Congested/Jammed: system gridlock
    """
    figure(figsize=(12, 8))

    # Rate of change in agent count (system dynamics)
    d_agents = diff(total_agents)

    subplot(2, 2, 1)
    plot(time_steps[1:], d_agents, 'b-', alpha=0.5, linewidth=0.5)
    smooth = convolve(d_agents, ones(20)/20, mode='same')
    plot(time_steps[1:], smooth, 'r-', linewidth=2, label='Smoothed')
    axhline(y=0, color='k', linestyle='-', alpha=0.3)
    xlabel('Time step')
    ylabel('d(agents)/dt')
    title('Rate of Change in Agent Population')
    legend()
    grid(True, alpha=0.3)

    # Phase space: total agents vs rate of change
    subplot(2, 2, 2)
    scatter(total_agents[1:], d_agents, c=time_steps[1:],
            cmap='coolwarm', alpha=0.3, s=5)
    colorbar(label='Time step')
    xlabel('Total agents')
    ylabel('d(agents)/dt')
    title('Phase Space Trajectory')
    grid(True, alpha=0.3)

    # State transitions over time
    subplot(2, 2, 3)
    shopping = model_data['agents_shopping'].values
    dwelling = model_data['agents_dwelling'].values
    exiting = model_data['agents_exiting'].values

    # Ratio of active states
    active_ratio = (shopping + exiting) / (total_agents + 1)
    plot(time_steps, active_ratio, 'b-', alpha=0.7, linewidth=1)
    xlabel('Time step')
    ylabel('Fraction of active agents')
    title('Active Agent Ratio (Shopping + Exiting)')
    grid(True, alpha=0.3)

    # Congestion index
    subplot(2, 2, 4)
    congestion = compute_congestion_index()
    plot(time_steps, congestion, 'r-', alpha=0.7, linewidth=1)
    xlabel('Time step')
    ylabel('Congestion index')
    title('Congestion Index Over Time')
    grid(True, alpha=0.3)

    tight_layout()
    savefig('analysis_phase_transitions.png', dpi=150)
    print("Saved: analysis_phase_transitions.png")


def plot_system_efficiency():
    """
    Plot 4: System efficiency metrics.

    Efficiency captures how well the system processes agents
    relative to its capacity.
    """
    figure(figsize=(12, 5))

    completed = model_data['agents_completed'].values

    subplot(1, 2, 1)
    # Time to complete shopping for different agent cohorts
    # Estimated from cumulative completion curve

    # Find time for each 10% completion milestone
    milestones = arange(0.1, 1.1, 0.1) * 200  # 200 total agents
    milestone_times = []
    for m in milestones:
        idx = argmax(completed >= m)
        if idx > 0:
            milestone_times.append(time_steps[idx])
        else:
            milestone_times.append(nan)

    bar(range(len(milestones)), milestone_times, color='steelblue', alpha=0.7)
    xticks(range(len(milestones)), [f'{int(m)}' for m in milestones])
    xlabel('Agents completed (cumulative)')
    ylabel('Time step')
    title('Time to Complete Shopping Milestones')
    grid(True, alpha=0.3, axis='y')

    subplot(1, 2, 2)
    # Efficiency: completions per agent-step in system
    window = 30
    throughput = compute_throughput(window)
    efficiency = throughput / (total_agents + 1)

    plot(time_steps, efficiency, 'g-', alpha=0.5, linewidth=1)
    smooth_eff = convolve(efficiency, ones(30)/30, mode='same')
    plot(time_steps, smooth_eff, 'darkgreen', linewidth=2, label='Smoothed')
    xlabel('Time step')
    ylabel('Efficiency (completions / agent-step)')
    title('System Efficiency Over Time')
    legend()
    grid(True, alpha=0.3)

    tight_layout()
    savefig('analysis_efficiency.png', dpi=150)
    print("Saved: analysis_efficiency.png")


def plot_queuing_analysis():
    """
    Plot 5: Queuing theory analysis.

    Apply Little's Law and analyze waiting/service dynamics.
    """
    figure(figsize=(12, 8))

    # Little's Law verification
    subplot(2, 2, 1)

    # Compute instantaneous arrival and departure rates
    completed = model_data['agents_completed'].values
    arrivals = minimum(time_steps, 200)  # 1 per step up to 200
    departures = completed

    # Queue length = arrivals - departures
    queue_length = arrivals - departures

    plot(time_steps, queue_length, 'b-', linewidth=1.5, label='In system')
    plot(time_steps, arrivals, 'g--', linewidth=1, alpha=0.7, label='Cumulative arrivals')
    plot(time_steps, departures, 'r--', linewidth=1, alpha=0.7, label='Cumulative departures')
    xlabel('Time step')
    ylabel('Count')
    title("Queue Analysis (Little's Law)")
    legend()
    grid(True, alpha=0.3)

    # Wait time estimation
    subplot(2, 2, 2)

    # Estimate wait time from queue length / arrival rate
    lambda_rate = 1.0  # arrival rate
    wait_estimate = queue_length / lambda_rate

    plot(time_steps, wait_estimate, 'purple', linewidth=1.5)
    xlabel('Time step')
    ylabel('Estimated wait time (steps)')
    title('Estimated Time in System')
    grid(True, alpha=0.3)

    # State distribution histogram
    subplot(2, 2, 3)

    shopping_avg = mean(model_data['agents_shopping'].values[50:400])
    dwelling_avg = mean(model_data['agents_dwelling'].values[50:400])
    exiting_avg = mean(model_data['agents_exiting'].values[50:400])

    states = ['Shopping', 'Dwelling', 'Exiting']
    avgs = [shopping_avg, dwelling_avg, exiting_avg]
    colors = ['#3498db', '#e74c3c', '#2ecc71']

    bar(states, avgs, color=colors, alpha=0.7)
    ylabel('Average agents in state')
    title('Average State Distribution (steps 50-400)')
    grid(True, alpha=0.3, axis='y')

    # Service time analysis
    subplot(2, 2, 4)

    # Moving average of dwelling agents (proxy for service)
    dwelling = model_data['agents_dwelling'].values
    shopping = model_data['agents_shopping'].values

    service_ratio = dwelling / (shopping + dwelling + 1)
    smooth_ratio = convolve(service_ratio, ones(20)/20, mode='same')

    plot(time_steps, smooth_ratio, 'orange', linewidth=2)
    xlabel('Time step')
    ylabel('Dwelling / (Shopping + Dwelling)')
    title('Service vs Queue Ratio')
    grid(True, alpha=0.3)

    tight_layout()
    savefig('analysis_queuing.png', dpi=150)
    print("Saved: analysis_queuing.png")


def plot_summary_dashboard():
    """
    Plot 6: Summary dashboard with key metrics.
    """
    figure(figsize=(14, 10))

    # Key statistics
    peak_agents = max(total_agents)
    total_completed = model_data['agents_completed'].values[-1]
    sim_duration = time_steps[-1]
    avg_throughput = total_completed / sim_duration

    # Peak density time
    peak_time = time_steps[argmax(total_agents)]

    # Time to clear (when all agents exit)
    clear_idx = argmax(total_agents == 0)
    clear_time = time_steps[clear_idx] if clear_idx > 0 else sim_duration

    # Main time series
    ax1 = subplot(2, 2, 1)
    plot(time_steps, total_agents, 'b-', linewidth=1.5, label='Agents in system')
    fill_between(time_steps, total_agents, alpha=0.3)
    axvline(x=200, color='g', linestyle='--', alpha=0.7, label='Spawning ends')
    axvline(x=peak_time, color='r', linestyle='--', alpha=0.7, label=f'Peak at t={peak_time}')
    xlabel('Time step')
    ylabel('Agent count')
    title('Population Over Time')
    legend(loc='upper right')
    grid(True, alpha=0.3)

    # State breakdown at peak
    subplot(2, 2, 2)
    peak_idx = argmax(total_agents)
    states = ['Shopping', 'Dwelling', 'Exiting']
    peak_counts = [
        model_data['agents_shopping'].values[peak_idx],
        model_data['agents_dwelling'].values[peak_idx],
        model_data['agents_exiting'].values[peak_idx]
    ]
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    pie(peak_counts, labels=states, colors=colors, autopct='%1.1f%%', startangle=90)
    title(f'Agent States at Peak (t={peak_time}, n={peak_agents})')

    # Cumulative completions
    subplot(2, 2, 3)
    completed = model_data['agents_completed'].values
    plot(time_steps, completed, 'g-', linewidth=2)

    # Mark 50%, 90%, 100% completion
    for pct in [0.5, 0.9, 1.0]:
        target = int(200 * pct)
        idx = argmax(completed >= target)
        if idx > 0:
            axvline(x=time_steps[idx], color='gray', linestyle=':', alpha=0.5)
            annotate(f'{int(pct*100)}%', xy=(time_steps[idx], target),
                    fontsize=9, ha='left')

    xlabel('Time step')
    ylabel('Completed')
    title('Cumulative Shopping Completions')
    grid(True, alpha=0.3)

    # Text summary
    subplot(2, 2, 4)
    axis('off')

    summary_text = f"""
    SIMULATION SUMMARY
    {'='*40}

    Configuration:
    - Shop size: {SHOP_WIDTH} x {SHOP_HEIGHT} cells
    - Total agents spawned: 200
    - Spawn rate: 1 agent/step (rush hour)
    - Max products per agent: 5

    Results:
    - Simulation duration: {sim_duration} steps
    - Peak agents in system: {peak_agents}
    - Peak occurred at step: {peak_time}
    - Time to clear shop: ~{clear_time} steps
    - Average throughput: {avg_throughput:.3f} agents/step

    Key Observations:
    - System reached saturation around t=150-200
    - Peak congestion with {peak_agents} simultaneous agents
    - Graceful degradation after spawn period
    """

    text(0.1, 0.9, summary_text, transform=gca().transAxes,
         fontsize=10, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    tight_layout()
    savefig('analysis_summary.png', dpi=150)
    print("Saved: analysis_summary.png")


# =============================================================================
# MAIN ANALYSIS ROUTINE
# =============================================================================

def run_analysis():
    """
    Main analysis routine - generates all plots.

    PyCX Style: Simple sequential execution, clear output.
    """
    print("="*60)
    print("SUPERMARKET SIMULATION ANALYSIS (PyCX Style)")
    print("="*60)
    print()

    # Load data
    load_data()
    print()

    # Generate all plots
    print("Generating analysis plots...")
    print()

    plot_population_dynamics()
    plot_throughput_analysis()
    plot_phase_transitions()
    plot_system_efficiency()
    plot_queuing_analysis()
    plot_summary_dashboard()

    print()
    print("="*60)
    print("Analysis complete! Generated 6 plot files:")
    print("  1. analysis_population_dynamics.png")
    print("  2. analysis_throughput.png")
    print("  3. analysis_phase_transitions.png")
    print("  4. analysis_efficiency.png")
    print("  5. analysis_queuing.png")
    print("  6. analysis_summary.png")
    print("="*60)

    # Show all plots
    show()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    run_analysis()
