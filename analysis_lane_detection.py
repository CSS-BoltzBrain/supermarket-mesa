"""
PyCX-Style Lane Formation & Self-Organized Criticality Analysis
================================================================

Analyzes spatial patterns in supermarket simulation to detect:
1. Self-organized lane formation (two-way traffic in aisles)
2. Self-organized criticality (power-law distributions, avalanches)
3. Emergent spatial order from simple rules

Key hypothesis: When shoppers enter from bottom-left and exit top-right,
counter-flowing agents may self-organize into lanes to minimize collisions.

Author: PyCX Style Analysis
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ast
from collections import Counter, defaultdict
from scipy import stats
from scipy.ndimage import gaussian_filter

# PyCX-style imports
from numpy import (zeros, ones, mean, std, diff, minimum, maximum,
                   argmax, argmin, arange, histogram, histogram2d,
                   sqrt, log, exp, pi, nan, inf, array, linspace)
from numpy.random import random, rand, choice

# Matplotlib aliases
figure = plt.figure
subplot = plt.subplot
plot = plt.plot
scatter = plt.scatter
imshow = plt.imshow
contour = plt.contourf
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
tight_layout = plt.tight_layout
savefig = plt.savefig
show = plt.show
gca = plt.gca

# =============================================================================
# GLOBAL VARIABLES
# =============================================================================

# Data storage
agent_data = None
model_data = None
positions = None           # Dict: step -> list of (x, y) tuples
velocities = None          # Dict: agent_id -> list of velocity vectors
agent_trajectories = None  # Dict: agent_id -> list of positions

# Shop geometry
SHOP_WIDTH = 50
SHOP_HEIGHT = 30
ENTRANCE = (1, 1)      # Bottom-left
EXIT = (44, 28)        # Top-right

# Aisle regions (between shelves - key areas for lane formation)
# From config: shelves at y=5,9,15,21,25 (height 2 each)
AISLE_Y_RANGES = [
    (1, 5),    # Bottom corridor
    (7, 9),    # Aisle between shelf rows
    (11, 15),  # Middle aisle
    (17, 21),  # Upper middle aisle
    (23, 25),  # Upper aisle
    (27, 29),  # Top corridor
]

# =============================================================================
# DATA LOADING
# =============================================================================

def parse_position(pos_str):
    """Parse position string '(x, y)' to tuple."""
    try:
        return ast.literal_eval(pos_str)
    except:
        return None


def load_data(agent_path='data/agent_data.csv', model_path='data/model_data.csv'):
    """Load and parse simulation data."""
    global agent_data, model_data, positions, velocities, agent_trajectories

    print("Loading agent data...")
    agent_data = pd.read_csv(agent_path)
    model_data = pd.read_csv(model_path)

    # Parse positions
    print("Parsing positions...")
    agent_data['pos_tuple'] = agent_data['position'].apply(parse_position)
    agent_data = agent_data.dropna(subset=['pos_tuple'])

    # Build position dict by step
    positions = defaultdict(list)
    for _, row in agent_data.iterrows():
        step = int(row['Step'])
        pos = row['pos_tuple']
        positions[step].append(pos)

    # Build agent trajectories
    agent_trajectories = defaultdict(list)
    for _, row in agent_data.sort_values('Step').iterrows():
        aid = row['agent_id']
        pos = row['pos_tuple']
        agent_trajectories[aid].append(pos)

    # Compute velocities (displacement per step)
    velocities = {}
    for aid, traj in agent_trajectories.items():
        vels = []
        for i in range(1, len(traj)):
            dx = traj[i][0] - traj[i-1][0]
            dy = traj[i][1] - traj[i-1][1]
            vels.append((dx, dy))
        velocities[aid] = vels

    print(f"Loaded {len(agent_data)} position records")
    print(f"Time steps: {min(positions.keys())} to {max(positions.keys())}")
    print(f"Unique agents: {len(agent_trajectories)}")


# =============================================================================
# LANE DETECTION METRICS
# =============================================================================

def compute_flow_field(step_range=None):
    """
    Compute average velocity field across the shop floor.

    Returns 2D arrays of mean vx, vy at each cell.
    """
    vx_sum = zeros((SHOP_HEIGHT, SHOP_WIDTH))
    vy_sum = zeros((SHOP_HEIGHT, SHOP_WIDTH))
    count = zeros((SHOP_HEIGHT, SHOP_WIDTH))

    if step_range is None:
        step_range = range(50, 400)  # Skip transient, use congested period

    for _, row in agent_data.iterrows():
        step = int(row['Step'])
        if step not in step_range:
            continue

        aid = row['agent_id']
        pos = row['pos_tuple']
        if pos is None:
            continue

        x, y = int(pos[0]), int(pos[1])
        if 0 <= x < SHOP_WIDTH and 0 <= y < SHOP_HEIGHT:
            # Find velocity at this step for this agent
            traj = agent_trajectories.get(aid, [])
            traj_idx = None
            for i, p in enumerate(traj):
                if p == pos:
                    traj_idx = i
                    break

            if traj_idx is not None and traj_idx < len(velocities.get(aid, [])):
                vx, vy = velocities[aid][traj_idx]
                vx_sum[y, x] += vx
                vy_sum[y, x] += vy
                count[y, x] += 1

    # Avoid division by zero
    count[count == 0] = 1
    vx_mean = vx_sum / count
    vy_mean = vy_sum / count

    return vx_mean, vy_mean, count


def compute_lane_order_parameter(y_range, step_range=None):
    """
    Compute lane order parameter for a specific aisle (y-range).

    Lane formation = bimodal distribution of y-coordinates within aisle.
    Order parameter: 0 = random mixing, 1 = perfect two lanes.

    Method: Check if agents moving in opposite directions occupy
    different y-positions within the aisle.
    """
    if step_range is None:
        step_range = range(100, 400)

    # Collect (y, vy) pairs for agents in this aisle
    y_positions_up = []    # Agents moving up (vy > 0)
    y_positions_down = []  # Agents moving down (vy < 0)
    y_positions_right = [] # Agents moving right (vx > 0)
    y_positions_left = []  # Agents moving left (vx < 0)

    for aid, traj in agent_trajectories.items():
        vels = velocities.get(aid, [])
        for i, pos in enumerate(traj):
            if pos is None:
                continue
            x, y = pos
            if y_range[0] <= y <= y_range[1] and i < len(vels):
                vx, vy = vels[i]
                # Horizontal flow classification (main flow direction)
                if vx > 0:
                    y_positions_right.append(y)
                elif vx < 0:
                    y_positions_left.append(y)

    if len(y_positions_right) < 10 or len(y_positions_left) < 10:
        return nan  # Not enough data

    # Lane order = separation of means normalized by aisle width
    mean_right = mean(y_positions_right)
    mean_left = mean(y_positions_left)
    aisle_width = y_range[1] - y_range[0]

    if aisle_width == 0:
        return nan

    # Order parameter: |mean_right - mean_left| / aisle_width
    # 0 = same position, 1 = maximally separated
    order = abs(mean_right - mean_left) / aisle_width

    return min(order, 1.0)  # Cap at 1


def compute_bimodality_coefficient(y_data):
    """
    Compute Sarle's bimodality coefficient.

    b = (skewness^2 + 1) / kurtosis

    b > 0.555 suggests bimodality (two lanes).
    """
    if len(y_data) < 10:
        return nan

    n = len(y_data)
    skew = stats.skew(y_data)
    kurt = stats.kurtosis(y_data, fisher=False)  # Pearson kurtosis

    if kurt == 0:
        return nan

    b = (skew**2 + 1) / kurt
    return b


def detect_lane_formation_over_time(aisle_y_range, window=20):
    """
    Track lane formation over time for a specific aisle.

    Returns time series of lane order parameter.
    """
    steps = sorted(positions.keys())
    order_time = []
    step_times = []

    for start in range(0, len(steps) - window, window // 2):
        step_window = steps[start:start + window]
        order = compute_lane_order_parameter(aisle_y_range, step_window)
        order_time.append(order)
        step_times.append(mean(step_window))

    return array(step_times), array(order_time)


# =============================================================================
# SELF-ORGANIZED CRITICALITY METRICS
# =============================================================================

def compute_waiting_avalanches():
    """
    Detect avalanches: sequences of steps where agent velocity is zero.

    In SOC systems, avalanche size distribution follows power law.
    """
    avalanche_sizes = []

    for aid, vels in velocities.items():
        if len(vels) == 0:
            continue

        # Find runs of zero velocity (waiting/blocked)
        in_avalanche = False
        avalanche_size = 0

        for vx, vy in vels:
            is_blocked = (vx == 0 and vy == 0)

            if is_blocked:
                if not in_avalanche:
                    in_avalanche = True
                    avalanche_size = 1
                else:
                    avalanche_size += 1
            else:
                if in_avalanche:
                    avalanche_sizes.append(avalanche_size)
                    in_avalanche = False
                    avalanche_size = 0

        # End of trajectory
        if in_avalanche and avalanche_size > 0:
            avalanche_sizes.append(avalanche_size)

    return array(avalanche_sizes)


def compute_congestion_clusters(step, distance_threshold=2):
    """
    Find clusters of blocked/slow-moving agents at a given step.

    Cluster size distribution may show SOC behavior.
    """
    pos_list = positions.get(step, [])
    if len(pos_list) < 2:
        return []

    # Simple clustering: connected components within distance threshold
    n = len(pos_list)
    visited = [False] * n
    clusters = []

    def bfs(start):
        cluster = [start]
        queue = [start]
        visited[start] = True

        while queue:
            curr = queue.pop(0)
            cx, cy = pos_list[curr]
            for j in range(n):
                if not visited[j]:
                    jx, jy = pos_list[j]
                    dist = abs(cx - jx) + abs(cy - jy)  # Manhattan distance
                    if dist <= distance_threshold:
                        visited[j] = True
                        queue.append(j)
                        cluster.append(j)

        return len(cluster)

    for i in range(n):
        if not visited[i]:
            size = bfs(i)
            clusters.append(size)

    return clusters


def fit_power_law(data, xmin=1):
    """
    Fit power law to data using MLE.

    P(x) ~ x^(-alpha)

    Returns alpha and goodness of fit.
    """
    data = array(data)
    data = data[data >= xmin]

    if len(data) < 10:
        return nan, nan

    # MLE estimator for power law exponent
    alpha = 1 + len(data) / sum(log(data / (xmin - 0.5)))

    # Kolmogorov-Smirnov test for goodness of fit
    # (simplified version)
    return alpha, len(data)


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_spatial_density_map():
    """
    Plot heatmap of agent density across shop floor.
    """
    figure(figsize=(14, 6))

    # Density during congested period (steps 100-300)
    subplot(1, 2, 1)

    density = zeros((SHOP_HEIGHT, SHOP_WIDTH))
    step_range = range(100, 300)

    for step in step_range:
        for pos in positions.get(step, []):
            x, y = int(pos[0]), int(pos[1])
            if 0 <= x < SHOP_WIDTH and 0 <= y < SHOP_HEIGHT:
                density[y, x] += 1

    density = density / len(step_range)  # Average per step
    density_smooth = gaussian_filter(density, sigma=1)

    im = imshow(density_smooth, origin='lower', cmap='hot',
                extent=[0, SHOP_WIDTH, 0, SHOP_HEIGHT], aspect='auto')
    colorbar(im, label='Avg agents per cell per step')

    # Mark entrance and exit
    scatter([ENTRANCE[0]], [ENTRANCE[1]], c='green', s=100,
            marker='s', label='Entrance', zorder=5)
    scatter([EXIT[0]], [EXIT[1]], c='blue', s=100,
            marker='s', label='Exit', zorder=5)

    # Mark aisle regions
    for y_min, y_max in AISLE_Y_RANGES:
        axhline(y=y_min, color='white', linestyle='--', alpha=0.3)
        axhline(y=y_max, color='white', linestyle='--', alpha=0.3)

    xlabel('X position')
    ylabel('Y position')
    title('Agent Density Heatmap (Steps 100-300)')
    legend(loc='upper right')

    # Density during exit phase (steps 300-500)
    subplot(1, 2, 2)

    density2 = zeros((SHOP_HEIGHT, SHOP_WIDTH))
    step_range2 = range(300, min(500, max(positions.keys())))

    for step in step_range2:
        for pos in positions.get(step, []):
            x, y = int(pos[0]), int(pos[1])
            if 0 <= x < SHOP_WIDTH and 0 <= y < SHOP_HEIGHT:
                density2[y, x] += 1

    if len(step_range2) > 0:
        density2 = density2 / len(step_range2)
    density2_smooth = gaussian_filter(density2, sigma=1)

    im2 = imshow(density2_smooth, origin='lower', cmap='hot',
                 extent=[0, SHOP_WIDTH, 0, SHOP_HEIGHT], aspect='auto')
    colorbar(im2, label='Avg agents per cell per step')

    scatter([ENTRANCE[0]], [ENTRANCE[1]], c='green', s=100, marker='s', zorder=5)
    scatter([EXIT[0]], [EXIT[1]], c='blue', s=100, marker='s', zorder=5)

    xlabel('X position')
    ylabel('Y position')
    title('Agent Density Heatmap (Steps 300-500)')

    tight_layout()
    savefig('analysis_spatial_density.png', dpi=150)
    print("Saved: analysis_spatial_density.png")


def plot_flow_field():
    """
    Plot velocity field showing flow directions.
    """
    figure(figsize=(14, 6))

    vx_mean, vy_mean, count = compute_flow_field(range(100, 400))

    # Flow field with arrows
    subplot(1, 2, 1)

    # Subsample for visibility
    skip = 2
    Y, X = np.mgrid[0:SHOP_HEIGHT:skip, 0:SHOP_WIDTH:skip]

    vx_sub = vx_mean[::skip, ::skip]
    vy_sub = vy_mean[::skip, ::skip]

    # Speed magnitude
    speed = sqrt(vx_sub**2 + vy_sub**2)

    plt.quiver(X, Y, vx_sub, vy_sub, speed, cmap='coolwarm',
               scale=30, alpha=0.8)
    colorbar(label='Speed')

    scatter([ENTRANCE[0]], [ENTRANCE[1]], c='green', s=100,
            marker='s', label='Entrance', zorder=5)
    scatter([EXIT[0]], [EXIT[1]], c='blue', s=100,
            marker='s', label='Exit', zorder=5)

    xlabel('X position')
    ylabel('Y position')
    title('Average Flow Field (Steps 100-400)')
    legend()
    axis([0, SHOP_WIDTH, 0, SHOP_HEIGHT])

    # Flow direction histogram
    subplot(1, 2, 2)

    # Compute flow angles where there's movement
    angles = []
    for y in range(SHOP_HEIGHT):
        for x in range(SHOP_WIDTH):
            if count[y, x] > 5:  # Enough data
                vx, vy = vx_mean[y, x], vy_mean[y, x]
                if vx != 0 or vy != 0:
                    angle = np.arctan2(vy, vx) * 180 / pi
                    angles.append(angle)

    plt.hist(angles, bins=36, range=(-180, 180), color='steelblue',
             alpha=0.7, edgecolor='black')
    axvline(x=45, color='red', linestyle='--', label='Entrance→Exit direction')
    xlabel('Flow angle (degrees)')
    ylabel('Count')
    title('Distribution of Flow Directions')
    legend()
    grid(True, alpha=0.3)

    tight_layout()
    savefig('analysis_flow_field.png', dpi=150)
    print("Saved: analysis_flow_field.png")


def plot_lane_detection():
    """
    Analyze lane formation in aisles.
    """
    figure(figsize=(14, 10))

    # Y-coordinate distribution by flow direction
    subplot(2, 2, 1)

    y_right = []  # Agents moving right
    y_left = []   # Agents moving left

    for aid, vels in velocities.items():
        traj = agent_trajectories[aid]
        for i, (vx, vy) in enumerate(vels):
            if i < len(traj):
                y = traj[i][1]
                if vx > 0:
                    y_right.append(y)
                elif vx < 0:
                    y_left.append(y)

    bins = arange(0, SHOP_HEIGHT + 1)
    plt.hist(y_right, bins=bins, alpha=0.5, label='Moving right (→)', color='blue')
    plt.hist(y_left, bins=bins, alpha=0.5, label='Moving left (←)', color='red')
    xlabel('Y position')
    ylabel('Count')
    title('Y-Distribution by Horizontal Flow Direction')
    legend()
    grid(True, alpha=0.3)

    # Lane order parameter over time for main corridor
    subplot(2, 2, 2)

    main_aisle = (11, 15)  # Middle aisle
    times, orders = detect_lane_formation_over_time(main_aisle)

    plot(times, orders, 'b-', linewidth=2, marker='o', markersize=4)
    axhline(y=0.3, color='green', linestyle='--', alpha=0.7,
            label='Lane formation threshold')
    xlabel('Time step')
    ylabel('Lane order parameter')
    title(f'Lane Formation in Aisle Y={main_aisle}')
    legend()
    grid(True, alpha=0.3)

    # Bimodality analysis for each aisle
    subplot(2, 2, 3)

    aisle_names = []
    bimodality_scores = []

    for y_min, y_max in AISLE_Y_RANGES:
        y_data = []
        for aid, traj in agent_trajectories.items():
            for pos in traj:
                if y_min <= pos[1] <= y_max:
                    y_data.append(pos[1])

        if len(y_data) > 50:
            bc = compute_bimodality_coefficient(y_data)
            bimodality_scores.append(bc)
            aisle_names.append(f'{y_min}-{y_max}')

    if bimodality_scores:
        colors = ['red' if b > 0.555 else 'steelblue' for b in bimodality_scores]
        plt.bar(aisle_names, bimodality_scores, color=colors, alpha=0.7)
        axhline(y=0.555, color='green', linestyle='--',
                label='Bimodality threshold (0.555)')
        xlabel('Aisle (Y range)')
        ylabel('Bimodality coefficient')
        title('Bimodality Test for Lane Formation')
        legend()
        plt.xticks(rotation=45)
        grid(True, alpha=0.3, axis='y')

    # Order parameter for all aisles
    subplot(2, 2, 4)

    aisle_orders = []
    for y_min, y_max in AISLE_Y_RANGES:
        order = compute_lane_order_parameter((y_min, y_max))
        aisle_orders.append(order)
        aisle_names_short = [f'{r[0]}-{r[1]}' for r in AISLE_Y_RANGES]

    plt.bar(aisle_names_short, aisle_orders, color='purple', alpha=0.7)
    axhline(y=0.3, color='green', linestyle='--', label='Weak lane formation')
    axhline(y=0.5, color='orange', linestyle='--', label='Strong lane formation')
    xlabel('Aisle (Y range)')
    ylabel('Lane order parameter')
    title('Lane Order Parameter by Aisle')
    legend()
    plt.xticks(rotation=45)
    grid(True, alpha=0.3, axis='y')

    tight_layout()
    savefig('analysis_lane_detection.png', dpi=150)
    print("Saved: analysis_lane_detection.png")


def plot_soc_analysis():
    """
    Self-organized criticality analysis.
    """
    figure(figsize=(14, 10))

    # Waiting time (avalanche) distribution
    subplot(2, 2, 1)

    avalanches = compute_waiting_avalanches()
    if len(avalanches) > 0:
        max_size = min(int(max(avalanches)) + 1, 50)
        bins = arange(1, max_size + 1)
        hist, edges = histogram(avalanches, bins=bins)

        # Log-log plot for power law detection
        sizes = edges[:-1]
        mask = hist > 0
        plot(sizes[mask], hist[mask], 'bo-', markersize=5)
        plt.yscale('log')
        plt.xscale('log')
        xlabel('Wait duration (steps)')
        ylabel('Frequency')
        title('Waiting Time Distribution (Log-Log)')
        grid(True, alpha=0.3)

        # Fit power law
        alpha, n_fit = fit_power_law(avalanches)
        if not np.isnan(alpha):
            text(0.7, 0.9, f'α ≈ {alpha:.2f}\n(n={n_fit})',
                 transform=gca().transAxes, fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='wheat'))

    # Cluster size distribution
    subplot(2, 2, 2)

    all_clusters = []
    for step in range(100, 400, 5):
        clusters = compute_congestion_clusters(step)
        all_clusters.extend(clusters)

    if len(all_clusters) > 0:
        max_cluster = min(max(all_clusters) + 1, 30)
        bins = arange(1, max_cluster + 1)
        hist, edges = histogram(all_clusters, bins=bins)

        sizes = edges[:-1]
        mask = hist > 0
        plot(sizes[mask], hist[mask], 'ro-', markersize=5)
        plt.yscale('log')
        plt.xscale('log')
        xlabel('Cluster size')
        ylabel('Frequency')
        title('Congestion Cluster Size Distribution')
        grid(True, alpha=0.3)

        alpha_c, n_fit_c = fit_power_law(all_clusters)
        if not np.isnan(alpha_c):
            text(0.7, 0.9, f'α ≈ {alpha_c:.2f}\n(n={n_fit_c})',
                 transform=gca().transAxes, fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='wheat'))

    # Velocity magnitude distribution
    subplot(2, 2, 3)

    speeds = []
    for aid, vels in velocities.items():
        for vx, vy in vels:
            speed = sqrt(vx**2 + vy**2)
            speeds.append(speed)

    plt.hist(speeds, bins=30, color='green', alpha=0.7, edgecolor='black')
    xlabel('Speed (cells/step)')
    ylabel('Frequency')
    title('Agent Speed Distribution')
    grid(True, alpha=0.3)

    # Fraction of blocked agents over time
    subplot(2, 2, 4)

    steps = sorted(positions.keys())
    blocked_fraction = []

    for step in steps:
        total = len(positions[step])
        if total == 0:
            blocked_fraction.append(0)
            continue

        blocked = 0
        for _, row in agent_data[agent_data['Step'] == step].iterrows():
            aid = row['agent_id']
            traj = agent_trajectories.get(aid, [])
            vels = velocities.get(aid, [])

            # Find this step in trajectory
            pos = row['pos_tuple']
            for i, p in enumerate(traj):
                if p == pos and i < len(vels):
                    vx, vy = vels[i]
                    if vx == 0 and vy == 0:
                        blocked += 1
                    break

        blocked_fraction.append(blocked / total)

    plot(steps, blocked_fraction, 'r-', linewidth=1, alpha=0.7)

    # Smooth
    from numpy import convolve
    if len(blocked_fraction) > 20:
        smooth = convolve(blocked_fraction, ones(20)/20, mode='same')
        plot(steps, smooth, 'darkred', linewidth=2, label='Smoothed')

    xlabel('Time step')
    ylabel('Fraction of blocked agents')
    title('Blocked Agent Ratio Over Time')
    legend()
    grid(True, alpha=0.3)

    tight_layout()
    savefig('analysis_soc.png', dpi=150)
    print("Saved: analysis_soc.png")


def plot_trajectory_sample():
    """
    Plot sample agent trajectories to visualize paths.
    """
    figure(figsize=(12, 8))

    # Sample trajectories
    sample_aids = list(agent_trajectories.keys())[:20]

    colors = plt.cm.tab20(linspace(0, 1, len(sample_aids)))

    for aid, color in zip(sample_aids, colors):
        traj = agent_trajectories[aid]
        if len(traj) > 2:
            x = [p[0] for p in traj]
            y = [p[1] for p in traj]
            plot(x, y, '-', color=color, alpha=0.5, linewidth=1)
            scatter([x[0]], [y[0]], color=color, marker='o', s=30)  # Start
            scatter([x[-1]], [y[-1]], color=color, marker='x', s=30)  # End

    # Mark entrance and exit
    scatter([ENTRANCE[0]], [ENTRANCE[1]], c='green', s=200,
            marker='s', label='Entrance', zorder=10, edgecolor='black')
    scatter([EXIT[0]], [EXIT[1]], c='blue', s=200,
            marker='s', label='Exit', zorder=10, edgecolor='black')

    # Draw shelf regions (approximate)
    shelf_regions = [
        (4, 5, 10, 2), (4, 9, 10, 2), (4, 15, 10, 2), (4, 21, 10, 2), (4, 25, 10, 2),
        (18, 5, 10, 2), (18, 9, 10, 2), (18, 15, 10, 2), (18, 21, 10, 2), (18, 25, 10, 2),
        (32, 5, 10, 2), (32, 9, 10, 2), (32, 15, 10, 2), (32, 21, 10, 2), (32, 25, 10, 2),
    ]

    for sx, sy, sw, sh in shelf_regions:
        rect = plt.Rectangle((sx, sy), sw, sh, color='gray', alpha=0.5)
        gca().add_patch(rect)

    xlabel('X position')
    ylabel('Y position')
    title('Sample Agent Trajectories (First 20 Agents)')
    legend()
    axis([0, SHOP_WIDTH, 0, SHOP_HEIGHT])
    grid(True, alpha=0.3)

    tight_layout()
    savefig('analysis_trajectories.png', dpi=150)
    print("Saved: analysis_trajectories.png")


def print_summary():
    """
    Print summary of lane formation and SOC findings.
    """
    print("\n" + "="*60)
    print("LANE FORMATION & SOC ANALYSIS SUMMARY")
    print("="*60)

    # Lane formation metrics
    print("\n--- LANE FORMATION ANALYSIS ---")

    for y_min, y_max in AISLE_Y_RANGES:
        order = compute_lane_order_parameter((y_min, y_max))
        if not np.isnan(order):
            status = "STRONG" if order > 0.5 else "WEAK" if order > 0.3 else "NONE"
            print(f"  Aisle Y={y_min}-{y_max}: Order={order:.3f} [{status}]")

    # SOC metrics
    print("\n--- SELF-ORGANIZED CRITICALITY ---")

    avalanches = compute_waiting_avalanches()
    if len(avalanches) > 0:
        alpha, n = fit_power_law(avalanches)
        print(f"  Waiting time distribution:")
        print(f"    - Mean wait: {mean(avalanches):.2f} steps")
        print(f"    - Max wait: {max(avalanches)} steps")
        print(f"    - Power law exponent α ≈ {alpha:.2f}")
        if 1.5 < alpha < 3.0:
            print(f"    - CONSISTENT with SOC (typical α: 1.5-3.0)")
        else:
            print(f"    - NOT typical SOC range")

    all_clusters = []
    for step in range(100, 400, 5):
        all_clusters.extend(compute_congestion_clusters(step))

    if all_clusters:
        alpha_c, _ = fit_power_law(all_clusters)
        print(f"  Cluster size distribution:")
        print(f"    - Mean cluster: {mean(all_clusters):.2f} agents")
        print(f"    - Max cluster: {max(all_clusters)} agents")
        print(f"    - Power law exponent α ≈ {alpha_c:.2f}")

    print("\n" + "="*60)


# =============================================================================
# MAIN
# =============================================================================

def run_lane_analysis():
    """
    Main analysis routine for lane detection and SOC.
    """
    print("="*60)
    print("LANE FORMATION & SELF-ORGANIZED CRITICALITY ANALYSIS")
    print("="*60)
    print()

    load_data()
    print()

    print("Generating visualizations...")
    plot_spatial_density_map()
    plot_flow_field()
    plot_lane_detection()
    plot_soc_analysis()
    plot_trajectory_sample()

    print_summary()

    print("\nAnalysis complete! Generated files:")
    print("  1. analysis_spatial_density.png")
    print("  2. analysis_flow_field.png")
    print("  3. analysis_lane_detection.png")
    print("  4. analysis_soc.png")
    print("  5. analysis_trajectories.png")
    print("="*60)

    show()


if __name__ == '__main__':
    run_lane_analysis()
