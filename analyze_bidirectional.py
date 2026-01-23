"""
Quick Analysis of Bi-directional Flow Simulation
================================================
Checks for lane formation in counter-flow scenario.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ast
from collections import defaultdict
from scipy.ndimage import gaussian_filter

# Shop geometry for bidirectional config
SHOP_WIDTH = 60
SHOP_HEIGHT = 10
CORRIDOR_Y_MIN = 4
CORRIDOR_Y_MAX = 6

def parse_position(pos_str):
    try:
        return ast.literal_eval(pos_str)
    except:
        return None

def load_data():
    print("Loading bidirectional simulation data...")
    agent_df = pd.read_csv('data_bidirectional/agent_data.csv')
    model_df = pd.read_csv('data_bidirectional/model_data.csv')

    agent_df['pos_tuple'] = agent_df['position'].apply(parse_position)
    agent_df = agent_df.dropna(subset=['pos_tuple'])

    # Build trajectories and velocities
    positions = defaultdict(list)
    agent_trajectories = defaultdict(list)

    for _, row in agent_df.iterrows():
        step = int(row['Step'])
        pos = row['pos_tuple']
        aid = row['agent_id']
        positions[step].append(pos)
        agent_trajectories[aid].append((step, pos))

    # Sort trajectories by step
    for aid in agent_trajectories:
        agent_trajectories[aid].sort(key=lambda x: x[0])

    # Compute velocities
    velocities = {}
    for aid, traj in agent_trajectories.items():
        vels = []
        for i in range(1, len(traj)):
            _, pos_prev = traj[i-1]
            _, pos_curr = traj[i]
            dx = pos_curr[0] - pos_prev[0]
            dy = pos_curr[1] - pos_prev[1]
            vels.append((dx, dy))
        velocities[aid] = vels

    print(f"Loaded {len(agent_df)} records, {len(agent_trajectories)} agents")
    return agent_df, positions, agent_trajectories, velocities

def analyze_flow_direction(agent_df, velocities, agent_trajectories):
    """Classify agents by primary flow direction."""
    rightward_agents = []  # Moving predominantly right
    leftward_agents = []   # Moving predominantly left

    for aid, vels in velocities.items():
        if len(vels) < 5:
            continue
        total_dx = sum(v[0] for v in vels)
        if total_dx > 5:
            rightward_agents.append(aid)
        elif total_dx < -5:
            leftward_agents.append(aid)

    print(f"\nFlow direction analysis:")
    print(f"  Rightward agents: {len(rightward_agents)}")
    print(f"  Leftward agents: {len(leftward_agents)}")

    return rightward_agents, leftward_agents

def analyze_lane_separation(agent_trajectories, velocities, rightward, leftward):
    """Check if rightward and leftward agents use different Y-positions."""
    y_positions_right = []
    y_positions_left = []

    for aid in rightward:
        for step, pos in agent_trajectories[aid]:
            if CORRIDOR_Y_MIN <= pos[1] <= CORRIDOR_Y_MAX:
                y_positions_right.append(pos[1])

    for aid in leftward:
        for step, pos in agent_trajectories[aid]:
            if CORRIDOR_Y_MIN <= pos[1] <= CORRIDOR_Y_MAX:
                y_positions_left.append(pos[1])

    if y_positions_right and y_positions_left:
        mean_y_right = np.mean(y_positions_right)
        mean_y_left = np.mean(y_positions_left)
        separation = abs(mean_y_right - mean_y_left)

        print(f"\nLane separation analysis (in corridor Y={CORRIDOR_Y_MIN}-{CORRIDOR_Y_MAX}):")
        print(f"  Mean Y for rightward agents: {mean_y_right:.2f}")
        print(f"  Mean Y for leftward agents: {mean_y_left:.2f}")
        print(f"  Separation: {separation:.2f}")

        if separation > 0.3:
            print(f"  => LANE FORMATION DETECTED!")
        else:
            print(f"  => No significant lane separation")

        return y_positions_right, y_positions_left, separation
    return [], [], 0

def plot_analysis(agent_df, positions, agent_trajectories, velocities,
                  rightward, leftward, y_right, y_left):
    """Generate visualization plots."""
    fig = plt.figure(figsize=(16, 12))

    # 1. Density heatmap
    ax1 = fig.add_subplot(2, 2, 1)
    density = np.zeros((SHOP_HEIGHT, SHOP_WIDTH))
    for step in range(50, 300):
        for pos in positions.get(step, []):
            x, y = int(pos[0]), int(pos[1])
            if 0 <= x < SHOP_WIDTH and 0 <= y < SHOP_HEIGHT:
                density[y, x] += 1
    density_smooth = gaussian_filter(density, sigma=1)
    im = ax1.imshow(density_smooth, origin='lower', cmap='hot',
                    extent=[0, SHOP_WIDTH, 0, SHOP_HEIGHT], aspect='auto')
    plt.colorbar(im, ax=ax1, label='Agent density')
    ax1.axhline(y=CORRIDOR_Y_MIN, color='white', linestyle='--', alpha=0.5)
    ax1.axhline(y=CORRIDOR_Y_MAX, color='white', linestyle='--', alpha=0.5)
    ax1.set_xlabel('X position')
    ax1.set_ylabel('Y position')
    ax1.set_title('Agent Density Heatmap')

    # 2. Y-distribution by flow direction
    ax2 = fig.add_subplot(2, 2, 2)
    if y_right and y_left:
        bins = np.arange(0, SHOP_HEIGHT + 0.5, 0.5)
        ax2.hist(y_right, bins=bins, alpha=0.5, label=f'Rightward (n={len(rightward)})', color='blue')
        ax2.hist(y_left, bins=bins, alpha=0.5, label=f'Leftward (n={len(leftward)})', color='red')
        ax2.axvline(x=np.mean(y_right), color='blue', linestyle='--', linewidth=2)
        ax2.axvline(x=np.mean(y_left), color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Y position')
    ax2.set_ylabel('Count')
    ax2.set_title('Y-Position Distribution by Flow Direction')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Sample trajectories colored by direction
    ax3 = fig.add_subplot(2, 2, 3)
    # Plot rightward agents in blue
    for aid in rightward[:15]:
        traj = agent_trajectories[aid]
        x = [p[1][0] for p in traj]
        y = [p[1][1] for p in traj]
        ax3.plot(x, y, 'b-', alpha=0.4, linewidth=1)
    # Plot leftward agents in red
    for aid in leftward[:15]:
        traj = agent_trajectories[aid]
        x = [p[1][0] for p in traj]
        y = [p[1][1] for p in traj]
        ax3.plot(x, y, 'r-', alpha=0.4, linewidth=1)

    ax3.axhline(y=CORRIDOR_Y_MIN, color='gray', linestyle='--', alpha=0.5)
    ax3.axhline(y=CORRIDOR_Y_MAX, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlim(0, SHOP_WIDTH)
    ax3.set_ylim(0, SHOP_HEIGHT)
    ax3.set_xlabel('X position')
    ax3.set_ylabel('Y position')
    ax3.set_title('Sample Trajectories (Blue=Right, Red=Left)')
    ax3.grid(True, alpha=0.3)

    # 4. Flow field
    ax4 = fig.add_subplot(2, 2, 4)
    vx_sum = np.zeros((SHOP_HEIGHT, SHOP_WIDTH))
    vy_sum = np.zeros((SHOP_HEIGHT, SHOP_WIDTH))
    count = np.zeros((SHOP_HEIGHT, SHOP_WIDTH))

    for aid, traj in agent_trajectories.items():
        vels = velocities.get(aid, [])
        for i, (step, pos) in enumerate(traj):
            if i < len(vels):
                x, y = int(pos[0]), int(pos[1])
                if 0 <= x < SHOP_WIDTH and 0 <= y < SHOP_HEIGHT:
                    vx, vy = vels[i]
                    vx_sum[y, x] += vx
                    vy_sum[y, x] += vy
                    count[y, x] += 1

    count[count == 0] = 1
    vx_mean = vx_sum / count
    vy_mean = vy_sum / count

    skip = 3
    Y, X = np.mgrid[0:SHOP_HEIGHT:skip, 0:SHOP_WIDTH:skip]
    vx_sub = vx_mean[::skip, ::skip]
    vy_sub = vy_mean[::skip, ::skip]
    speed = np.sqrt(vx_sub**2 + vy_sub**2)

    ax4.quiver(X, Y, vx_sub, vy_sub, speed, cmap='coolwarm', scale=20, alpha=0.8)
    ax4.set_xlim(0, SHOP_WIDTH)
    ax4.set_ylim(0, SHOP_HEIGHT)
    ax4.set_xlabel('X position')
    ax4.set_ylabel('Y position')
    ax4.set_title('Average Flow Field')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('analysis_bidirectional.png', dpi=150)
    print("\nSaved: analysis_bidirectional.png")

def main():
    print("="*60)
    print("BI-DIRECTIONAL FLOW LANE FORMATION ANALYSIS")
    print("="*60)

    agent_df, positions, agent_trajectories, velocities = load_data()
    rightward, leftward = analyze_flow_direction(agent_df, velocities, agent_trajectories)
    y_right, y_left, separation = analyze_lane_separation(
        agent_trajectories, velocities, rightward, leftward)

    plot_analysis(agent_df, positions, agent_trajectories, velocities,
                  rightward, leftward, y_right, y_left)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    if separation > 0.5:
        print("Strong evidence of SELF-ORGANIZED LANE FORMATION")
    elif separation > 0.3:
        print("Weak evidence of lane formation")
    else:
        print("NO significant lane formation detected")

    # Additional: check if there's meaningful counter-flow
    if len(rightward) > 5 and len(leftward) > 5:
        print(f"\nCounter-flow present: {len(rightward)} right, {len(leftward)} left")
    else:
        print("\nInsufficient counter-flow for lane formation test")
        print("(Need agents moving in BOTH directions)")

if __name__ == '__main__':
    main()
