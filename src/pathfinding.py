"""BFS pathfinding for the supermarket simulation."""

from collections import deque


def bfs_path(start, goal, walkable_cells, occupied_cells=None, max_depth=None):
    """
    Find shortest path from start to goal using BFS.

    Args:
        start: Tuple (x, y) starting position
        goal: Tuple (x, y) target position
        walkable_cells: Set of (x, y) tuples that are walkable
        occupied_cells: Set of (x, y) tuples currently occupied by agents (optional)
        max_depth: Maximum search depth (optional, for performance)

    Returns:
        List of (x, y) tuples representing the path (excluding start, including goal),
        or None if no path exists.
    """
    if occupied_cells is None:
        occupied_cells = set()

    if start == goal:
        return []

    if goal not in walkable_cells:
        return None

    # BFS with parent tracking (more memory efficient than storing full paths)
    queue = deque([(start, 0)])  # (position, depth)
    parent = {start: None}

    while queue:
        current, depth = queue.popleft()

        # Check depth limit
        if max_depth is not None and depth >= max_depth:
            continue

        # Get neighbors (4-directional movement)
        x, y = current
        neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]

        for neighbor in neighbors:
            if neighbor in parent:
                continue
            # Check if walkable (goal is always reachable, others must not be occupied)
            if neighbor not in walkable_cells:
                continue
            if neighbor != goal and neighbor in occupied_cells:
                continue

            parent[neighbor] = current

            if neighbor == goal:
                # Reconstruct path
                path = []
                node = goal
                while node != start:
                    path.append(node)
                    node = parent[node]
                path.reverse()
                return path

            queue.append((neighbor, depth + 1))

    return None  # No path found


def get_adjacent_walkable(position, walkable_cells):
    """
    Get all walkable cells adjacent to a position.

    Args:
        position: Tuple (x, y)
        walkable_cells: Set of walkable (x, y) tuples

    Returns:
        List of adjacent walkable (x, y) tuples
    """
    x, y = position
    neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
    return [n for n in neighbors if n in walkable_cells]


def manhattan_distance(pos1, pos2):
    """Calculate Manhattan distance between two positions."""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
