"""BFS pathfinding for the supermarket simulation."""

from collections import deque


def bfs_path(start, goal, walkable_cells, occupied_cells=None):
    """
    Find shortest path from start to goal using BFS.

    Args:
        start: Tuple (x, y) starting position
        goal: Tuple (x, y) target position
        walkable_cells: Set of (x, y) tuples that are walkable
        occupied_cells: Set of (x, y) tuples currently occupied by agents (optional)

    Returns:
        List of (x, y) tuples representing the path (excluding start, including goal),
        or None if no path exists.
    """
    if occupied_cells is None:
        occupied_cells = set()

    if start == goal:
        return []

    # Cells we can walk through (walkable and not occupied, except goal)
    available = walkable_cells - occupied_cells
    available.add(goal)  # Goal is always considered reachable
    available.add(start)  # Start is always available

    if goal not in walkable_cells:
        return None

    # BFS
    queue = deque([(start, [start])])
    visited = {start}

    while queue:
        current, path = queue.popleft()

        # Get neighbors (4-directional movement)
        x, y = current
        neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]

        for neighbor in neighbors:
            if neighbor in visited:
                continue
            if neighbor not in available:
                continue

            new_path = path + [neighbor]

            if neighbor == goal:
                return new_path[1:]  # Exclude start position

            visited.add(neighbor)
            queue.append((neighbor, new_path))

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
