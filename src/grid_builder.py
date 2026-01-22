"""Build the shop grid from YAML configuration."""

from enum import Enum
from dataclasses import dataclass, field


class CellType(Enum):
    """Types of cells in the shop grid."""
    FLOOR = "."
    WALL = "#"
    SHELF = "S"
    PRODUCT = "P"
    ENTRANCE = "E"
    EXIT = "X"


@dataclass
class ProductLocation:
    """Represents a product location in the shop."""
    position: tuple  # (x, y)
    product_name: str


@dataclass
class ShopGrid:
    """Represents the shop layout."""
    width: int
    height: int
    cells: dict  # {(x, y): CellType}
    walkable_cells: set  # Set of (x, y) that agents can walk on
    entrance_cells: list  # List of (x, y) entrance positions
    exit_cells: list  # List of (x, y) exit positions
    product_locations: list  # List of ProductLocation
    product_positions: dict = field(default_factory=dict)  # {product_name: (x, y)}

    def is_walkable(self, x, y):
        """Check if a cell is walkable."""
        return (x, y) in self.walkable_cells

    def get_cell_type(self, x, y):
        """Get the type of a cell."""
        return self.cells.get((x, y), CellType.FLOOR)


def build_grid(config):
    """
    Build the shop grid from configuration.

    Args:
        config: Dictionary containing shop_layout configuration

    Returns:
        ShopGrid object
    """
    layout = config["shop_layout"]
    width = layout["size"]["width"]
    height = layout["size"]["height"]

    cells = {}
    walkable_cells = set()
    entrance_cells = []
    exit_cells = []
    product_locations = []
    product_positions = {}

    # Initialize all cells as floor
    for x in range(width):
        for y in range(height):
            cells[(x, y)] = CellType.FLOOR
            walkable_cells.add((x, y))

    # Add boundary walls
    for x in range(width):
        cells[(x, 0)] = CellType.WALL
        cells[(x, height - 1)] = CellType.WALL
        walkable_cells.discard((x, 0))
        walkable_cells.discard((x, height - 1))

    for y in range(height):
        cells[(0, y)] = CellType.WALL
        cells[(width - 1, y)] = CellType.WALL
        walkable_cells.discard((0, y))
        walkable_cells.discard((width - 1, y))

    # Add shelves and products
    for shelf in layout.get("shelves", []):
        shelf_cells, shelf_products = _build_shelf(shelf, cells, walkable_cells, width, height)
        product_locations.extend(shelf_products)
        for prod in shelf_products:
            product_positions[prod.product_name] = prod.position

    # Add entrances
    for entrance in layout.get("entrances", []):
        for dx in range(entrance.get("width", 1)):
            for dy in range(entrance.get("height", 1)):
                x, y = entrance["x"] + dx, entrance["y"] + dy
                if 0 <= x < width and 0 <= y < height:
                    cells[(x, y)] = CellType.ENTRANCE
                    walkable_cells.add((x, y))
                    entrance_cells.append((x, y))

    # Add exits
    for exit_zone in layout.get("exits", []):
        for dx in range(exit_zone.get("width", 1)):
            for dy in range(exit_zone.get("height", 1)):
                x, y = exit_zone["x"] + dx, exit_zone["y"] + dy
                if 0 <= x < width and 0 <= y < height:
                    cells[(x, y)] = CellType.EXIT
                    walkable_cells.add((x, y))
                    exit_cells.append((x, y))

    return ShopGrid(
        width=width,
        height=height,
        cells=cells,
        walkable_cells=walkable_cells,
        entrance_cells=entrance_cells,
        exit_cells=exit_cells,
        product_locations=product_locations,
        product_positions=product_positions,
    )


def _build_shelf(shelf, cells, walkable_cells, grid_width, grid_height):
    """
    Build a shelf and place products on accessible edges.

    Args:
        shelf: Shelf configuration dict
        cells: Grid cells dict to modify
        walkable_cells: Set of walkable cells to modify
        grid_width: Width of the grid
        grid_height: Height of the grid

    Returns:
        Tuple of (shelf_cells, product_locations)
    """
    shelf_cells = set()
    product_locations = []

    if shelf["type"] == "rectangle":
        sx, sy = shelf["x"], shelf["y"]
        sw, sh = shelf["width"], shelf["height"]

        # Mark shelf cells
        for dx in range(sw):
            for dy in range(sh):
                x, y = sx + dx, sy + dy
                if 0 <= x < grid_width and 0 <= y < grid_height:
                    cells[(x, y)] = CellType.SHELF
                    walkable_cells.discard((x, y))
                    shelf_cells.add((x, y))

        # Find accessible edge cells for products
        edge_cells = _get_accessible_edges(shelf_cells, cells, grid_width, grid_height)

        # Place products on edge cells
        products = shelf.get("products", [])
        if products and edge_cells:
            for i, product_name in enumerate(products):
                edge_pos = edge_cells[i % len(edge_cells)]
                cells[edge_pos] = CellType.PRODUCT
                walkable_cells.discard(edge_pos)
                product_locations.append(ProductLocation(edge_pos, product_name))

    return shelf_cells, product_locations


def _get_accessible_edges(shelf_cells, cells, grid_width, grid_height):
    """
    Get shelf edge cells that have at least one adjacent walkable floor cell.

    Args:
        shelf_cells: Set of (x, y) cells that are part of the shelf
        cells: Current grid cells dict
        grid_width: Width of the grid
        grid_height: Height of the grid

    Returns:
        List of (x, y) edge cells that are accessible
    """
    edge_cells = []

    for x, y in shelf_cells:
        # Check if this cell is on an edge (has at least one non-shelf neighbor)
        neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
        for nx, ny in neighbors:
            if (nx, ny) not in shelf_cells:
                # Check if neighbor is walkable floor
                if 0 <= nx < grid_width and 0 <= ny < grid_height:
                    if cells.get((nx, ny)) == CellType.FLOOR:
                        edge_cells.append((x, y))
                        break

    return edge_cells
