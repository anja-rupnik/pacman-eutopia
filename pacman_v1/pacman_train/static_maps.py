"""Per-layout static maps (numpy only, Python 3.8 compatible): used by training tables and by the contest agent."""
from __future__ import annotations

import numpy as np

from pacman_v1.pacman_jax.common import UNREACHABLE


def boundary_cells(walls: np.ndarray, width: int, team: int) -> tuple[np.ndarray, np.ndarray]:
    """Cell ids of a team's own boundary column, and which of them are open."""
    height = walls.shape[1]
    x = width // 2 - 1 if team == 0 else width // 2
    cells = np.array([x * height + y for y in range(height)], dtype=np.int32)
    return cells, ~walls[x]


def dead_end_depth(walls: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """Peel cells of degree <= 1 until only cycles remain; depth = maze distance to the nearest remaining cell."""
    open_ = ~walls
    alive = open_.copy()
    while True:
        padded = np.pad(alive, 1)
        degree = padded[2:, 1:-1].astype(int) + padded[:-2, 1:-1] + padded[1:-1, 2:] + padded[1:-1, :-2]
        leaves = alive & (degree <= 1)
        if not leaves.any():
            break
        alive &= ~leaves
    if not alive.any():
        return np.zeros(walls.shape, dtype=np.uint8)
    to_cycle = dist[:, alive.reshape(-1)].min(axis=1).reshape(walls.shape)
    return np.where(open_, to_cycle, 0).astype(np.uint8)


def layout_maps(walls: np.ndarray, width: int, dist: np.ndarray) -> dict:
    """home_dist [2,W,H] uint8, boundary [2,H] int32, boundary_mask [2,H] bool, dead_end [W,H] uint8."""
    shape = walls.shape
    home_dist = np.full((2,) + shape, UNREACHABLE, dtype=np.uint8)
    boundary = np.zeros((2, shape[1]), dtype=np.int32)
    boundary_mask = np.zeros((2, shape[1]), dtype=bool)
    for team in range(2):
        cells, mask = boundary_cells(walls, width, team)
        boundary[team], boundary_mask[team] = cells, mask
        home_dist[team] = dist[cells[mask]].min(axis=0).reshape(shape)
    return dict(home_dist=home_dist, boundary=boundary, boundary_mask=boundary_mask,
                dead_end=dead_end_depth(walls, dist))
