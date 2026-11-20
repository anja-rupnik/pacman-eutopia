"""Per-layout static maps (numpy only, Python 3.8 compatible): used by training tables and by the contest agent."""
from __future__ import annotations

import numpy as np

from pacman_jax.common import UNREACHABLE


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


def nearest_junction(walls: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """[W*H] int32: the nearest open cell with at least 3 open neighbours (a junction, where a pacman can turn away
    from a ghost) of every cell; the cell itself on junctions, walls and layouts without junctions."""
    padded = np.pad(~walls, 1)
    degree = padded[2:, 1:-1].astype(int) + padded[:-2, 1:-1] + padded[1:-1, 2:] + padded[1:-1, :-2]
    junction = (~walls & (degree >= 3)).reshape(-1)
    own = np.arange(walls.size, dtype=np.int32)
    if not junction.any():
        return own
    ids = np.flatnonzero(junction)
    nearest = ids[dist[:, ids].argmin(axis=1)].astype(np.int32)
    return np.where(walls.reshape(-1), own, nearest)


def layout_maps(walls: np.ndarray, width: int, dist: np.ndarray) -> dict:
    """home_dist [2,W,H] uint8, boundary [2,H] int32, boundary_mask [2,H] bool, dead_end [W,H] uint8, and the
    pocket maps (pocket_maps)."""
    shape = walls.shape
    home_dist = np.full((2,) + shape, UNREACHABLE, dtype=np.uint8)
    boundary = np.zeros((2, shape[1]), dtype=np.int32)
    boundary_mask = np.zeros((2, shape[1]), dtype=bool)
    for team in range(2):
        cells, mask = boundary_cells(walls, width, team)
        boundary[team], boundary_mask[team] = cells, mask
        home_dist[team] = dist[cells[mask]].min(axis=0).reshape(shape)
    return dict(home_dist=home_dist, boundary=boundary, boundary_mask=boundary_mask,
                dead_end=dead_end_depth(walls, dist), **pocket_maps(walls))


def open_graph(walls: np.ndarray) -> list:
    """Neighbour lists of the open cells (cell id = x * H + y; walls and padding have no neighbours)."""
    width, height = walls.shape
    adj = [[] for _ in range(width * height)]
    for x in range(width):
        for y in range(height):
            if walls[x, y]:
                continue
            for dx, dy in ((1, 0), (0, 1)):
                nx, ny = x + dx, y + dy
                if nx < width and ny < height and not walls[nx, ny]:
                    adj[x * height + y].append(nx * height + ny)
                    adj[nx * height + ny].append(x * height + y)
    return adj


def cut_pockets(adj: list, cells: list, removed: int = -1) -> dict:
    """Articulation points of the graph without `removed`, with the cells each one cuts off.

    One iterative DFS (Tarjan). For an articulation point a, the parts left when it is removed are the DFS subtrees of
    children c with low[c] >= disc[a] and the rest of a's component; the largest part is the main body and the others
    are a's pockets. Returns {a: [pocket cell arrays]}.
    """
    n = len(adj)
    disc = [-1] * n
    low = [0] * n
    size = [1] * n
    order = []
    out: dict = {}
    for root in cells:
        if disc[root] >= 0 or root == removed:
            continue
        start = len(order)
        disc[root] = low[root] = start
        order.append(root)
        stack = [(root, -1, 0)]
        cuts = []  # (articulation point, child)
        while stack:
            v, parent, i = stack[-1]
            if i < len(adj[v]):
                stack[-1] = (v, parent, i + 1)
                w = adj[v][i]
                if w == removed or w == parent:
                    continue
                if disc[w] < 0:
                    disc[w] = low[w] = len(order)
                    order.append(w)
                    stack.append((w, v, 0))
                else:
                    low[v] = min(low[v], disc[w])
            else:
                stack.pop()
                if parent >= 0:
                    low[parent] = min(low[parent], low[v])
                    size[parent] += size[v]
                    if low[v] >= disc[parent]:
                        cuts.append((parent, v))
        component = np.array(order[start:], dtype=np.int64)
        by_point: dict = {}
        for a, c in cuts:
            by_point.setdefault(a, []).append(c)
        for a, children in by_point.items():
            if a == root and len(children) < 2:
                continue  # the root is an articulation point only with two or more DFS children
            parts = [np.array(order[disc[c]:disc[c] + size[c]], dtype=np.int64) for c in children]
            rest = len(component) - 1 - sum(len(p) for p in parts)
            if rest > 0:
                mask = np.ones(n, dtype=bool)
                mask[a] = False
                for p in parts:
                    mask[p] = False
                parts.append(component[mask[component]])
            main = max(range(len(parts)), key=lambda j: len(parts[j]))
            out[a] = [p for j, p in enumerate(parts) if j != main]
    return out


def pocket_maps(walls: np.ndarray) -> dict:
    """Map structure: areas that a few cells seal off from the rest of the maze (at most a third of it).

    pocket_gate [W*H] int32: cell id of the outermost single gate cutting the cell off (W*H if none);
    pocket_paint [W*H] int32: pocket_gate on pocket cells, the cell itself on gate cells, W*H elsewhere;
    gate [W,H] float32: 1 on single gates, 0.5 on the junctions of two-junction cuts;
    seal_cost [W,H] uint8: cells we must hold to seal the cell off: 1, 2, or 0 (not sealable).
    """
    shape = walls.shape
    n = shape[0] * shape[1]
    adj = open_graph(walls)
    cells = [v for v in range(n) if adj[v]]
    limit = len(cells) / 3.0
    pocket_gate = np.full(n, n, dtype=np.int32)
    gate = np.zeros(n, dtype=np.float32)
    seal = np.zeros(n, dtype=np.uint8)
    singles = cut_pockets(adj, cells)
    pockets = sorted(((len(p), a, p) for a, parts in singles.items() for p in parts if len(p) <= limit),
                     key=lambda t: (-t[0], t[1]))
    for _, a, p in pockets:  # largest first: each cell keeps its outermost gate
        fresh = p[pocket_gate[p] == n]
        if len(fresh):
            pocket_gate[fresh] = a
            gate[a] = 1.0
    seal[pocket_gate < n] = 1
    # two gates: pairs of junctions (cutting a corridor anywhere is like cutting it at its end junction) that seal
    # off an area with junctions of its own, i.e. more than a piece of corridor
    junctions = [v for v in cells if len(adj[v]) >= 3 and v not in singles]
    for u in junctions:
        for v, parts in cut_pockets(adj, cells, removed=u).items():
            if v in singles or len(adj[v]) < 3:
                continue
            for p in parts:
                if len(p) <= limit and any(len(adj[c]) >= 3 for c in p):
                    seal[p[seal[p] == 0]] = 2
                    for c in (u, v):
                        gate[c] = max(gate[c], 0.5)
    paint = pocket_gate.copy()
    gates = np.flatnonzero(gate == 1.0)
    paint[gates] = gates
    return dict(pocket_gate=pocket_gate, pocket_paint=paint, gate=gate.reshape(shape), seal_cost=seal.reshape(shape))
