from __future__ import annotations  # shipped with the contest agent, which runs on Python 3.8

from types import ModuleType
from typing import Any

import numpy as np

Array = Any

NUM_AGENTS: int = 4
NUM_ACTIONS: int = 5
STOP: int = 4
ACTION_NAMES: tuple[str, ...] = ("North", "South", "East", "West", "Stop")
ACTION_VECTORS: np.ndarray = np.array([[0, 1], [0, -1], [1, 0], [-1, 0], [0, 0]], dtype=np.int32)
FLIP_ACTION: np.ndarray = np.array([1, 0, 3, 2, 4], dtype=np.int32)
SIGHT_RANGE: int = 5
NOISE_MAX: int = 6
SCARED_TIME: int = 40
MIN_FOOD: int = 2
UNREACHABLE: int = 255
CARRY_SCALE: float = 20.0
SCORE_SCALE: float = 20.0
NUM_GRID_CHANNELS: int = 13
NUM_VEC_FEATURES: int = 22


def team_of(agent: int) -> int:
    return agent % 2


def team_members(team: int) -> tuple[int, int]:
    return team, team + 2


def enemies_of(team: int) -> tuple[int, int]:
    return 1 - team, 3 - team


def coords(xp: ModuleType, width: int, height: int) -> tuple[Array, Array]:
    return xp.arange(width)[:, None], xp.arange(height)[None, :]


def onehot_grid(xp: ModuleType, pos: Array, shape: tuple[int, int]) -> Array:
    xs, ys = coords(xp, shape[0], shape[1])
    return (xs == pos[0]) & (ys == pos[1])


def manhattan_grid(xp: ModuleType, pos: Array, shape: tuple[int, int]) -> Array:
    xs, ys = coords(xp, shape[0], shape[1])
    return xp.abs(xs - pos[0]) + xp.abs(ys - pos[1])


def legal_world_actions(xp: ModuleType, walls: Array, pos: Array) -> Array:
    vec = xp.asarray(ACTION_VECTORS)
    return ~walls[pos[0] + vec[:, 0], pos[1] + vec[:, 1]]


def to_world_action(xp: ModuleType, action: Array, team: Array) -> Array:
    return xp.where(team == 1, xp.asarray(FLIP_ACTION)[action], action)


def canonical_index(xp: ModuleType, size: int, extent: Array, flip: Array) -> Array:
    idx = xp.arange(size)
    return xp.where(flip & (idx < extent), extent - 1 - idx, idx)


def canonicalize(xp: ModuleType, grid: Array, team: Array, width: Array, height: Array) -> Array:
    flip = team == 1
    rx = canonical_index(xp, grid.shape[0], width, flip)
    ry = canonical_index(xp, grid.shape[1], height, flip)
    return grid[rx[:, None], ry[None, :]]


def neighbour_shifts(xp: ModuleType, grid: Array) -> list[Array]:
    return [xp.roll(grid, -1, axis=1), xp.roll(grid, 1, axis=1), xp.roll(grid, -1, axis=0), xp.roll(grid, 1, axis=0)]


def arrival_shifts(xp: ModuleType, grid: Array) -> list[Array]:
    return [xp.roll(grid, 1, axis=1), xp.roll(grid, -1, axis=1), xp.roll(grid, 1, axis=0), xp.roll(grid, -1, axis=0)]


def diffuse(xp: ModuleType, belief: Array, walls: Array) -> Array:
    open_ = (~walls).astype(belief.dtype)
    degree = 1.0 + sum(neighbour_shifts(xp, open_))
    outflow = belief * open_ / degree
    return (outflow + sum(arrival_shifts(xp, outflow))) * open_


def observe_beliefs(xp: ModuleType, belief: Array, walls: Array, observer: Array, team_pos: Array,
                    enemy_pos: Array, enemy_visible: Array, readings: Array, home: Array, enemy_pacman: Array,
                    eaten: Array, eater: Array, enemy_start: Array) -> Array:
    """Bayesian update of the team's belief over its two enemies.

    Evidence, strongest first: a visible enemy is exactly where it is; if our food or a capsule vanished since our
    last look, the enemy who just moved (slot `eater`, -1 if none) is on that cell; otherwise the enemy is unseen,
    consistent with the noisy distance reading, and on the half its `enemy_pacman` flag implies (pacman: our `home`
    half; ghost: its own half). The prior covers every cell the enemy could have moved to, so
    evidence contradicting it means the enemy was eaten and respawned: restart from its start cell (`enemy_start`),
    or from the evidence alone if even that is inconsistent.
    """
    shape = (walls.shape[0], walls.shape[1])
    unseen = (manhattan_grid(xp, team_pos[0], shape) > SIGHT_RANGE) & (manhattan_grid(xp, team_pos[1], shape) > SIGHT_RANGE)
    observer_dist = manhattan_grid(xp, observer, shape)
    rows = []
    for k in range(2):
        side = xp.where(enemy_pacman[k], home, ~home)
        heard = unseen & ~walls & (xp.abs(observer_dist - readings[k]) <= NOISE_MAX)
        likely = (heard & side).astype(belief.dtype)
        likely = xp.where(likely.sum() > 0, likely, heard.astype(belief.dtype))
        exact = onehot_grid(xp, enemy_pos[k], shape).astype(belief.dtype)
        ate = (eaten & heard).astype(belief.dtype) * (eater == k)
        posterior = xp.where(ate.sum() > 0, ate, belief[k] * likely)
        posterior = xp.where(enemy_visible[k], exact, posterior)
        start = onehot_grid(xp, enemy_start[k], shape).astype(belief.dtype)
        # `eater` is also the slot of the enemy that moved since our last look: it may have left its start already
        respawn = xp.where(eater == k, diffuse(xp, start, walls), start) * heard
        restart = xp.where(respawn.sum() > 0, respawn, likely)
        posterior = xp.where(posterior.sum() > 0, posterior, restart)
        rows.append(posterior / posterior.sum())
    return xp.stack(rows)


def home_side(xp: ModuleType, shape: tuple[int, int], width: Array, height: Array, team: Array) -> Array:
    xs, ys = coords(xp, shape[0], shape[1])
    valid = (xs < width) & (ys < height)
    red_side = valid & (xs < width // 2)
    return xp.where(team == 0, red_side, valid & ~red_side)


def enemy_visibility(xp: ModuleType, pos: Array, team: int) -> Array:
    members = xp.asarray(team_members(team))
    enemies = xp.asarray(enemies_of(team))
    diff = xp.abs(pos[enemies][:, None, :] - pos[members][None, :, :]).sum(-1)
    return (diff <= SIGHT_RANGE).any(-1)


def build_observation(xp: ModuleType, agent: int, walls: Array, width: Array, height: Array, dist_self: Array,
                      food: Array, capsules: Array, pos: Array, known: Array, is_pacman: Array, scared: Array,
                      carrying: Array, score: Array, timeleft: Array, max_turns: int, belief: Array) -> tuple[Array, Array]:
    team = team_of(agent)
    mate = (agent + 2) % NUM_AGENTS
    enemies = enemies_of(team)
    shape = (walls.shape[0], walls.shape[1])
    xs, ys = coords(xp, shape[0], shape[1])
    valid = (xs < width) & (ys < height)
    red_side = valid & (xs < width // 2)
    home = red_side if team == 0 else valid & ~red_side
    away = valid & ~home
    order = (agent, mate) + enemies
    positions = [onehot_grid(xp, pos[i], shape) & known[i] for i in order]
    planes = [walls, home, food & away, food & home, capsules & away, capsules & home] + positions + [belief[0], belief[1], dist_self / UNREACHABLE]
    grid = xp.stack([xp.asarray(p).astype(xp.float32) for p in planes], axis=-1)
    grid = canonicalize(xp, grid, xp.asarray(team), width, height)
    per_agent = [[scared[i] / SCARED_TIME, carrying[i] / CARRY_SCALE, is_pacman[i], known[i]] for i in order]
    sign = 1 if team == 0 else -1
    agent_id = [1.0 if i == agent else 0.0 for i in range(NUM_AGENTS)]
    vec = [v for row in per_agent for v in row] + [sign * score / SCORE_SCALE, timeleft / max_turns] + agent_id
    return grid, xp.stack([xp.asarray(v).astype(xp.float32) for v in vec])


def maze_distances(walls: np.ndarray) -> np.ndarray:
    width, height = walls.shape
    cells = width * height
    open_ = ~walls
    reached = np.zeros((cells, width, height), dtype=bool)
    reached.reshape(cells, cells)[np.arange(cells), np.arange(cells)] = open_.reshape(-1)
    dist = np.full((cells, width, height), UNREACHABLE, dtype=np.int32)
    dist[reached] = 0
    frontier = reached.copy()
    step = 0
    while frontier.any():
        step += 1
        assert step < UNREACHABLE
        grown = np.zeros_like(frontier)
        grown[:, 1:, :] |= frontier[:, :-1, :]
        grown[:, :-1, :] |= frontier[:, 1:, :]
        grown[:, :, 1:] |= frontier[:, :, :-1]
        grown[:, :, :-1] |= frontier[:, :, 1:]
        frontier = grown & open_[None] & ~reached
        reached |= frontier
        dist[frontier] = step
    return dist.reshape(cells, cells).astype(np.uint8)
