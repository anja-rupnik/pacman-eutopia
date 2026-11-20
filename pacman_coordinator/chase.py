"""Ghost-chase search for v3: prove that our ghost(s) can catch a visible invader whatever it does.

A visible enemy pacman on our side has an exact position, so a chase near it is a perfect-information game: our
chaser(s) and the invader move in the contest's turn order (ours, theirs, ours, theirs), and every other agent is
left out. The search is a retrograde analysis over all (chaser cells, invader cell, turn) states of a region
around the invader: a state is won when one of our moves leads to a won state (or onto the invader), and for the
invader every legal move does (or walks it into a ghost). Everything the model leaves out is decided against us,
so a proven capture is a real one:
- chasers stay on our side inside the region (fewer moves for us);
- the invader escapes by crossing home, stepping on one of our capsules (our ghosts would be scared) or leaving
  the region;
- the turn order is exact; the other enemy and an unused teammate never move in the model (neither can stop the
  capture: the teammate is a ghost too, and the brain only searches when the other enemy can't reach a capsule).
Python 3.8 + numpy, like the rest of the contest code.
"""
from __future__ import annotations

import time
from typing import Optional, Sequence

import numpy as np

from pacman_jax.common import ACTION_VECTORS, NUM_ACTIONS, UNREACHABLE

BLOCKED = -1   # a wall (or, for a chaser, a cell it may not enter)
ESCAPE = -2    # the invader gets away: home, a capsule or out of the region


class Region:
    """The cells of our side nearest the invader (at most `max_cells`), with each cell's move targets."""

    def __init__(self, walls: np.ndarray, home: np.ndarray, dist: np.ndarray, capsules: np.ndarray,
                 invader: Sequence[int], max_cells: int) -> None:
        width, height = walls.shape
        d = dist[invader[0] * height + invader[1]].reshape(width, height)  # dist: [cells, cells] (maze_distances)
        candidate = home & ~walls & (d < UNREACHABLE)
        xs, ys = np.nonzero(candidate)
        keep = np.argsort(d[xs, ys], kind="stable")[:max_cells]
        self.xs, self.ys = xs[keep], ys[keep]
        self.size = len(keep)
        self.index = np.full((width, height), -1, dtype=np.int64)
        self.index[self.xs, self.ys] = np.arange(self.size)
        tx = self.xs[:, None] + ACTION_VECTORS[None, :, 0]
        ty = self.ys[:, None] + ACTION_VECTORS[None, :, 1]
        wall = walls[tx, ty]
        target = self.index[tx, ty]
        # chasers: only region cells (our side, near the invader)
        self.chaser_moves = np.where(wall | (target < 0), BLOCKED, target)
        # invader: crossing home, a capsule or leaving the region all count as escapes
        gone = ~home[tx, ty] | capsules[tx, ty] | (target < 0)
        self.invader_moves = np.where(wall, BLOCKED, np.where(gone, ESCAPE, target))
        self.invader_moves[:, -1] = np.arange(self.size)  # STOP: it is on a region cell already

    def cell(self, pos: Sequence[int]) -> int:
        return int(self.index[pos[0], pos[1]])


def solve(region: Region, chasers: Sequence[int], invader: int, phases: Sequence[str], max_plies: int,
          deadline: float = float("inf")) -> Optional[tuple]:
    """Shortest forced capture from the root (`phases[0]` to move, chaser 0 being the mover).

    `chasers`: region cells of the chasers (1 or 2), `invader`: its region cell. `phases`: the roles of the next 4
    turns, starting now, each "c0", "c1", "inv" or "pass". Returns (plies, set of the mover's capturing moves) or
    None if there is no capture within `max_plies` (or the deadline passes first)."""
    n, k = region.size, len(chasers)
    if invader < 0 or min(chasers) < 0:
        raise ValueError("chasers and invader must be region cells (Region.cell >= 0)")
    shape = (n,) * (k + 1)
    ar = np.arange(n)
    root = tuple(chasers) + (invader,)
    win = [np.zeros(shape, dtype=bool) for _ in range(4)]
    dist = [np.full(shape, -1, dtype=np.int16) for _ in range(4)]

    def along(axis: int, values: np.ndarray) -> np.ndarray:  # a vector laid along one axis of the state array
        view = [1] * (k + 1)
        view[axis] = n
        return values.reshape(view)

    def chaser_step(nxt: np.ndarray, j: int) -> np.ndarray:
        out = np.zeros(shape, dtype=bool)
        for m in range(NUM_ACTIONS):
            target = region.chaser_moves[:, m]
            valid = target >= 0
            safe = np.where(valid, target, 0)
            caught = along(j, safe) == along(k, ar)  # moved onto the invader
            out |= (np.take(nxt, safe, axis=j) | caught) & along(j, valid)
        return out

    def invader_step(nxt: np.ndarray) -> np.ndarray:
        out = np.ones(shape, dtype=bool)
        for m in range(NUM_ACTIONS):
            target = region.invader_moves[:, m]
            legal = target != BLOCKED
            safe = np.where(target >= 0, target, 0)
            hit = np.zeros(shape, dtype=bool)
            for j in range(k):  # walking into a ghost dies
                hit |= along(j, ar) == along(k, safe)
            ok = np.where(along(k, target == ESCAPE), False, hit | np.take(nxt, safe, axis=k))
            out &= np.where(along(k, legal), ok, True)
        return out

    for ply in range(1, max_plies + 1):
        new = []
        for p, role in enumerate(phases):
            if time.perf_counter() > deadline:  # also within a sweep: a big one takes a while on a slow CPU
                return None
            nxt = win[(p + 1) % 4]
            if role == "pass":
                new.append(nxt.copy())
            elif role == "inv":
                new.append(invader_step(nxt))
            else:
                new.append(chaser_step(nxt, int(role[1])))
        for p in range(4):
            dist[p][new[p] & ~win[p]] = ply
        win = new
        if win[0][root]:
            break
        if time.perf_counter() > deadline:
            return None
    else:
        return None
    plies = int(dist[0][root])
    best = set()
    for m in range(NUM_ACTIONS):
        target = int(region.chaser_moves[chasers[0], m])
        if target < 0:
            continue
        if target == invader:
            best.add(m)
            continue
        state = (target,) + tuple(chasers[1:]) + (invader,)
        if win[1][state] and dist[1][state] == plies - 1:
            best.add(m)
    return plies, best


def escapes(region: Region, chasers: Sequence[int], invader: int, phases: Sequence[str], max_plies: int = 400,
            deadline: float = float("inf")) -> Optional[bool]:
    """Can the invader force its way out (home, a capsule or out of the region) against the chasers, however long
    it takes? False means the chasers can contain it for good (catching it or keeping it from every exit), the
    model's simplifications counting against them as in `solve`. None if the deadline passes first."""
    if invader < 0 or min(chasers) < 0:
        raise ValueError("chasers and invader must be region cells (Region.cell >= 0)")
    out = escape_map(region, len(chasers), phases, max_plies, deadline)
    return None if out is None else bool(out[0][tuple(chasers) + (invader,)])


def escape_map(region: Region, k: int, phases: Sequence[str], max_plies: int = 400,
               deadline: float = float("inf")) -> Optional[list]:
    """For every state of `k` chasers and the invader, per turn phase: can the invader force its way out?
    [4] boolean arrays indexed [chaser cells..., invader cell]. The invader's winning states grow from the exits
    backwards until nothing changes (at most `max_plies` sweeps). None if the deadline passes first."""
    n = region.size
    shape = (n,) * (k + 1)
    ar = np.arange(n)
    out = [np.zeros(shape, dtype=bool) for _ in range(4)]  # the invader can force an exit from here

    def along(axis: int, values: np.ndarray) -> np.ndarray:
        view = [1] * (k + 1)
        view[axis] = n
        return values.reshape(view)

    def chaser_step(nxt: np.ndarray, j: int) -> np.ndarray:  # every move of ours still lets it out
        res = np.ones(shape, dtype=bool)
        for m in range(NUM_ACTIONS):
            target = region.chaser_moves[:, m]
            valid = target >= 0
            safe = np.where(valid, target, 0)
            caught = along(j, safe) == along(k, ar)
            res &= np.where(along(j, valid), np.take(nxt, safe, axis=j) & ~caught, True)
        return res

    def invader_step(nxt: np.ndarray) -> np.ndarray:  # some move of its gets it out
        res = np.zeros(shape, dtype=bool)
        for m in range(NUM_ACTIONS):
            target = region.invader_moves[:, m]
            safe = np.where(target >= 0, target, 0)
            hit = np.zeros(shape, dtype=bool)
            for j in range(k):
                hit |= along(j, ar) == along(k, safe)
            res |= np.where(along(k, target == ESCAPE), True,
                            np.where(along(k, target == BLOCKED), False, ~hit & np.take(nxt, safe, axis=k)))
        return res

    for _ in range(max_plies):
        new = []
        for p, role in enumerate(phases):
            if time.perf_counter() > deadline:
                return None
            nxt = out[(p + 1) % 4]
            new.append(nxt.copy() if role == "pass" else invader_step(nxt) if role == "inv"
                       else chaser_step(nxt, int(role[1])))
        if all((a == b).all() for a, b in zip(new, out)):
            break
        out = new
    return out
