"""v3: the v2 coordinator plus a ghost-chase search that takes over only for proven captures.

A v3 checkpoint is a v2 one whose .json has a "search" entry (pacman_coordinator.versions picks this brain). On each
of our turns the network decides as in v2; then, if this agent is a ghost on our side and an enemy pacman is visible
there, pacman_coordinator.chase looks for a capture the invader cannot avoid. When one exists (and it is safe, see
`chase`) the agent plays a move of the shortest forced capture, the network's own move whenever that is one of them.
Team memory (last action, intent) records the move actually played, as the network expects. Python 3.8 + numpy.
"""
from __future__ import annotations

import time
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import numpy as np

from pacman_coordinator.brain import Brain
from pacman_coordinator.chase import Region, escape_map, solve
from pacman_jax import common
from pacman_jax.common import ACTION_NAMES, FLIP_ACTION, NUM_AGENTS, STOP, UNREACHABLE

DEFAULTS: dict = {
    "max_cells_one": 150,  # region size for a chase with one ghost (states: cells^2)
    "max_cells_two": 60,   # ... with both ghosts (cells^3)
    "max_plies": 48,       # 12 moves of ours
    # seconds per move for everything (belief, network and search), measured on the machine that runs the game, so
    # a slower CPU just searches less: the contest allows 1 s; the network takes ~0.04 s here (Python 3.8, 1 thread)
    "budget": 0.4,
    "min_search": 0.02,    # don't start a search with less time than this left
    "two_ghosts": True,
    # containment: when no capture can be proved, a ghost that can keep a visible invader carrying food from ever
    # getting home plays only moves that keep it so (the network's move whenever it does)
    "contain": True,
    "contain_cells": 80,   # region of the containment map (states: cells^2, solved to a fixpoint)
    "contain_carry": 1,    # only invaders carrying at least this much food
}


class SearchBrain(Brain):
    def __init__(self, is_red: bool, weights: Path) -> None:
        super().__init__(is_red, weights)
        self.search = dict(DEFAULTS, **(self.meta.get("search") or {}))
        self.stats: Counter = Counter()
        self.containing: dict = {}  # enemy -> (our agent containing it, centre of its region) while it holds

    def start(self, game_state: Any) -> None:
        if self.ready:
            return
        super().start(game_state)
        w, h = self.shape
        cells = (self.home & ~self.walls).reshape(-1)
        # maze distance from every cell to our side (UNREACHABLE where cut off)
        self.to_home = self.dist[:, cells].min(1).reshape(w, h) if cells.any() else np.full((w, h), UNREACHABLE)

    def choose(self, game_state: Any, agent: int) -> str:
        move_start = time.perf_counter()
        grid, vec, cell, legal = self.inputs(game_state, agent)
        logits, intent = self.think(grid, vec, cell)
        action = int(np.argmax(np.where(legal, logits, -np.inf)))
        forced = self.chase(game_state, agent, self.to_world(action), move_start)
        if forced is not None:
            action = forced if self.team == 0 else int(FLIP_ACTION[forced])  # world -> canonical (its own inverse)
        self.remember(agent, action, intent)
        return ACTION_NAMES[self.to_world(action)]

    # ------------------------------------------------------------------ the search
    def reach(self, slot: int, known: bool, target: np.ndarray) -> int:
        """Fewest moves the enemy in `slot` needs to reach any `target` cell: from its position when known,
        otherwise from every cell its belief allows."""
        if not target.any():
            return UNREACHABLE
        w, h = self.shape
        if known:
            enemy = self.enemies[slot]
            starts = np.array([self.pos[enemy][0] * h + self.pos[enemy][1]])
        else:
            starts = np.flatnonzero(self.belief[slot].reshape(-1) > 1e-6)
            if not len(starts):
                return 0
        return int(self.dist[starts][:, target.reshape(-1)].min())

    def chase(self, game_state: Any, agent: int, net_move: int, move_start: Optional[float] = None) -> Optional[int]:
        """World move of a proven, safe capture for `agent`, or None (the network decides). The search stops at
        `move_start` (default: now) + budget."""
        start = time.perf_counter()
        deadline = (start if move_start is None else move_start) + self.search["budget"]
        if deadline - start < self.search["min_search"]:
            self.stats["no_time"] += 1
            return None
        states = [game_state.get_agent_state(i) for i in range(NUM_AGENTS)]
        me = states[agent]
        if me.is_pacman or me.scared_timer > 0:
            return None
        known = [game_state.get_agent_position(i) is not None for i in range(NUM_AGENTS)]
        mate = (agent + 2) % NUM_AGENTS
        mate_free = not states[mate].is_pacman and states[mate].scared_timer == 0
        invaders = [e for e in self.enemies if known[e] and states[e].is_pacman and self.home[tuple(self.pos[e])]]
        if not invaders and not (self.search["contain"] and self.containing):
            return None
        self.stats["chances"] += 1
        capsules = np.zeros(self.shape, dtype=bool)
        for x, y in game_state.get_capsules():
            capsules[x, y] = True
        capsules &= self.home  # ours: an enemy that eats one scares our ghosts
        h = self.shape[1]
        here = self.pos[agent]

        def d(a, b) -> int:
            return int(self.dist[a[0] * h + a[1], b[0] * h + b[1]])

        order = [(agent + t) % NUM_AGENTS for t in range(NUM_AGENTS)]
        for inv in sorted(invaders, key=lambda e: d(here, self.pos[e])):
            other = [e for e in self.enemies if e != inv][0]
            slot = self.enemies.index(other)
            other_invading = states[other].is_pacman
            # two invaders: when my teammate is nearer this one, it is the natural chaser and I may be the one
            # the network keeps for the other; leave this one to the network
            if other_invading and mate_free and d(self.pos[mate], self.pos[inv]) < d(here, self.pos[inv]):
                self.stats["left_to_mate"] += 1
                continue
            result, two = None, False
            region = Region(self.walls, self.home, self.dist, capsules, self.pos[inv], self.search["max_cells_one"])
            if region.cell(here) >= 0:
                self.stats["tried_one"] += 1
                roles = ["c0" if a == agent else "inv" if a == inv else "pass" for a in order]
                result = solve(region, [region.cell(here)], region.cell(self.pos[inv]), roles,
                               self.search["max_plies"], deadline)
            if (result is None and self.search["two_ghosts"] and mate_free and not other_invading
                    and time.perf_counter() < deadline):
                region = Region(self.walls, self.home, self.dist, capsules, self.pos[inv],
                                self.search["max_cells_two"])
                cells = [region.cell(here), region.cell(self.pos[mate])]
                if min(cells) >= 0:
                    self.stats["tried_two"] += 1
                    roles = ["c0" if a == agent else "c1" if a == mate else "inv" if a == inv else "pass"
                             for a in order]
                    result = solve(region, cells, region.cell(self.pos[inv]), roles, self.search["max_plies"],
                                   deadline)
                    two = result is not None
            spent = time.perf_counter() - start
            self.stats["max_search_ms"] = max(self.stats["max_search_ms"], int(spent * 1000))
            if result is None:
                continue
            plies, moves = result
            rounds = plies // NUM_AGENTS + 1  # moves each agent makes before the capture
            # never open a door elsewhere: the other enemy must not reach one of our capsules first, and when
            # nobody of ours is left free to defend (both chase, or my teammate is away), not even our side
            if self.reach(slot, known[other], capsules) <= rounds:
                self.stats["unsafe_capsule"] += 1
                continue
            if (two or not mate_free) and not other_invading and self.reach(slot, known[other], self.home) <= rounds:
                self.stats["unsafe_side"] += 1
                continue
            self.stats["two" if two else "one"] += 1
            if net_move in moves:
                self.stats["agree"] += 1
                return net_move
            self.stats["override"] += 1
            moving = sorted(m for m in moves if m != STOP)
            return moving[0] if moving else STOP
        if self.search["contain"]:
            return self.contain(states, agent, invaders, mate_free, capsules, net_move, known, deadline)
        return None

    def contain(self, states, agent, invaders, mate_free, capsules, net_move, known, deadline) -> Optional[int]:
        """World move that keeps an invader from ever getting home (with this ghost alone), or None.

        A visible invader is taken at its position. One this ghost was containing that went out of sight stays
        contained only if every cell its belief allows (on our side) is: the move must keep all of them sealed."""
        h = self.shape[1]
        here, mate = self.pos[agent], (agent + 2) % NUM_AGENTS

        def d(a, b) -> int:
            return int(self.dist[a[0] * h + a[1], b[0] * h + b[1]])

        targets = []  # (enemy, centre of the region, possible cells)
        for e in sorted(invaders, key=lambda e: d(here, self.pos[e])):
            targets.append((e, tuple(self.pos[e]), [tuple(self.pos[e])]))
        for e, (who, last) in list(self.containing.items()):
            if not states[e].is_pacman:
                del self.containing[e]
            elif who == agent and not known[e]:
                cells = [tuple(c) for c in np.argwhere((self.belief[self.enemies.index(e)] > 1e-6) & self.home)]
                if cells:
                    targets.append((e, last, cells))
        order = [(agent + t) % NUM_AGENTS for t in range(NUM_AGENTS)]
        for inv, centre, cells in targets:
            hidden = len(cells) > 1 or not known[inv]
            if time.perf_counter() > deadline:
                break
            if not hidden and states[inv].num_carrying < self.search["contain_carry"]:
                continue
            other = [e for e in self.enemies if e != inv][0]
            if states[other].is_pacman and (not mate_free or (known[inv] and
                                                               d(self.pos[mate], centre) < d(here, centre))):
                continue  # two invaders: I may be the one needed for the other
            if self.reach(self.enemies.index(other), known[other], capsules) <= 2:
                continue  # our ghosts are about to be scared anyway
            region = Region(self.walls, self.home, self.dist, capsules, centre, self.search["contain_cells"])
            mine = region.cell(here)
            theirs = [region.cell(c) for c in cells]
            if mine < 0 or min(theirs) < 0:
                continue  # it may be outside the region: no guarantee
            self.stats["contain_tried_hidden" if hidden else "contain_tried"] += 1
            roles = ["c0" if a == agent else "inv" if a == inv else "pass" for a in order]
            out = escape_map(region, 1, roles, deadline=deadline)
            if out is None:
                self.stats["contain_timeout"] += 1
                continue
            theirs = np.array(theirs)
            if out[0][mine, theirs].any():
                self.containing.pop(inv, None)
                continue  # it can (or may) get home whatever I do
            keep = []
            for m in range(len(ACTION_NAMES)):
                target = int(region.chaser_moves[mine, m])
                if target >= 0 and not (out[1][target, theirs] & (theirs != target)).any():
                    keep.append(m)
            if not keep:
                continue
            self.containing[inv] = (agent, centre)
            self.stats["contain_hidden" if hidden else "contain"] += 1
            if net_move in keep:
                self.stats["contain_agree"] += 1
                return net_move
            self.stats["contain_override"] += 1
            return min(keep, key=lambda m: (d((int(region.xs[region.chaser_moves[mine, m]]),
                                                int(region.ys[region.chaser_moves[mine, m]])), centre), m == STOP))
        return None
