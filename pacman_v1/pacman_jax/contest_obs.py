from __future__ import annotations  # shipped with contest agents, which run on Python 3.8

from typing import Any

import numpy as np

from pacman_v1.pacman_jax import common
from pacman_v1.pacman_jax.common import NUM_AGENTS


def grid_array(grid: Any, shape: tuple[int, int], fill: bool) -> np.ndarray:
    out = np.full(shape, fill, dtype=bool)
    data = np.array(grid.data, dtype=bool)
    out[: data.shape[0], : data.shape[1]] = data
    return out


def food_and_capsules(game_state: Any, shape: tuple[int, int]) -> np.ndarray:
    out = grid_array(game_state.data.food, shape, False)
    for x, y in game_state.get_capsules():
        out[x, y] = True
    return out


def eaten_since(game_state: Any, shape: tuple[int, int], snapshot: np.ndarray, home: np.ndarray, agent: int,
                turn: int, enemies: tuple[int, int]) -> tuple[np.ndarray, int, np.ndarray]:
    """Our food/capsules that vanished since our team's last look (exactly one enemy moved in between, and only an
    enemy can eat on our half), that enemy's slot (-1 before any move), and the new snapshot."""
    now = food_and_capsules(game_state, shape)
    eater = enemies.index((agent - 1) % NUM_AGENTS) if turn > 0 else -1
    return snapshot & ~now & home, eater, now


class TeamObserver:
    def __init__(self, game_state: Any, shape: tuple[int, int], is_red: bool) -> None:
        layout = game_state.data.layout
        assert layout.width <= shape[0] and layout.height <= shape[1], (layout.width, layout.height, shape)
        self.shape = shape
        self.team = 0 if is_red else 1
        self.width = layout.width
        self.height = layout.height
        self.walls = grid_array(layout.walls, shape, True)
        self.dist = common.maze_distances(self.walls)
        self.max_turns = game_state.data.timeleft
        self.enemies = common.enemies_of(self.team)
        starts = [game_state.get_initial_agent_position(e) for e in self.enemies]
        self.belief = np.stack([common.onehot_grid(np, np.array(s), shape) for s in starts]).astype(np.float32)
        self.last_turn = 0
        self.home = common.home_side(np, shape, self.width, self.height, self.team)
        self.snapshot = food_and_capsules(game_state, shape)

    def observe(self, game_state: Any, agent: int) -> tuple[np.ndarray, np.ndarray]:
        assert common.team_of(agent) == self.team
        turn = self.max_turns - game_state.data.timeleft
        starter = (agent - turn) % NUM_AGENTS
        for t in range(self.last_turn, turn):
            mover = (starter + t) % NUM_AGENTS
            if mover in self.enemies:
                slot = self.enemies.index(mover)
                self.belief[slot] = common.diffuse(np, self.belief[slot], self.walls)
        self.last_turn = turn
        raw = [game_state.get_agent_position(i) for i in range(NUM_AGENTS)]
        known = np.array([p is not None for p in raw])
        assert all(known[i] for i in common.team_members(self.team))
        pos = np.array([p if p is not None else (0, 0) for p in raw], dtype=np.int64)
        readings = np.array([game_state.get_agent_distances()[e] for e in self.enemies], dtype=np.int64)
        states = [game_state.get_agent_state(i) for i in range(NUM_AGENTS)]
        eaten, eater, self.snapshot = eaten_since(game_state, self.shape, self.snapshot, self.home, agent, turn,
                                                  self.enemies)
        self.belief = common.observe_beliefs(np, self.belief, self.walls, pos[agent],
                                             pos[list(common.team_members(self.team))], pos[list(self.enemies)],
                                             known[list(self.enemies)], readings, self.home,
                                             np.array([states[e].is_pacman for e in self.enemies]), eaten, eater)
        assert np.allclose(self.belief.sum((1, 2)), 1.0, atol=1e-4), self.belief.sum((1, 2))
        capsules = np.zeros(self.shape, dtype=bool)
        for x, y in game_state.get_capsules():
            capsules[x, y] = True
        dist_self = self.dist[pos[agent, 0] * self.shape[1] + pos[agent, 1]].reshape(self.shape).astype(np.float32)
        return common.build_observation(
            np, agent, self.walls, self.width, self.height, dist_self,
            grid_array(game_state.data.food, self.shape, False), capsules, pos, known,
            np.array([s.is_pacman for s in states]), np.array([s.scared_timer for s in states], dtype=np.int32),
            np.array([s.num_carrying for s in states], dtype=np.int32), np.int32(game_state.get_score()),
            np.int32(game_state.data.timeleft), self.max_turns, self.belief)

    def legal_canonical(self, game_state: Any, agent: int) -> np.ndarray:
        pos = np.array(game_state.get_agent_position(agent), dtype=np.int64)
        legal = common.legal_world_actions(np, self.walls, pos)
        return legal[common.FLIP_ACTION] if self.team == 1 else legal
