"""The team brain inside the contest: one coordinator, shared memory, called on each of our agents' turns.

Rebuilds exactly the inputs used in training (pacman_train.features.team_view) from the contest GameState:
the belief filter follows pacman_jax.contest_obs.TeamObserver, and the team memory (last actions, intent maps)
follows pacman_train.core.Memory.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

from pacman_v1.pacman_coordinator.numpy_net import forward, load
from pacman_v1.pacman_jax import common
from pacman_v1.pacman_jax.common import NUM_AGENTS, STOP
from pacman_v1.pacman_jax.contest_obs import eaten_since, food_and_capsules, grid_array
from pacman_v1.pacman_train import features
from pacman_v1.pacman_train.static_maps import layout_maps

UNSEEN = -1000  # position used for enemies we cannot see (never visible, never on the board)


class Brain:
    def __init__(self, is_red: bool, weights: Path) -> None:
        self.team = 0 if is_red else 1
        self.params, self.meta = load(weights)
        assert self.meta["channels"] == list(features.CHANNELS), "weights were trained with other input channels"
        self.ready = False

    def start(self, game_state: Any) -> None:
        """Called once per game (by whichever of our agents registers first)."""
        if self.ready:
            return
        layout = game_state.data.layout
        self.width, self.height = layout.width, layout.height
        # weights trained on boards padded with walls (pad_to in the .json) see every board padded the same way
        pad_to = self.meta.get("pad_to") or (0, 0)
        self.shape = (max(self.width, pad_to[0]), max(self.height, pad_to[1]))
        self.walls = grid_array(layout.walls, self.shape, True)
        self.dist = common.maze_distances(self.walls)
        self.maps = layout_maps(self.walls, self.width, self.dist)
        self.total_food = int(np.array(layout.food.data, dtype=bool).sum())
        self.max_turns = game_state.data.timeleft
        self.enemies = common.enemies_of(self.team)
        shape = self.shape
        starts = [game_state.get_initial_agent_position(e) for e in self.enemies]
        # checkpoints trained with v2's respawn rule restart a contradicted belief at the enemy's start cell
        self.enemy_starts = np.array(starts, dtype=np.int64) if self.meta.get("respawn_belief") else None
        self.belief = np.stack([common.onehot_grid(np, np.array(s), shape) for s in starts]).astype(np.float32)
        self.last_turn = 0
        self.home = common.home_side(np, shape, self.width, self.height, self.team)
        self.snapshot = food_and_capsules(game_state, shape)
        self.last_action = np.full(NUM_AGENTS, STOP, dtype=np.int64)
        self.intent = np.zeros((NUM_AGENTS,) + shape, dtype=np.float32)
        self.ready = True
        # warm-up: first forward pass allocates and touches all weights inside the 15 s start-up allowance
        forward(self.params, self.meta, np.zeros(shape + (features.NUM_CHANNELS,), np.float32),
                np.zeros(features.NUM_VEC, np.float32), np.array([1, 1]))

    def update_belief(self, game_state: Any, agent: int, pos: np.ndarray, known: np.ndarray) -> None:
        turn = self.max_turns - game_state.data.timeleft
        starter = (agent - turn) % NUM_AGENTS
        for t in range(self.last_turn, turn):
            mover = (starter + t) % NUM_AGENTS
            if mover in self.enemies:
                slot = self.enemies.index(mover)
                self.belief[slot] = common.diffuse(np, self.belief[slot], self.walls)
        self.last_turn = turn
        readings = np.array([game_state.get_agent_distances()[e] for e in self.enemies], dtype=np.int64)
        members = list(common.team_members(self.team))
        eaten, eater, self.snapshot = eaten_since(game_state, self.shape, self.snapshot, self.home, agent, turn,
                                                  self.enemies)
        pacman = np.array([game_state.get_agent_state(e).is_pacman for e in self.enemies])
        self.belief = common.observe_beliefs(np, self.belief, self.walls, pos[agent], pos[members],
                                             pos[list(self.enemies)], known[list(self.enemies)], readings,
                                             self.home, pacman, eaten, eater, self.enemy_starts)

    def inputs(self, game_state: Any, agent: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(grid, vec, canonical cell, legal canonical actions) for `agent`, exactly as in training."""
        raw = [game_state.get_agent_position(i) for i in range(NUM_AGENTS)]
        known = np.array([p is not None for p in raw])
        pos = np.array([p if p is not None else (UNSEEN, UNSEEN) for p in raw], dtype=np.int64)
        belief_pos = np.where(known[:, None], pos, 0)
        self.update_belief(game_state, agent, belief_pos, known)
        states = [game_state.get_agent_state(i) for i in range(NUM_AGENTS)]
        capsules = np.zeros(self.shape, dtype=bool)
        for x, y in game_state.get_capsules():
            capsules[x, y] = True
        m = self.maps
        grid, vec = features.team_view(
            np, agent, self.walls, self.width, self.height, self.dist, m["home_dist"], m["boundary"],
            m["boundary_mask"], m["dead_end"], self.total_food, pos,
            np.array([s.is_pacman for s in states]), np.array([s.scared_timer for s in states], dtype=np.int64),
            np.array([s.num_carrying for s in states], dtype=np.int64),
            np.array([s.num_returned for s in states], dtype=np.int64),
            grid_array(game_state.data.food, self.shape, False), capsules, int(game_state.get_score()),
            int(game_state.data.timeleft), self.max_turns, self.belief, self.last_action, self.intent)
        if not self.meta.get("use_engineered_features", True):  # weights trained without the engineered inputs
            grid[..., slice(*features.ENGINEERED_CHANNELS)] = 0.0
            vec[slice(*features.ENGINEERED_VEC)] = 0.0
        cell = features.canonical_cell(np, pos[agent], self.team, self.width, self.height)
        legal = common.legal_world_actions(np, self.walls, pos[agent])
        if self.team == 1:
            legal = legal[common.FLIP_ACTION]
        return grid, vec, cell, legal

    def choose(self, game_state: Any, agent: int) -> str:
        """World action name for `agent` (greedy coordinator move)."""
        return self.decide(agent, *self.inputs(game_state, agent))

    def decide(self, agent: int, grid: np.ndarray, vec: np.ndarray, cell: np.ndarray, legal: np.ndarray) -> str:
        """Run the coordinator, update team memory (last action, intent map) and return the world action name."""
        logits, intent = self.think(grid, vec, cell)
        action = int(np.argmax(np.where(legal, logits, -np.inf)))
        self.remember(agent, action, intent)
        return common.ACTION_NAMES[self.to_world(action)]

    def think(self, grid: np.ndarray, vec: np.ndarray, cell: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Move logits [5] and intent probabilities [W,H] (walls excluded), both in the canonical frame."""
        logits, intent = forward(self.params, self.meta, grid, vec, cell)
        masked = np.where(grid[..., 0] > 0.5, -np.inf, intent)
        probs = np.exp(masked - masked.max())
        return logits, probs / probs.sum()

    def remember(self, agent: int, action: int, intent: np.ndarray) -> None:
        """Team memory after `agent` played canonical `action` with this intent map."""
        self.intent[agent] = intent
        self.last_action[agent] = action

    def to_world(self, action: int) -> int:
        """Canonical action index -> world action index (the canonical frame is flipped for blue)."""
        return action if self.team == 0 else int(common.FLIP_ACTION[action])

    def safe_choose(self, game_state: Any, agent: int) -> str:
        """choose(), but never crash the game: on any error log it and play a legal move."""
        try:
            return self.choose(game_state, agent)
        except Exception as error:  # noqa: BLE001 - an exception would forfeit the game
            print("coordinator error: %r" % (error,), file=sys.stderr)
            legal = game_state.get_legal_actions(agent)
            return legal[0] if legal else "Stop"
