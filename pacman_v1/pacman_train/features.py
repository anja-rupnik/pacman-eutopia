"""Coordinator input: the team's view of the board when one of its agents is about to act.

Everything here is numpy/jnp generic (`xp`) and takes the acting agent as a possibly traced integer, so the same
code builds observations inside jitted training and in the numpy-only contest agent.

Layout of the grid (canonical frame: the acting team always attacks rightward):
  0-12  base channels, identical to pacman_jax.common.build_observation
  13-24 engineered channels (EXTRA_CHANNELS)
  25    teammate's intent map (its own predicted position a few turns ahead, from team memory)
The critic additionally gets TRUTH_CHANNELS (true enemy positions) and CRITIC_FLAGS.
Shipped with the contest agent, so it must stay numpy-only and Python 3.8 compatible.
"""
from __future__ import annotations

from pacman_v1.pacman_jax import common
from pacman_v1.pacman_jax.common import CARRY_SCALE, NUM_ACTIONS, NUM_AGENTS, SCARED_TIME, SCORE_SCALE, UNREACHABLE

BASE_CHANNELS: tuple[str, ...] = (
    "walls", "home", "food_attack", "food_defend", "capsule_attack", "capsule_defend",
    "self", "mate", "enemy0_seen", "enemy1_seen", "enemy0_belief", "enemy1_belief", "dist_self")
EXTRA_CHANNELS: tuple[str, ...] = (
    "dist_mate",        # maze distance from the teammate
    "enemy0_dist",      # belief-weighted expected maze distance from enemy 0
    "enemy1_dist",
    "threat_dist",      # earliest time a non-scared enemy can be at the cell: min_k max(E_k, scared_k)
    "reach_self",       # threat_dist - dist_self: > 0 means I get there before any ghost
    "reach_mate",
    "escape_self",      # best over our boundary cells b of threat_dist(b) - dist(c, b), minus dist_self(c):
                        #   > 0 means I can reach the cell AND bring it home before a ghost cuts me off
    "escape_mate",
    "defend",           # min(dist_self, dist_mate) - min_k E_k: > 0 means an enemy is closer than both of us
    "split",            # 1 if I am closer than the teammate, 0.5 on ties, 0 otherwise (division of labour)
    "home_dist",        # maze distance to our boundary column (static per layout)
    "dead_end",         # steps to the nearest cell on a cycle (static per layout): how deep a trap is
)
INTENT_CHANNELS: tuple[str, ...] = ("mate_intent",)
TRUTH_CHANNELS: tuple[str, ...] = ("enemy0_true", "enemy1_true")
CHANNELS: tuple[str, ...] = BASE_CHANNELS + EXTRA_CHANNELS + INTENT_CHANNELS
NUM_CHANNELS: int = len(CHANNELS)
NUM_CRITIC_CHANNELS: int = NUM_CHANNELS + len(TRUTH_CHANNELS)

NUM_BASE_VEC: int = 22
EXTRA_VEC: tuple[str, ...] = (
    *(f"last_action_self_{i}" for i in range(NUM_ACTIONS)),
    *(f"last_action_mate_{i}" for i in range(NUM_ACTIONS)),
    "home_dist_self", "home_dist_mate", "threat_dist_self", "threat_dist_mate", "escape_self", "escape_mate",
    "food_attack_left", "food_defend_left", "capsule_attack_left", "capsule_defend_left",
    "returned_team", "returned_enemy")
NUM_VEC: int = NUM_BASE_VEC + len(EXTRA_VEC)
CRITIC_FLAGS: tuple[str, ...] = ("mover_is_learner", "other_is_learner", "vs_self", "vs_pool", "vs_baseline")
NUM_CRITIC_VEC: int = NUM_VEC + len(CRITIC_FLAGS)

# Inputs switched off when a run is trained with use_engineered_features=False ("no tools"):
# everything beyond the base observation (engineered channels, teammate intent, extra vector entries).
ENGINEERED_CHANNELS: tuple[int, int] = (len(BASE_CHANNELS), NUM_CHANNELS)
ENGINEERED_VEC: tuple[int, int] = (NUM_BASE_VEC, NUM_VEC)

DIST_SCALE: float = 1.0 / 40.0
DIST_CLIP: float = 3.0
MARGIN: float = 20.0
DEAD_END_SCALE: float = 1.0 / 10.0
NO_PATH: float = 1.0e4


def scaled_dist(xp, d):
    return xp.clip(d * DIST_SCALE, 0.0, DIST_CLIP)


def margin(xp, d):
    return xp.clip(d, -MARGIN, MARGIN) / MARGIN


def cell_index(pos, height):
    return pos[0] * height + pos[1]


def canonical_cell(xp, pos, team, width, height):
    """(x, y) of a world position in the acting team's canonical frame."""
    flip = team == 1
    return xp.stack([xp.where(flip, width - 1 - pos[0], pos[0]), xp.where(flip, height - 1 - pos[1], pos[1])])


def visibility(xp, pos, team):
    """known[i]: own agents always, enemies when within sight of either teammate."""
    visible = common.enemy_visibility(xp, pos, team)
    idx = xp.arange(NUM_AGENTS)
    return (idx % 2 == team) | visible[idx // 2]


def base_observation(xp, agent, walls, width, height, dist_self, food, capsules, pos, known, is_pacman, scared,
                     carrying, score, timeleft, max_turns, belief):
    """common.build_observation with a traced agent index (no Python branching on the agent)."""
    team = agent % 2
    mate = (agent + 2) % NUM_AGENTS
    order = [agent, mate, 1 - team, 3 - team]
    shape = (walls.shape[0], walls.shape[1])
    xs, ys = common.coords(xp, shape[0], shape[1])
    valid = (xs < width) & (ys < height)
    red_side = valid & (xs < width // 2)
    home = xp.where(team == 0, red_side, valid & ~red_side)
    away = valid & ~home
    positions = [common.onehot_grid(xp, pos[i], shape) & known[i] for i in order]
    planes = [walls, home, food & away, food & home, capsules & away, capsules & home] + positions + [
        belief[0], belief[1], dist_self / UNREACHABLE]
    grid = xp.stack([xp.asarray(p).astype(xp.float32) for p in planes], axis=-1)
    grid = common.canonicalize(xp, grid, team, width, height)
    per_agent = [[scared[i] / SCARED_TIME, carrying[i] / CARRY_SCALE, is_pacman[i], known[i]] for i in order]
    sign = 1 - 2 * team
    head = [v for row in per_agent for v in row] + [sign * score / SCORE_SCALE, timeleft / max_turns]
    vec = xp.concatenate([xp.stack([xp.asarray(v).astype(xp.float32) for v in head]),
                          (xp.arange(NUM_AGENTS) == agent).astype(xp.float32)])
    return grid, vec


def team_view(xp, agent, walls, width, height, dist, home_dist, boundary, boundary_mask, dead_end, total_food,
              pos, is_pacman, scared, carrying, returned, food, capsules, score, timeleft, max_turns, belief,
              last_action, intent):
    """Grid [W, H, NUM_CHANNELS] and vec [NUM_VEC] for `agent`, in its team's canonical frame.

    dist: [W*H, W*H] maze distances of this layout. home_dist: [2, W, H]. boundary/boundary_mask: [2, H].
    belief: [2, W, H] the team's belief over its two enemies. last_action: [4] each agent's last action in its
    own canonical frame. intent: [4, W, H] each agent's last intent map in its own canonical frame.
    """
    team = agent % 2
    mate = (agent + 2) % NUM_AGENTS
    enemies = [1 - team, 3 - team]
    shape = (walls.shape[0], walls.shape[1])
    height_cells = shape[1]
    known = visibility(xp, pos, team)
    d_self = dist[cell_index(pos[agent], height_cells)].astype(xp.float32).reshape(shape)
    d_mate = dist[cell_index(pos[mate], height_cells)].astype(xp.float32).reshape(shape)
    grid, vec = base_observation(xp, agent, walls, width, height, d_self, food, capsules, pos, known, is_pacman,
                                 scared, carrying, score, timeleft, max_turns, belief)

    xs, ys = common.coords(xp, shape[0], shape[1])
    open_ = ~walls & (xs < width) & (ys < height)
    expected = (belief.reshape(2, -1) @ dist.astype(xp.float32)).reshape((2,) + shape)
    threat = xp.minimum(xp.maximum(expected[0], scared[enemies[0]]), xp.maximum(expected[1], scared[enemies[1]]))
    cells = boundary[team]
    threat_b = threat.reshape(-1)[cells]
    to_boundary = dist[:, cells].astype(xp.float32)
    best_exit = xp.where(boundary_mask[team][None, :], threat_b[None, :] - to_boundary, -NO_PATH).max(-1).reshape(shape)
    home_d = home_dist[team].astype(xp.float32)
    planes = [
        d_mate / UNREACHABLE,
        scaled_dist(xp, expected[0]),
        scaled_dist(xp, expected[1]),
        scaled_dist(xp, threat),
        margin(xp, threat - d_self),
        margin(xp, threat - d_mate),
        margin(xp, best_exit - d_self),
        margin(xp, best_exit - d_mate),
        margin(xp, xp.minimum(d_self, d_mate) - xp.minimum(expected[0], expected[1])),
        0.5 + 0.5 * xp.sign(d_mate - d_self),
        scaled_dist(xp, home_d),
        xp.clip(dead_end.astype(xp.float32) * DEAD_END_SCALE, 0.0, DIST_CLIP),
    ]
    extra = xp.stack([xp.where(open_, p, 0.0).astype(xp.float32) for p in planes], axis=-1)
    extra = common.canonicalize(xp, extra, team, width, height)
    grid = xp.concatenate([grid, extra, intent[mate][..., None].astype(xp.float32)], axis=-1)

    def at(plane, i):
        return plane[pos[i][0], pos[i][1]]

    half = xp.maximum(total_food / 2, 1.0)
    food_attack = grid[..., 2].sum()
    food_defend = grid[..., 3].sum()
    caps_attack = grid[..., 4].sum()
    caps_defend = grid[..., 5].sum()
    returned_team = returned[team] + returned[team + 2]
    returned_enemy = returned[1 - team] + returned[3 - team]
    eye = xp.eye(NUM_ACTIONS, dtype=xp.float32)
    extra_vec = [
        eye[last_action[agent]], eye[last_action[mate]],
        xp.stack([
            scaled_dist(xp, at(home_d, agent)), scaled_dist(xp, at(home_d, mate)),
            scaled_dist(xp, at(threat, agent)), scaled_dist(xp, at(threat, mate)),
            margin(xp, at(best_exit, agent)), margin(xp, at(best_exit, mate)),
            food_attack / half, food_defend / half, caps_attack / 2.0, caps_defend / 2.0,
            returned_team / half, returned_enemy / half,
        ]).astype(xp.float32),
    ]
    vec = xp.concatenate([vec] + extra_vec)
    return grid, vec


def truth_planes(xp, agent, pos, walls, width, height):
    """True enemy positions (critic only), canonical frame."""
    team = agent % 2
    shape = (walls.shape[0], walls.shape[1])
    planes = [common.onehot_grid(xp, pos[1 - team], shape), common.onehot_grid(xp, pos[3 - team], shape)]
    grid = xp.stack([xp.asarray(p).astype(xp.float32) for p in planes], axis=-1)
    return common.canonicalize(xp, grid, team, width, height)
