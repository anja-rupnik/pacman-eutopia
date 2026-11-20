"""Coordinator input: the team's view of the board when one of its agents is about to act.

Everything here is numpy/jnp generic (`xp`) and takes the acting agent as a possibly traced integer, so the same
code builds observations inside jitted training and in the numpy-only contest agent.

Layout of the grid (canonical frame: the acting team always attacks rightward):
  0-12  base channels: pacman_jax.common.build_observation, with distances on the common DIST_SCALE
  13-34 engineered channels (EXTRA_CHANNELS)
  35    teammate's intent map (its own predicted position a few turns ahead, from team memory)
The critic additionally gets TRUTH_CHANNELS (true enemy positions) and CRITIC_FLAGS.

Enemy arrival times are soft minima over the belief, -tau log sum_p b(p) exp(-d(p, c) / tau): exact for a known
position, and close to the nearest believed cell (not the mean) when the belief is spread, so a band of possible
positions counts as dangerous where any of it is close.
Shipped with the contest agent, so it must stay numpy-only and Python 3.8 compatible.
"""
from __future__ import annotations

from pacman_jax import common
from pacman_jax.common import CARRY_SCALE, NUM_ACTIONS, NUM_AGENTS, SCARED_TIME, SCORE_SCALE

BASE_CHANNELS: tuple[str, ...] = (
    "walls", "home", "food_attack", "food_defend", "capsule_attack", "capsule_defend",
    "self", "mate", "enemy0_seen", "enemy1_seen", "enemy0_belief", "enemy1_belief", "dist_self")
EXTRA_CHANNELS: tuple[str, ...] = (
    "dist_mate",        # maze distance from the teammate
    "enemy0_dist",      # belief-weighted expected maze distance from enemy 0
    "enemy1_dist",
    "threat_dist",      # earliest time an enemy ghost that is no longer scared can be at the cell; only on the
                        #   enemy half (elsewhere it cannot kill us): min_k max(A_k, scared_k), A_k soft arrival
    "reach_self",       # threat_dist - dist_self: > 0 means I get there before any ghost
    "reach_mate",
    "escape_self",      # best over the exits b (enemy-side boundary cells with an open step home) of
                        #   threat_dist(b) - dist(c, b), minus dist_self(c): > 0 means I can reach the cell AND get out
                        #   before a ghost cuts me off (1 on our own half)
    "escape_mate",
    "hunted_self",      # on our half: my scared timer - earliest enemy arrival: > 0 means an enemy pacman can be
                        #   there while I am still scared (-1 elsewhere)
    "hunted_mate",
    "defend",           # earliest arrival of our agents that are not scared, minus earliest enemy arrival:
                        #   > 0 means an enemy gets there before any defender who could stop it
    "split",            # 1 if I am closer than the teammate, 0.5 on ties, 0 otherwise (division of labour)
    "home_dist",        # maze distance to our boundary column (static per layout)
    "dead_end",         # steps to the nearest cell on a cycle (static per layout): how deep a trap is
    "gate",             # 1 on cells that alone seal off a pocket, 0.5 on junctions sealing one in pairs (static)
    "seal_cost",        # cells we must hold to seal this cell off: 1 or 2 (/ 2), 0 if not sealable (static)
    "pocket_food",      # food inside the cell's pocket (single-gate area), on its cells and its gate
    "pocket_capsule",   # capsules inside the cell's pocket
    "trap",             # on gates: chance an enemy pacman is inside the pocket and we reach the gate first,
                        #   times what it would lose (1 + food it carries)
    "cutoff_self",      # on our half: chance the cell is on an enemy pacman's shortest way home and I (not scared on
                        #   arrival) get there first: one chases, the other cuts off
    "cutoff_mate",
    "capsule_race",     # on capsules: earliest enemy arrival - our earliest arrival: > 0 means we get there first
)
INTENT_CHANNELS: tuple[str, ...] = ("mate_intent",)
TRUTH_CHANNELS: tuple[str, ...] = ("enemy0_true", "enemy1_true")
CHANNELS: tuple[str, ...] = BASE_CHANNELS + EXTRA_CHANNELS + INTENT_CHANNELS
NUM_CHANNELS: int = len(CHANNELS)
NUM_CRITIC_CHANNELS: int = NUM_CHANNELS + len(TRUTH_CHANNELS)

BASE_VEC: tuple[str, ...] = (
    "self_scared", "self_carrying", "self_is_pacman", "mate_scared", "mate_carrying", "mate_is_pacman",
    "enemy0_scared", "enemy0_carrying", "enemy0_is_pacman", "enemy0_known",
    "enemy1_scared", "enemy1_carrying", "enemy1_is_pacman", "enemy1_known",
    "score", "time_left")
NUM_BASE_VEC: int = len(BASE_VEC)
EXTRA_VEC: tuple[str, ...] = (
    *(f"last_action_self_{i}" for i in range(NUM_ACTIONS)),
    *(f"last_action_mate_{i}" for i in range(NUM_ACTIONS)),
    "food_attack_left", "food_defend_left", "capsule_attack_left", "capsule_defend_left",
    "food_defend_open")  # our food outside areas we can seal: what the enemy can take however we defend
VEC_NAMES: tuple[str, ...] = BASE_VEC + EXTRA_VEC
NUM_VEC: int = len(VEC_NAMES)
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
COUNT_SCALE: float = 1.0 / 10.0  # food counts and trap values
SOFT_TAU: float = 2.0            # temperature (cells) of the soft-min enemy arrival
NO_PATH: float = 1.0e4
EARLIER: float = 0.5             # "we get there first" means at least this much earlier (keeps ties away from
                                 # float noise, so numpy and jnp agree)


def scaled_dist(xp, d):
    return xp.clip(d * DIST_SCALE, 0.0, DIST_CLIP)


def margin(xp, d):
    return xp.clip(d, -MARGIN, MARGIN) / MARGIN


def scaled_count(xp, v):
    return xp.clip(v * COUNT_SCALE, 0.0, DIST_CLIP)


def nearness(xp, dist):
    """exp(-d / tau) for every pair of cells: the kernel of the soft-min arrival (precomputed once per layout)."""
    return xp.exp(-dist.astype(xp.float32) / SOFT_TAU)


def segment_sum(xp, values, ids, n):
    """Sum of `values` per id in [0, n)."""
    if xp.__name__ == "numpy":
        return xp.bincount(ids, weights=values, minlength=n).astype(xp.float32)
    return xp.bincount(ids, weights=values, length=n)


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
    """common.build_observation with a traced agent index, distances on DIST_SCALE, and no constant vector entries
    (own agents are always known; the agent id is not used)."""
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
        belief[0], belief[1], scaled_dist(xp, dist_self)]
    grid = xp.stack([xp.asarray(p).astype(xp.float32) for p in planes], axis=-1)
    grid = common.canonicalize(xp, grid, team, width, height)
    head = []
    for j, i in enumerate(order):
        head += [scared[i] / SCARED_TIME, carrying[i] / CARRY_SCALE, is_pacman[i]] + ([known[i]] if j >= 2 else [])
    sign = 1 - 2 * team
    head += [sign * score / SCORE_SCALE, timeleft / max_turns]
    return grid, xp.stack([xp.asarray(v).astype(xp.float32) for v in head])


def team_view(xp, agent, walls, width, height, dist, near, home_dist, boundary, boundary_mask, dead_end,
              pocket_gate, pocket_paint, gate, seal_cost, total_food,
              pos, is_pacman, scared, carrying, food, capsules, score, timeleft, max_turns, belief, last_action,
              intent):
    """Grid [W, H, NUM_CHANNELS] and vec [NUM_VEC] for `agent`, in its team's canonical frame.

    dist: [W*H, W*H] maze distances of this layout; near: nearness(dist). home_dist: [2, W, H].
    boundary/boundary_mask: [2, H]. pocket_gate/pocket_paint [W*H], gate/seal_cost [W, H]: static_maps.pocket_maps.
    belief: [2, W, H] the team's belief over its two enemies. last_action: [4] each agent's last action in its
    own canonical frame. intent: [4, W, H] each agent's last intent map in its own canonical frame.
    """
    f32 = xp.float32
    team = agent % 2
    mate = (agent + 2) % NUM_AGENTS
    enemies = [1 - team, 3 - team]
    shape = (walls.shape[0], walls.shape[1])
    cells = shape[0] * shape[1]
    known = visibility(xp, pos, team)
    d_self = dist[cell_index(pos[agent], shape[1])].astype(f32).reshape(shape)
    d_mate = dist[cell_index(pos[mate], shape[1])].astype(f32).reshape(shape)
    grid, vec = base_observation(xp, agent, walls, width, height, d_self, food, capsules, pos, known, is_pacman,
                                 scared, carrying, score, timeleft, max_turns, belief)

    xs, ys = common.coords(xp, shape[0], shape[1])
    valid = (xs < width) & (ys < height)
    open_ = ~walls & valid
    home = common.home_side(xp, shape, width, height, team)
    away = valid & ~home
    b = belief.reshape(2, -1)
    expected = (b @ dist.astype(f32)).reshape((2,) + shape)
    arrive = (-SOFT_TAU * xp.log(xp.maximum(b @ near, 1e-30))).reshape((2,) + shape)
    first = xp.minimum(arrive[0], arrive[1])  # earliest arrival of any enemy

    ghost = [xp.maximum(arrive[k], scared[enemies[k]]) for k in range(2)]
    threat = xp.where(away, xp.minimum(ghost[0], ghost[1]), NO_PATH)
    exits = boundary_mask[1 - team] & boundary_mask[team]
    exit_cells = boundary[1 - team]
    to_exit = dist[:, exit_cells].astype(f32)
    best_exit = xp.where(exits[None, :], threat.reshape(-1)[exit_cells][None, :] - to_exit, -NO_PATH).max(-1)
    best_exit = xp.where(home, NO_PATH, best_exit.reshape(shape))

    guard_self = xp.where(scared[agent] > 0, NO_PATH, d_self)  # a scared ghost cannot stop a pacman
    guard_mate = xp.where(scared[mate] > 0, NO_PATH, d_mate)
    guard = xp.minimum(guard_self, guard_mate)

    pocket_cells = pocket_paint < cells
    food_f = food.reshape(-1).astype(f32)
    caps_f = capsules.reshape(-1).astype(f32)
    pocket_food = segment_sum(xp, food_f, pocket_gate, cells + 1)[pocket_paint] * pocket_cells
    pocket_caps = segment_sum(xp, caps_f, pocket_gate, cells + 1)[pocket_paint] * pocket_cells
    single_gate = gate.reshape(-1) == 1.0
    trap = 0.0
    cut_self = 0.0
    cut_mate = 0.0
    enemy_home = home_dist[1 - team].reshape(-1).astype(xp.int32)
    on_way_home = ((dist.astype(xp.int32) + enemy_home[None, :]) == enemy_home[:, None]).astype(f32)
    for k in range(2):
        e = enemies[k]
        mass = segment_sum(xp, b[k], pocket_gate, cells + 1)[:cells]
        first_to_gate = (guard.reshape(-1) + EARLIER < arrive[k].reshape(-1)).astype(f32)
        trap = trap + mass * first_to_gate * single_gate * is_pacman[e] * (1.0 + carrying[e])
        path = (b[k] @ on_way_home).reshape(shape) * is_pacman[e] * home
        cut_self = xp.maximum(cut_self, path * ((d_self + EARLIER < arrive[k]) & (scared[agent] <= d_self)))
        cut_mate = xp.maximum(cut_mate, path * ((d_mate + EARLIER < arrive[k]) & (scared[mate] <= d_mate)))
    open_food = (food & home & (seal_cost == 0)).sum()

    planes = [
        scaled_dist(xp, d_mate),
        scaled_dist(xp, expected[0]),
        scaled_dist(xp, expected[1]),
        scaled_dist(xp, threat),
        margin(xp, threat - d_self),
        margin(xp, threat - d_mate),
        margin(xp, best_exit - d_self),
        margin(xp, best_exit - d_mate),
        xp.where(home, margin(xp, scared[agent] - first), -1.0),
        xp.where(home, margin(xp, scared[mate] - first), -1.0),
        margin(xp, guard - first),
        0.5 + 0.5 * xp.sign(d_mate - d_self),
        scaled_dist(xp, home_dist[team].astype(f32)),
        xp.clip(dead_end.astype(f32) * DEAD_END_SCALE, 0.0, DIST_CLIP),
        gate,
        seal_cost.astype(f32) / 2.0,
        scaled_count(xp, pocket_food.reshape(shape)),
        pocket_caps.reshape(shape),
        scaled_count(xp, xp.asarray(trap).reshape(shape)),
        cut_self,
        cut_mate,
        xp.where(capsules, margin(xp, first - xp.minimum(d_self, d_mate)), 0.0),
    ]
    extra = xp.stack([xp.where(open_, p, 0.0).astype(f32) for p in planes], axis=-1)
    extra = common.canonicalize(xp, extra, team, width, height)
    grid = xp.concatenate([grid, extra, intent[mate][..., None].astype(f32)], axis=-1)

    half = xp.maximum(total_food / 2, 1.0)
    eye = xp.eye(NUM_ACTIONS, dtype=f32)
    extra_vec = [
        eye[last_action[agent]], eye[last_action[mate]],
        xp.stack([
            grid[..., 2].sum() / half, grid[..., 3].sum() / half, grid[..., 4].sum() / 2.0, grid[..., 5].sum() / 2.0,
            open_food / half,
        ]).astype(f32),
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
