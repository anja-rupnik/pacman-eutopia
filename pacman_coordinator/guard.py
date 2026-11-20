"""Single-entrance guard: on a layout where one cell cuts all of our food and capsules off from the border, one of
our agents stands on that cell while we lead. An enemy pacman can't pass a ghost in a one-cell corridor, and with
every capsule behind the guard it can't scare it either, so a lead can't be lost; the coordinator keeps playing the
other agent. Until we lead both agents play (guarding from the start turned too many wins into 0-0: on RANDOM129 and
RANDOM160 vs all opponents, guarding while ahead scored 0.894 / 0.932 against 0.860 / 0.924 for never or always). Plain Python (no numpy), contest coordinates (x, y). Python 3.8 compatible.
"""
from __future__ import annotations

from collections import deque
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

Cell = Tuple[int, int]
STEPS: Dict[str, Cell] = {"North": (0, 1), "South": (0, -1), "East": (1, 0), "West": (-1, 0)}


def _open(walls: Any, cell: Cell) -> bool:
    x, y = cell
    return 0 <= x < walls.width and 0 <= y < walls.height and not walls[x][y]


def _neighbours(walls: Any, cell: Cell) -> Iterable[Cell]:
    for dx, dy in STEPS.values():
        n = (cell[0] + dx, cell[1] + dy)
        if _open(walls, n):
            yield n


def guard_cell(walls: Any, is_red: bool, targets: Set[Cell]) -> Optional[Cell]:
    """The cell on our side that alone cuts every target (our food and capsules) off from the border cells enemies
    enter through, or None. Of several, the one leaving enemies the least room in front of it."""
    mid = walls.width // 2
    side = range(0, mid) if is_red else range(mid, walls.width)
    border, outside = (mid - 1, mid) if is_red else (mid, mid - 1)
    entries = [(border, y) for y in range(walls.height) if _open(walls, (border, y)) and _open(walls, (outside, y))]
    cells = [(x, y) for x in side for y in range(walls.height) if _open(walls, (x, y))]
    if not entries or not targets:
        return None
    best: Optional[Tuple[int, Cell]] = None
    for block in cells:
        seen = {e for e in entries if e != block}
        queue = deque(seen)
        while queue:
            for n in _neighbours(walls, queue.popleft()):
                if n[0] in side and n != block and n not in seen:
                    seen.add(n)
                    queue.append(n)
        if not (targets - {block}) & seen and (best is None or len(seen) < best[0]):
            best = (len(seen), block)
    return best[1] if best else None


def step_towards(walls: Any, start: Cell, goal: Cell) -> str:
    """First move of a shortest path from start to goal ("Stop" when already there or unreachable)."""
    if start == goal:
        return "Stop"
    first: Dict[Cell, str] = {}
    queue = deque()
    for name, (dx, dy) in STEPS.items():
        n = (start[0] + dx, start[1] + dy)
        if _open(walls, n) and n not in first:
            first[n] = name
            queue.append(n)
    while queue:
        cell = queue.popleft()
        if cell == goal:
            return first[cell]
        for n in _neighbours(walls, cell):
            if n not in first and n != start:
                first[n] = first[cell]
                queue.append(n)
    return "Stop"


class Guard:
    """Decides once per game (start) whether one of our agents guards, and which; then moves it (act)."""

    def __init__(self, is_red: bool, members: List[int]) -> None:
        self.is_red = is_red
        self.members = members
        self.cell: Optional[Cell] = None
        self.agent: Optional[int] = None
        self.ready = False

    def start(self, game_state: Any) -> None:
        if self.ready:
            return
        self.ready = True
        walls = game_state.get_walls()
        food = game_state.get_red_food() if self.is_red else game_state.get_blue_food()
        capsules = game_state.get_red_capsules() if self.is_red else game_state.get_blue_capsules()
        targets = {(x, y) for x in range(food.width) for y in range(food.height) if food[x][y]}
        targets |= {(int(x), int(y)) for x, y in capsules}
        self.walls = walls
        self.cell = guard_cell(walls, self.is_red, targets)
        if self.cell is not None:  # the agent that gets there first
            self.agent = min(self.members, key=lambda i: (self.distance(game_state.get_agent_position(i)), i))

    def distance(self, start: Cell) -> int:
        dist = {start: 0}
        queue = deque([start])
        while queue:
            cell = queue.popleft()
            if cell == self.cell:
                return dist[cell]
            for n in _neighbours(self.walls, cell):
                if n not in dist:
                    dist[n] = dist[cell] + 1
                    queue.append(n)
        return 10 ** 9

    def act(self, game_state: Any, index: int) -> Optional[str]:
        """The guard's move, or None to let the coordinator play this agent (not the guard, we don't lead, or the
        guard is scared: an enemy got past it to a capsule before it arrived)."""
        if index != self.agent or game_state.get_agent_state(index).scared_timer > 0:
            return None
        score = game_state.get_score() * (1 if self.is_red else -1)
        if score <= 0:  # until we lead, both agents attack (one attacker alone turns too many wins into 0-0)
            return None
        pos = game_state.get_agent_position(index)
        move = step_towards(self.walls, (int(pos[0]), int(pos[1])), self.cell)
        return move if move in game_state.get_legal_actions(index) else None
