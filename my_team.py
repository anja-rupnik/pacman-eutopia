"""Contest entry point (copied to the submission root by pacman_coordinator.bundle)."""
from __future__ import annotations

import ctypes
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):  # one core per game on the cluster
    os.environ[_var] = "1"  # only effective if numpy is not imported yet: see pin_blas_threads

import sys  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

from contest.capture_agents import CaptureAgent  # noqa: E402

from pacman_coordinator.guard import Guard  # noqa: E402 (plain Python)

WEIGHTS = Path(__file__).resolve().parent / "coordinator.npz"
VENDOR = Path(__file__).resolve().parent / "_vendor"  # numpy shipped with the submission (bundle --vendor-numpy)


def import_numpy() -> bool:
    """numpy from the machine if it has one, else the copy in _vendor (appended last, so it never shadows theirs)."""
    try:
        import numpy  # noqa: F401
        return True
    except ImportError:
        pass
    if VENDOR.is_dir() and str(VENDOR) not in sys.path:
        sys.path.append(str(VENDOR))
    try:
        import numpy  # noqa: F401,F811
        return True
    except ImportError:  # e.g. another Python version than the shipped build
        return False


def pin_blas_threads() -> None:
    """Cap numpy's OpenBLAS at one thread even if numpy was imported before us (the contest loads every team in one
    process, so another team or the engine may have started it with one thread per core). On a one-core slot
    extra BLAS threads only fight over the core, which shows up as move-time spikes. Never raises."""
    try:
        import numpy as np
        root = Path(np.__file__).resolve().parent
        for lib in sorted((root.parent / "numpy.libs").glob("*openblas*")) + sorted((root / ".libs").glob("*openblas*")):
            handle = ctypes.CDLL(str(lib))  # the already loaded library: dlopen returns the same handle
            for name in ("openblas_set_num_threads64_", "openblas_set_num_threads", "scipy_openblas_set_num_threads64_"):
                if hasattr(handle, name):
                    getattr(handle, name)(1)
                    return
    except Exception:
        pass


def create_team(first_index: int, second_index: int, is_red: bool, **kwargs: Any) -> list:
    if not import_numpy():  # no numpy at all: a numpy-free team rather than forfeiting every game
        from contest.baseline_team import create_team as numpy_free_team
        print("coordinator: numpy unavailable, playing the numpy-free fallback team", file=sys.stderr)
        return numpy_free_team(first_index, second_index, is_red)
    pin_blas_threads()
    from pacman_coordinator.versions import make_brain  # needs numpy
    brain = make_brain(is_red, WEIGHTS)  # v1, v2 or v3 checkpoints
    guard = Guard(is_red, [first_index, second_index])
    return [CoordinatedAgent(first_index, brain, guard), CoordinatedAgent(second_index, brain, guard)]


class CoordinatedAgent(CaptureAgent):
    """Both our agents share one Brain: the coordinator decides every move of the team, except on a layout with a
    single entrance to our food, where one agent guards it (pacman_coordinator.guard)."""

    def __init__(self, index: int, brain: Any, guard: Any = None) -> None:
        super().__init__(index, 0.1)
        self.brain = brain
        self.guard = guard

    def register_initial_state(self, game_state: Any) -> None:
        self.start = game_state.get_agent_position(self.index)
        self.brain.start(game_state)
        if self.guard is not None:
            try:
                self.guard.start(game_state)
            except Exception:  # noqa: BLE001 - without a guard the coordinator plays both agents
                self.guard = None

    def choose_action(self, game_state: Any) -> str:
        if self.guard is not None:
            try:
                move = self.guard.act(game_state, self.index)
                if move is not None:
                    return move
            except Exception:  # noqa: BLE001
                pass
        return self.brain.safe_choose(game_state, self.index)

    def final(self, game_state: Any) -> None:
        """v3 search statistics, appended as one JSON line to $V3_STATS when set (benchmarks only). Never raises."""
        super().final(game_state)
        path = os.environ.get("V3_STATS")
        stats = getattr(self.brain, "stats", None)
        if not path or stats is None or getattr(self.brain, "reported", False):
            return
        try:
            import json
            self.brain.reported = True
            with open(path, "a") as f:
                f.write(json.dumps(dict(stats, red=self.brain.team == 0, score=game_state.get_score())) + "\n")
        except Exception:
            pass
