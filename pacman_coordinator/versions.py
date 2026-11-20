"""Which feature code a coordinator checkpoint needs, read from the .json written next to its weights.

v2 is the current code (pacman_train.features); v1 is the frozen copy in pacman_v1. v3 is a v2 checkpoint whose .json
has a "search" entry: the v2 network plus the ghost-chase search (pacman_coordinator.search_brain). Python 3.8
compatible.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pacman_train import features
from pacman_v1.pacman_train import features as features_v1

VERSIONS: dict = {"v2": features, "v1": features_v1}


def feature_version(weights: Path) -> str:
    meta = json.loads(Path(weights).with_suffix(".json").read_text())
    for name, module in VERSIONS.items():
        if list(meta["channels"]) == list(module.CHANNELS) and meta["vec"] == module.NUM_VEC:
            return name
    raise ValueError(f"{weights}: its inputs ({len(meta['channels'])} channels, {meta['vec']} vector entries) match "
                     f"no known feature version")


def brain_version(weights: Path) -> str:
    """"v1", "v2" or "v3" (v2 inputs plus the "search" entry)."""
    version = feature_version(weights)
    if version == "v2" and json.loads(Path(weights).with_suffix(".json").read_text()).get("search") is not None:
        return "v3"
    return version


def make_brain(is_red: bool, weights: Path) -> Any:
    """The Brain of the version the checkpoint was made for."""
    version = brain_version(weights)
    if version == "v1":
        from pacman_v1.pacman_coordinator.brain import Brain
    elif version == "v3":
        from pacman_coordinator.search_brain import SearchBrain as Brain
    else:
        from pacman_coordinator.brain import Brain
    return Brain(is_red, weights)
