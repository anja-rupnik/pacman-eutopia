"""Frozen copy of the features-v1 contest code (numpy only, Python 3.8), from commit 0615e12.

Checkpoints trained before features v2 (e.g. the laptop run runs/3050-a) only work with this code. The packages are
renamed (pacman_v1.pacman_jax, ...) so that a v1 team and a current team can play in the same process (real-engine
games load both teams into one interpreter). pacman_coordinator.versions picks the right one from a checkpoint's
.json. Do not edit: this has to stay identical to what the v1 models were trained with.
"""
