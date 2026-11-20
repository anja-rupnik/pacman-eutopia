# Pacman capture-the-flag agent

The contest entry point is `my_team.py` (`create_team`), as in the contest template. Both agents are driven by one
team coordinator: a convolutional network (weights in `coordinator.npz`, settings in `coordinator.json`) evaluated
with numpy, plus a small exact search for defensive chases and a guard for layouts with a single entrance. The
code beside `my_team.py` (`pacman_coordinator/`, `pacman_jax/`, `pacman_train/`, `pacman_v1/`) is only what the agent
imports at run time; it needs Python 3.8 and numpy, nothing else, and uses one thread.

`_vendor/` holds a copy of numpy 1.24.4 (CPython 3.8, Linux x86-64) as insurance only: `my_team.py` imports the
machine's own numpy first and falls back to this copy only if none is installed.

Run a game locally from the contest framework's `src/contest` folder:

    python capture.py -r <path to this repo>/my_team.py -b baseline_team
