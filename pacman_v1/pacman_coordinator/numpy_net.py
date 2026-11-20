"""Numpy forward pass of pacman_train.networks.Coordinator (Python 3.8, old numpy: no sliding_window_view)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pacman_v1.pacman_jax.common import ACTION_VECTORS


def load(path: Path) -> tuple[dict, dict]:
    """Weights exported by pacman_train.train.export_actor, and the metadata json written next to them."""
    with np.load(str(path)) as data:
        params = {k: data[k].astype(np.float32) for k in data.files}
    meta = json.loads(Path(path).with_suffix(".json").read_text())
    return params, meta


def conv3x3(x: np.ndarray, kernel: np.ndarray, bias: np.ndarray, dilation: int) -> np.ndarray:
    """SAME-padded dilated 3x3 cross-correlation, like flax.linen.Conv. x [W,H,C], kernel [3,3,C,O]."""
    w, h, c = x.shape
    d = dilation
    padded = np.zeros((w + 2 * d, h + 2 * d, c), dtype=np.float32)
    padded[d:d + w, d:d + h] = x
    cols = np.concatenate([padded[i * d:i * d + w, j * d:j * d + h] for i in range(3) for j in range(3)], axis=-1)
    return (cols.reshape(w * h, 9 * c) @ kernel.reshape(9 * c, -1) + bias).reshape(w, h, -1)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def board_mask(grid: np.ndarray) -> np.ndarray:
    """[W,H,1] True on the real board (pacman_train.networks.board_mask): all True unless the board was padded."""
    home = grid[..., 1] > 0.5
    width, height = 2 * int(home.any(1).sum()), int(home.any(0).sum())
    if width == 0:  # no home cells (the all-zero warm-up grid): the whole grid
        width, height = grid.shape[:2]
    mask = np.zeros(grid.shape[:2] + (1,), dtype=bool)
    mask[:width, :height] = True
    return mask


def forward(params: dict, meta: dict, grid: np.ndarray, vec: np.ndarray, cell: np.ndarray
            ) -> tuple[np.ndarray, np.ndarray]:
    """Move logits [5] and intent logits [W,H] for one acting agent (unbatched)."""
    # v1 learned on unpadded boards: masking anything a board is padded with keeps it playing like the board alone
    mask = board_mask(grid)
    x = np.where(mask, grid.astype(np.float32), 0.0).astype(np.float32)
    for i, d in enumerate(meta["actor_dilations"]):
        x = relu(conv3x3(x, params["Trunk_0/Conv_%d/kernel" % i], params["Trunk_0/Conv_%d/bias" % i], d)) * mask
    intent = (x @ params["Conv_0/kernel"].reshape(x.shape[-1], 1))[..., 0] + params["Conv_0/bias"][0]
    pts = np.asarray(cell)[None, :] + ACTION_VECTORS
    local = x[pts[:, 0], pts[:, 1]].reshape(-1)
    pooled = np.concatenate([x.sum((0, 1)) / mask.sum(), x.max((0, 1))])
    v = relu(vec.astype(np.float32) @ params["Dense_0/kernel"] + params["Dense_0/bias"])
    hid = np.concatenate([local, pooled, v])
    hid = relu(hid @ params["Dense_1/kernel"] + params["Dense_1/bias"])
    hid = relu(hid @ params["Dense_2/kernel"] + params["Dense_2/bias"])
    return hid @ params["Dense_3/kernel"] + params["Dense_3/bias"], intent
