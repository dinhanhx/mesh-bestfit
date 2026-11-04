from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .mesh import best_pair, empty_grids

ALGORITHM_VERSION = 1  # bump on any change to draw order or tie-breaks (spec section 6)

StopReason = Literal["max_elements", "no_fit", "below_threshold", "small_cap", "pool_exhausted"]


class ShortStratumError(ValueError):
    """A large or small stratum has fewer elements than its quota (exit code 4)."""


@dataclass(frozen=True, slots=True, kw_only=True)
class LayoutParams:
    candidates: int = 500
    large_ratio: float = 0.99
    max_elements: int | None = 35  # repo preset: None
    fill_threshold: float = 1e-4  # repo preset: 0.0
    max_small: int | None = None  # repo preset: 5
    shrink: tuple[float, float] = (0.8, 0.95)
    on_short_stratum: Literal["error", "take-all"] = "error"


@dataclass(frozen=True, slots=True)
class Layout:
    boxes: NDArray[np.float64]  # (n, 4) xyxy after shrink
    element_idx: NDArray[np.intp]  # row indices into the pool arrays
    stop_reason: StopReason


def nominal_quotas(p: LayoutParams) -> tuple[int, int]:
    k_large = int(np.floor(p.candidates * p.large_ratio + 1e-9))
    return k_large, p.candidates - k_large


def quotas(small: NDArray[np.bool_], p: LayoutParams) -> tuple[int, int]:
    """(k_large, k_small) to draw; ShortStratumError under `error` when a stratum is short."""
    k_large, k_small = nominal_quotas(p)
    n_small = int(small.sum())
    n_large = small.size - n_small
    if p.on_short_stratum == "take-all":
        k_large, k_small = min(k_large, n_large), min(k_small, n_small)
    elif k_large > n_large or k_small > n_small:
        raise ShortStratumError(
            f"pool has {n_large} large / {n_small} small elements, "
            f"need {k_large} / {k_small} (use --on-short-stratum take-all)"
        )
    if k_large + k_small == 0:
        raise ShortStratumError("pool has no elements to draw")
    return k_large, k_small


def generate_layout(
    w: NDArray[np.float64],
    h: NDArray[np.float64],
    small: NDArray[np.bool_],
    p: LayoutParams,
    rng: np.random.Generator,
) -> Layout:
    """w, h, small: pool arrays already filtered by epsilon. All randomness comes from `rng`."""
    k_large, k_small = quotas(small, p)
    large_pool, small_pool = np.flatnonzero(~small), np.flatnonzero(small)
    cand = np.concatenate(
        [
            rng.choice(large_pool, size=k_large, replace=False),
            rng.choice(small_pool, size=k_small, replace=False),
        ]
    )
    alive = np.ones(cand.size, dtype=bool)

    s = int(rng.integers(cand.size))  # start element
    alive[s] = False
    e = cand[s]
    cx = rng.uniform(w[e] / 2, 1 - w[e] / 2)
    cy = rng.uniform(h[e] / 2, 1 - h[e] / 2)
    boxes = [(cx - w[e] / 2, cy - h[e] / 2, cx + w[e] / 2, cy + h[e] / 2)]
    chosen = [e]
    small_count = int(small[e])
    reason: StopReason

    while True:
        if p.max_elements is not None and len(chosen) >= p.max_elements:
            reason = "max_elements"
            break
        if not alive.any():
            reason = "pool_exhausted"
            break
        live = np.flatnonzero(alive)
        grids = empty_grids(np.asarray(boxes))
        fill, ci, gi = best_pair(w[cand[live]], h[cand[live]], grids)
        if ci < 0:
            reason = "no_fit"
            break
        if fill < p.fill_threshold:
            reason = "below_threshold"
            break
        e = cand[live[ci]]
        if small[e] and p.max_small is not None and small_count + 1 > p.max_small:
            reason = "small_cap"
            break
        small_count += int(small[e])
        alive[live[ci]] = False
        gx = (grids[0][gi] + grids[2][gi]) / 2
        gy = (grids[1][gi] + grids[3][gi]) / 2
        boxes.append((gx - w[e] / 2, gy - h[e] / 2, gx + w[e] / 2, gy + h[e] / 2))
        chosen.append(e)

    b = np.clip(np.asarray(boxes), 0.0, 1.0)
    c = (b[:, :2] + b[:, 2:]) / 2
    half = (b[:, 2:] - b[:, :2]) / 2 * rng.uniform(p.shrink[0], p.shrink[1], size=(len(b), 2))
    return Layout(np.hstack([c - half, c + half]), np.asarray(chosen, dtype=np.intp), reason)
