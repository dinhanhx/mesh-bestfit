from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

F64 = NDArray[np.float64]
Grids = tuple[F64, F64, F64, F64]  # x0, y0, x1, y1 (normalized)

_BLOCK = 512  # X-pair rows per block; caps the `covered` temporaries when many boxes are placed


def empty_grids(boxes: F64) -> Grids:
    """All lattice rectangles with zero overlap area against `boxes` ((n, 4) xyxy in [0, 1])."""
    xs = np.unique(np.concatenate(([0.0, 1.0], boxes[:, 0], boxes[:, 2])))
    ys = np.unique(np.concatenate(([0.0, 1.0], boxes[:, 1], boxes[:, 3])))
    occ = np.zeros((xs.size - 1, ys.size - 1), dtype=np.int32)
    for x1, y1, x2, y2 in boxes:  # edges are ticks, so searchsorted hits exactly
        occ[np.searchsorted(xs, x1) : np.searchsorted(xs, x2), np.searchsorted(ys, y1) : np.searchsorted(ys, y2)] = 1
    sat = np.zeros((xs.size, ys.size), dtype=np.int32)
    sat[1:, 1:] = occ.cumsum(0).cumsum(1)  # summed-area table over cells
    i0, i1 = np.triu_indices(xs.size, 1)
    j0, j1 = np.triu_indices(ys.size, 1)
    rows: list[NDArray[np.intp]] = []
    cols: list[NDArray[np.intp]] = []
    for s in range(0, i0.size, _BLOCK):
        r0, r1 = i0[s : s + _BLOCK, None], i1[s : s + _BLOCK, None]
        covered = sat[r1, j1] - sat[r0, j1] - sat[r1, j0] + sat[r0, j0]
        a, b = np.nonzero(covered == 0)  # touching an element's edge is allowed
        rows.append(a + s)
        cols.append(b)
    a, b = np.concatenate(rows), np.concatenate(cols)
    return xs[i0[a]], ys[j0[b]], xs[i1[a]], ys[j1[b]]


def best_pair(cw: F64, ch: F64, grids: Grids) -> tuple[float, int, int]:
    """Max over (candidate, grid) of w*h / grid_area subject to w <= gw and h <= gh.

    Returns (fill, candidate_index, grid_index); candidate_index is -1 when nothing fits.
    Ties: lowest candidate index, then lexicographically lowest (x0, y0, x1, y1).
    """
    x0, y0, x1, y1 = grids
    if x0.size == 0:
        return 0.0, -1, -1
    gw, gh = x1 - x0, y1 - y0
    area = gw * gh
    uw, wi = np.unique(gw, return_inverse=True)
    uh, hi = np.unique(gh, return_inverse=True)
    table = np.full((uw.size, uh.size), np.inf)
    np.minimum.at(table, (wi, hi), area)
    # table[a, b] = smallest grid area with width rank >= a and height rank >= b
    table = np.minimum.accumulate(table[::-1], axis=0)[::-1]
    table = np.minimum.accumulate(table[:, ::-1], axis=1)[:, ::-1]
    a = np.searchsorted(uw, cw, side="left")
    b = np.searchsorted(uh, ch, side="left")
    ok = (a < uw.size) & (b < uh.size)
    min_area = np.full(cw.size, np.inf)
    min_area[ok] = table[a[ok], b[ok]]
    fill = np.where(np.isfinite(min_area), cw * ch / min_area, 0.0)
    ci = int(np.argmax(fill))  # first maximum -> lowest candidate index
    if fill[ci] <= 0.0:
        return 0.0, -1, -1
    hits = np.flatnonzero((gw >= cw[ci]) & (gh >= ch[ci]) & (area == min_area[ci]))
    gi = int(hits[np.lexsort((y1[hits], x1[hits], y0[hits], x0[hits]))[0]])
    return float(fill[ci]), ci, gi
