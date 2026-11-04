"""Slow, literal port of the repo's lattice-and-loop logic (float64, strict zero-overlap test)."""

import itertools


def ref_grids(boxes):
    xs = sorted({0.0, 1.0, *boxes[:, 0], *boxes[:, 2]})
    ys = sorted({0.0, 1.0, *boxes[:, 1], *boxes[:, 3]})
    out = set()
    for xa, xb in itertools.combinations(xs, 2):
        for ya, yb in itertools.combinations(ys, 2):
            if all(not (min(xb, b[2]) - max(xa, b[0]) > 0 and min(yb, b[3]) - max(ya, b[1]) > 0) for b in boxes):
                out.add((xa, ya, xb, yb))
    return out


def ref_best(cw, ch, grids):
    best = (0.0, -1)
    for ei, (w, h) in enumerate(zip(cw, ch)):
        for g in grids:
            gw, gh = g[2] - g[0], g[3] - g[1]
            if w > gw or h > gh:
                continue
            f = (w * h) / (gw * gh)
            if f > best[0]:
                best = (f, ei)
    return best
