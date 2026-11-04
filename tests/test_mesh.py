import numpy as np
import pytest
from reference import ref_best, ref_grids

from meshbestfit import mesh
from meshbestfit.mesh import best_pair, empty_grids


def _random_boxes(rng: np.random.Generator, n: int) -> np.ndarray:
    """Non-overlapping boxes: a coarse jittered grid keeps them disjoint."""
    cells = rng.permutation(16)[:n]
    out = []
    for c in cells:
        x0, y0 = (c % 4) / 4 + rng.uniform(0, 0.05), (c // 4) / 4 + rng.uniform(0, 0.05)
        out.append((x0, y0, x0 + rng.uniform(0.05, 0.19), y0 + rng.uniform(0.05, 0.19)))
    return np.array(out)


@pytest.mark.parametrize("seed", range(20))
def test_grids_match_reference(seed: int) -> None:
    rng = np.random.default_rng(seed)
    boxes = _random_boxes(rng, int(rng.integers(1, 5)))
    got = set(zip(*(g.tolist() for g in empty_grids(boxes))))
    assert got == ref_grids(boxes)


@pytest.mark.parametrize("seed", range(20))
def test_best_pair_matches_reference(seed: int) -> None:
    rng = np.random.default_rng(100 + seed)
    boxes = _random_boxes(rng, int(rng.integers(1, 5)))
    grids = empty_grids(boxes)
    cw, ch = rng.uniform(0.02, 0.5, 30), rng.uniform(0.02, 0.5, 30)
    fill, ci, gi = best_pair(cw, ch, grids)
    rf, rci = ref_best(cw, ch, set(zip(*(g.tolist() for g in grids))))
    assert (ci, fill) == (rci, pytest.approx(rf))
    if ci >= 0:
        x0, y0, x1, y1 = (g[gi] for g in grids)
        assert cw[ci] <= x1 - x0 and ch[ci] <= y1 - y0
        assert cw[ci] * ch[ci] / ((x1 - x0) * (y1 - y0)) == pytest.approx(fill)


def test_nothing_fits() -> None:
    boxes = np.array([[0.0, 0.0, 1.0, 1.0]])
    assert best_pair(np.array([0.1]), np.array([0.1]), empty_grids(boxes)) == (0.0, -1, -1)


def test_blocking_does_not_change_grids(monkeypatch: pytest.MonkeyPatch) -> None:
    boxes = _random_boxes(np.random.default_rng(1), 6)
    ref = empty_grids(boxes)
    monkeypatch.setattr(mesh, "_BLOCK", 7)
    for a, b in zip(ref, empty_grids(boxes), strict=True):
        assert np.array_equal(a, b)
