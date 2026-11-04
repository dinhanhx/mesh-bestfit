import itertools

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from meshbestfit.layout import LayoutParams, ShortStratumError, generate_layout

rng0 = np.random.default_rng(0)
W = np.concatenate([rng0.uniform(0.05, 0.9, 3500), rng0.uniform(0.01, 0.05, 500)])
H = np.concatenate([rng0.uniform(0.05, 0.5, 3500), rng0.uniform(0.01, 0.05, 500)])
SMALL = (W < 0.05) | (H < 0.05)


@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 2**32 - 1), page=st.integers(0, 10_000))
def test_invariants(seed: int, page: int) -> None:
    p = LayoutParams(max_elements=35, fill_threshold=1e-4)
    ss = np.random.SeedSequence(seed, spawn_key=(page,))
    a = generate_layout(W, H, SMALL, p, np.random.default_rng(ss))
    b = generate_layout(W, H, SMALL, p, np.random.default_rng(ss))
    assert np.array_equal(a.boxes, b.boxes)  # determinism
    bx = a.boxes
    assert 1 <= len(bx) <= 35 and (bx >= 0).all() and (bx <= 1).all()
    for u, v in itertools.combinations(range(len(bx)), 2):  # no overlap after shrink
        ox = min(bx[u, 2], bx[v, 2]) - max(bx[u, 0], bx[v, 0])
        oy = min(bx[u, 3], bx[v, 3]) - max(bx[u, 1], bx[v, 1])
        assert not (ox > 0 and oy > 0)


def test_small_cap_and_stop_reason() -> None:
    p = LayoutParams(candidates=100, large_ratio=0.5, max_small=2, max_elements=None, fill_threshold=0.0)
    lay = generate_layout(W, H, SMALL, p, np.random.default_rng(3))
    assert SMALL[lay.element_idx].sum() <= 2
    assert lay.stop_reason in {"no_fit", "small_cap", "pool_exhausted"}


def test_short_stratum() -> None:
    few_small = np.zeros(SMALL.size, dtype=bool)
    few_small[:2] = True
    with pytest.raises(ShortStratumError):
        generate_layout(W, H, few_small, LayoutParams(candidates=50, large_ratio=0.5), np.random.default_rng(0))
    lay = generate_layout(
        W,
        H,
        few_small,
        LayoutParams(candidates=50, large_ratio=0.5, on_short_stratum="take-all"),
        np.random.default_rng(0),
    )
    assert len(lay.boxes) >= 1
