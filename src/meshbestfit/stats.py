from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any

import numpy as np

from .schema import LayoutRecord


def _align(b: np.ndarray) -> float:
    """Mean over boxes of -log(1 - d), d = smallest gap to any other box on left/center/right/top/
    center/bottom lines. [Unverified] the paper's normalization (spec section 9); this follows the
    LayoutGAN++ style."""
    if len(b) < 2:
        return 0.0
    cx, cy = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2
    feats = [b[:, 0], cx, b[:, 2], b[:, 1], cy, b[:, 3]]
    d = np.min([np.abs(f[:, None] - f[None, :]) + np.eye(len(b)) * 2 for f in feats], axis=(0, 2))
    return float(np.mean(-np.log1p(-np.minimum(d, 1 - 1e-9))))


def _mean(v: list[float] | list[int]) -> float:
    return float(np.mean(v)) if v else 0.0


def compute_stats(records: Iterable[LayoutRecord]) -> dict[str, Any]:
    """Density = sum of box areas / page area (Paper 1 Eq. 7, arxiv.org/abs/2410.12628), alignment, counts."""
    pages = 0
    density: list[float] = []
    align: list[float] = []
    per_page: list[int] = []
    cats: Counter[int] = Counter()
    reasons: Counter[str] = Counter()
    for r in records:
        b = np.asarray(r.boxes)
        pages += 1
        per_page.append(len(b))
        density.append(float(((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])).sum()))
        align.append(_align(b))
        cats.update(r.category_ids)
        reasons[r.stop_reason] += 1
    return {
        "pages": pages,
        "boxes_per_page": _mean(per_page),
        "density": _mean(density),
        "alignment": _mean(align),
        "category_counts": {str(k): v for k, v in sorted(cats.items())},
        "stop_reasons": dict(sorted(reasons.items())),
    }
