"""Rare-category variants. NumPy + OpenCV only (the `[augment]` extra).

Operations follow the list in spec section 2 (brightness/contrast, resized crop, Sobel, elastic,
Gaussian noise, optional flip). Their strengths are this package's own choices: the repo's
`augmentation.py` was not available when this was written.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from .pool import JPEG_QUALITY
from .schema import PoolManifest, load_manifest

log = logging.getLogger(__name__)
U8 = NDArray[np.uint8]


def _brightness_contrast(cv2: Any, img: U8, rng: np.random.Generator) -> U8:
    return np.asarray(cv2.convertScaleAbs(img, alpha=1 + rng.uniform(-0.2, 0.2), beta=rng.uniform(-0.2, 0.2) * 255))


def _resized_crop(cv2: Any, img: U8, rng: np.random.Generator) -> U8:
    h, w = img.shape[:2]
    cw, ch = max(1, int(w * rng.uniform(0.8, 1.0))), max(1, int(h * rng.uniform(0.8, 1.0)))
    x, y = int(rng.integers(0, w - cw + 1)), int(rng.integers(0, h - ch + 1))
    return np.asarray(cv2.resize(img[y : y + ch, x : x + cw], (w, h), interpolation=cv2.INTER_LINEAR))


def _sobel(cv2: Any, img: U8, rng: np.random.Generator) -> U8:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mag = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    edges = cv2.cvtColor(np.clip(mag, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    a = rng.uniform(0.05, 0.2)
    return np.asarray(cv2.addWeighted(img, 1 - a, edges, a, 0))


def _elastic(cv2: Any, img: U8, rng: np.random.Generator) -> U8:
    h, w = img.shape[:2]
    sigma = max(2.0, 0.05 * min(h, w))
    amp = max(1.0, rng.uniform(0.01, 0.04) * min(h, w))
    d = []
    for _ in range(2):
        f = cv2.GaussianBlur(rng.uniform(-1, 1, (h, w)).astype(np.float32), (0, 0), sigma)
        d.append(f / max(float(np.abs(f).max()), 1e-6) * amp)
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return np.asarray(cv2.remap(img, gx + d[0], gy + d[1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101))


def _noise(cv2: Any, img: U8, rng: np.random.Generator) -> U8:
    n = rng.normal(0, rng.uniform(3, 7), img.shape)
    return np.clip(img + n, 0, 255).astype(np.uint8)


_OPS = (_brightness_contrast, _resized_crop, _sobel, _elastic, _noise)


def augment_image(cv2: Any, img: U8, rng: np.random.Generator, flip: bool) -> U8:
    """Each op runs with p = 0.5; at least one always runs."""
    use = rng.random(len(_OPS)) < 0.5
    if not use.any():
        use[rng.integers(len(_OPS))] = True
    for op, on in zip(_OPS, use, strict=True):
        if on:
            img = op(cv2, img, rng)
    if flip and rng.random() < 0.5:
        img = np.ascontiguousarray(img[:, ::-1])
    return img


def augment_pool(
    pool_json: Path, *, min_count: int, aug_times: int, seed: int, flip: bool, tick: Callable[[], None] = lambda: None
) -> PoolManifest:
    """Give every element of a category with <= min_count elements `aug_times` variants."""
    try:
        import cv2  # noqa: PLC0415  # optional extra, imported lazily
    except ImportError as e:
        raise ImportError("pool augment needs the extra: pip install 'mesh-bestfit[augment]'") from e
    m = load_manifest(pool_json)
    root = pool_json.parent
    counts = Counter(e.category_id for e in m.elements)
    rare = {c for c, n in counts.items() if n <= min_count}
    out = []
    for idx, el in enumerate(m.elements):
        out_el = el
        if el.category_id in rare:
            rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(idx,)))
            with Image.open(root / el.path) as im:
                src = np.asarray(im.convert("RGB"))
            d = Path(el.path).parent / "aug" / el.id
            (root / d).mkdir(parents=True, exist_ok=True)
            variants = []
            for k in range(aug_times):
                rel = (d / f"{el.id}_{k:04d}.jpg").as_posix()
                Image.fromarray(augment_image(cv2, src, rng, flip)).save(
                    root / rel, format="JPEG", quality=JPEG_QUALITY
                )
                variants.append(rel)
            out_el = el.model_copy(update={"variants": tuple(variants)})
            tick()
        out.append(out_el)
    new = m.model_copy(update={"elements": tuple(out)})
    log.info("augment: %d rare categories, %d elements gained variants", len(rare), sum(1 for e in out if e.variants))
    pool_json.write_text(new.model_dump_json(indent=2), encoding="utf-8")
    return new
