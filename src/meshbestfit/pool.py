from __future__ import annotations

import hashlib
import logging
import math
import re
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import orjson
from numpy.typing import NDArray
from PIL import Image

from .config import PoolConfig
from .schema import Category, DataError, Element, PoolManifest, Source

log = logging.getLogger(__name__)

JPEG_QUALITY = 95


@dataclass(frozen=True, slots=True)
class PoolArrays:
    """Pool rows that survive epsilon and the category exclusions, as flat arrays."""

    w: NDArray[np.float64]
    h: NDArray[np.float64]
    small: NDArray[np.bool_]
    ids: tuple[str, ...]
    category_ids: NDArray[np.intp]


def pool_arrays(m: PoolManifest, cfg: PoolConfig) -> PoolArrays:
    excluded = {c.id for c in m.categories if c.name in cfg.exclude_categories}
    els = [e for e in m.elements if e.w >= cfg.min_size and e.h >= cfg.min_size and e.category_id not in excluded]
    w = np.array([e.w for e in els], dtype=np.float64)
    h = np.array([e.h for e in els], dtype=np.float64)
    return PoolArrays(
        w,
        h,
        ~((w >= cfg.small_threshold) & (h >= cfg.small_threshold)),
        tuple(e.id for e in els),
        np.array([e.category_id for e in els], dtype=np.intp),
    )


def _safe(name: str) -> str:
    return re.sub(r"[^\w.-]+", "_", name)


def _coco_dict(raw: bytes, path: Path) -> dict[str, Any]:
    try:
        d = orjson.loads(raw)
        for key in ("images", "annotations", "categories"):
            if not isinstance(d[key], list):
                raise TypeError(key)
    except (orjson.JSONDecodeError, KeyError, TypeError) as e:
        raise DataError(f"{path}: not a COCO file (need images, annotations, categories): {e!r}") from e
    return dict(d)


def resolve_dataset(root: Path, subset: str | None) -> tuple[Path, Path]:
    """CVAT/Datumaro COCO 1.0 tree -> (annotations json, images folder)."""
    found = {p.stem.removeprefix("instances_"): p for p in sorted((root / "annotations").glob("instances_*.json"))}
    if subset is None:
        if len(found) != 1:
            raise DataError(f"{root}: pass --subset; annotations/ holds {sorted(found) or 'no instances_*.json'}")
        subset = next(iter(found))
    if subset not in found:
        raise DataError(f"{root}: no annotations/instances_{subset}.json")
    images = root / "images" / subset
    if not images.is_dir():
        raise DataError(f"{root}: no images/{subset}/ folder")
    return found[subset], images


def build_pool(
    coco: Path,
    images_dir: Path,
    out: Path,
    cfg: PoolConfig,
    tick: Callable[[], None] = lambda: None,
) -> PoolManifest:
    """Crop every COCO instance into `out/<category>/<id>.jpg` and write `out/pool.json`."""
    raw = coco.read_bytes()
    d = _coco_dict(raw, coco)
    try:
        cats = {int(c["id"]): str(c["name"]) for c in d["categories"]}
        imgs = {int(i["id"]): i for i in d["images"]}
        anns = [
            (str(a["id"]), int(a["image_id"]), int(a["category_id"]), [float(v) for v in a["bbox"]])
            for a in d["annotations"]
        ]
    except (KeyError, TypeError, ValueError) as e:
        raise DataError(f"{coco}: malformed COCO entry: {e!r}") from e
    if len({a[0] for a in anns}) != len(anns):
        raise DataError(f"{coco}: duplicate annotation ids")
    if len(cats) != len(d["categories"]):
        raise DataError(f"{coco}: duplicate category ids")
    unknown = {n for n in cfg.exclude_categories if n not in cats.values()}
    if unknown:
        log.warning("--exclude-category names not in the COCO file: %s", sorted(unknown))
    kept = {i: n for i, n in cats.items() if n not in cfg.exclude_categories}

    by_image: dict[int, list[tuple[str, int, list[float]]]] = defaultdict(list)
    for aid, image_id, cat_id, bbox in anns:
        if image_id not in imgs:
            raise DataError(f"{coco}: annotation {aid}: unknown image id {image_id}")
        if cat_id not in cats:
            raise DataError(f"{coco}: annotation {aid}: unknown category id {cat_id}")
        if len(bbox) != 4:
            raise DataError(f"{coco}: annotation {aid}: bbox needs 4 values")
        if cat_id in kept:
            by_image[image_id].append((aid, cat_id, bbox))

    elements: list[Element] = []
    skipped = 0
    for image_id in sorted(by_image):
        info = imgs[image_id]
        W, H = float(info["width"]), float(info["height"])
        with Image.open(images_dir / info["file_name"]) as im:
            if im.size != (int(W), int(H)):
                raise DataError(f"{info['file_name']}: size {im.size} != COCO {int(W)}x{int(H)}")
            rgb = im.convert("RGB")
        for aid, cat_id, (x, y, bw, bh) in by_image[image_id]:
            x0, y0 = max(x, 0.0), max(y, 0.0)
            x1, y1 = min(x + bw, W), min(y + bh, H)
            w, h = (x1 - x0) / W, (y1 - y0) / H
            if w <= 0 or h <= 0:
                raise DataError(f"{coco}: annotation {aid}: empty bbox")
            if w < cfg.min_size or h < cfg.min_size:
                skipped += 1
                continue
            rel = f"{_safe(cats[cat_id])}/{aid}.jpg"
            (out / rel).parent.mkdir(parents=True, exist_ok=True)
            box = (math.floor(x0), math.floor(y0), math.ceil(x1), math.ceil(y1))
            rgb.crop(box).save(out / rel, format="JPEG", quality=JPEG_QUALITY)
            elements.append(Element(id=aid, category_id=cat_id, w=w, h=h, path=rel))
        tick()
    log.info("pool: %d elements, %d below min_size skipped", len(elements), skipped)

    manifest = PoolManifest(
        source=Source(coco=coco.name, sha256=hashlib.sha256(raw).hexdigest()),
        categories=tuple(Category(id=i, name=n) for i, n in sorted(kept.items())),
        elements=tuple(elements),
    )
    (out / "pool.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return manifest
