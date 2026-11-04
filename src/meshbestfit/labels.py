from __future__ import annotations

import logging
import os
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import orjson
from numpy.typing import NDArray

from . import __version__
from .config import RenderConfig
from .render import pixel_boxes, plan_page
from .schema import Category, Element, LayoutRecord

log = logging.getLogger(__name__)


def atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def class_index(categories: Sequence[Category], offset: int | None) -> dict[int, int]:
    """COCO category id -> 0-based class id (contiguous by id order, or id + offset)."""
    if offset is not None:
        return {c.id: c.id + offset for c in categories}
    return {c.id: i for i, c in enumerate(sorted(categories, key=lambda c: c.id))}


def classes_txt(categories: Sequence[Category], offset: int | None) -> str:
    """One name per line; the line number is the class id (blank lines fill id gaps)."""
    idx = class_index(categories, offset)
    names = {idx[c.id]: c.name for c in categories}
    if not names:
        return ""
    return "".join(names.get(i, "") + "\n" for i in range(max(names) + 1))


def yolo_text(rects: NDArray[np.float64], size: tuple[int, int], classes: Sequence[int], poly: bool) -> str:
    W, H = size
    lines = []
    norm = rects / np.array([W, H, W, H], dtype=np.float64)
    for c, (x0, y0, x1, y1) in zip(classes, norm, strict=True):
        if poly:
            pts = (x0, y0, x1, y0, x1, y1, x0, y1)
            lines.append(f"{c} " + " ".join(f"{v:.6f}" for v in pts))
        else:
            lines.append(f"{c} {(x0 + x1) / 2:.6f} {(y0 + y1) / 2:.6f} {x1 - x0:.6f} {y1 - y0:.6f}")
    return "".join(line + "\n" for line in lines)


def image_name(page: int, cfg: RenderConfig) -> str:
    return f"images/{cfg.subset}/{page:08d}.{cfg.image_format}"


def label_name(page: int, cfg: RenderConfig) -> str:
    return f"labels/{cfg.subset}/{page:08d}.txt"  # mirrors images/<subset>/ as YOLO trainers expect


def coco_name(cfg: RenderConfig) -> str:
    return f"annotations/instances_{cfg.subset}.json"


def write_coco(
    out: Path,
    records: Iterator[LayoutRecord],
    categories: Sequence[Category],
    elements: dict[str, Element],
    root: Path,
    cfg: RenderConfig,
    seed: int | None = None,
) -> int:
    """Rebuild annotations/instances_<subset>.json by replaying each page's render draws.

    Layout follows CVAT's COCO 1.0 (images/<subset>/, annotations/instances_<subset>.json), so
    `file_name` is the bare name inside images/<subset>/. The body follows cocodataset.org
    (info, licenses, images, annotations, categories). Boxes carry an empty `segmentation`, which
    CVAT imports as bounding boxes. No timestamps, so the bytes depend only on the inputs.
    Pages whose image is missing (failed or not yet rendered) are skipped. Returns the page count.
    Annotations stream to disk; only the (small) image list stays in memory.
    """
    if any(c.id == 0 for c in categories):
        log.warning("category id 0: Datumaro (CVAT's importer) reads it as 'not specified'")
    head = {
        "info": {
            "description": "mesh-bestfit synthetic layouts",
            "version": __version__,
            "contributor": "mesh-bestfit",
        },
        "licenses": [],
        "categories": [{"id": c.id, "name": c.name, "supercategory": ""} for c in categories],
    }
    images: list[dict[str, object]] = []
    ann_id = 0
    path = out / coco_name(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        f.write(orjson.dumps(head)[:-1] + b',"annotations":[')
        first = True
        for rec in records:
            name = image_name(rec.page, cfg)
            if not (out / name).is_file():
                continue
            plan = plan_page(rec, elements, cfg, seed)
            images.append(
                {"id": rec.page + 1, "width": plan.size[0], "height": plan.size[1], "file_name": Path(name).name}
            )
            for cat_id, (x0, y0, x1, y1) in zip(
                rec.category_ids, pixel_boxes(rec, plan, root, cfg).tolist(), strict=True
            ):
                ann_id += 1
                a = {
                    "id": ann_id,
                    "image_id": rec.page + 1,
                    "category_id": cat_id,
                    "segmentation": [],
                    "area": (x1 - x0) * (y1 - y0),
                    "bbox": [x0, y0, x1 - x0, y1 - y0],
                    "iscrowd": 0,
                }
                f.write((b"" if first else b",") + orjson.dumps(a))
                first = False
        f.write(b'],"images":' + orjson.dumps(images) + b"}")
    os.replace(tmp, path)
    return len(images)
