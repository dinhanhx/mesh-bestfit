from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from .config import RenderConfig
from .schema import Element, LayoutRecord


@dataclass(frozen=True, slots=True)
class PagePlan:
    size: tuple[int, int]  # (W, H) pixels
    crops: tuple[str, ...]  # pool-relative crop path chosen for each box


class RenderError(Exception):
    """A crop could not be read; the message names the element."""


def _open(root: Path, rel: str, element_id: str) -> Image.Image:
    try:
        im = Image.open(root / rel)
        im.load()
        return im
    except (OSError, ValueError) as e:
        raise RenderError(f"element {element_id} ({rel}): {e}") from e


def render_rng(rec: LayoutRecord, seed: int | None = None) -> np.random.Generator:
    """Render stream, independent of the layout stream: spawn_key = (page, 1)."""
    entropy = rec.seed.entropy if seed is None else seed
    ss = np.random.SeedSequence(entropy, spawn_key=(*rec.seed.spawn_key, 1))
    return np.random.default_rng(ss)


def plan_page(
    rec: LayoutRecord, elements: Mapping[str, Element], cfg: RenderConfig, seed: int | None = None
) -> PagePlan:
    """Every random draw of a page. Cheap, so labels can replay it without decoding crops."""
    rng = render_rng(rec, seed)
    W = int(rng.integers(cfg.width[0], cfg.width[1] + 1))
    r = float(rng.uniform(cfg.aspect[0], cfg.aspect[1]))
    H = min(cfg.max_height, int(W * r))
    crops = []
    for eid in rec.element_ids:
        el = elements[eid]
        options = (el.path, *el.variants) if cfg.use_variants else (el.path,)
        crops.append(options[int(rng.integers(len(options)))])
    return PagePlan((W, H), tuple(crops))


def fit_rect(box_px: NDArray[np.float64], crop_size: tuple[int, int], keep_aspect: bool) -> NDArray[np.float64]:
    """The rectangle a crop occupies: its whole box, or the largest aspect-correct fit, centered."""
    if not keep_aspect:
        return box_px
    bw, bh = box_px[2] - box_px[0], box_px[3] - box_px[1]
    s = min(bw / crop_size[0], bh / crop_size[1])
    w, h = crop_size[0] * s, crop_size[1] * s
    cx, cy = (box_px[0] + box_px[2]) / 2, (box_px[1] + box_px[3]) / 2
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


def pixel_boxes(rec: LayoutRecord, plan: PagePlan, root: Path, cfg: RenderConfig) -> NDArray[np.float64]:
    """Unrounded label boxes in pixels, (n, 4) xyxy. Reads crop headers, only for keep_aspect."""
    W, H = plan.size
    boxes = np.asarray(rec.boxes, dtype=np.float64) * np.array([W, H, W, H], dtype=np.float64)
    if not cfg.keep_aspect:
        return boxes
    out: list[NDArray[np.float64]] = []
    for b, rel, eid in zip(boxes, plan.crops, rec.element_ids, strict=True):
        with _open(root, rel, eid) as im:
            out.append(fit_rect(b, im.size, True))
    return np.asarray(out, dtype=np.float64)


def render_page(
    rec: LayoutRecord, plan: PagePlan, root: Path, cfg: RenderConfig
) -> tuple[Image.Image, NDArray[np.float64]]:
    """Paste every crop on a canvas. Returns the canvas and the unrounded label boxes."""
    W, H = plan.size
    canvas = Image.new("RGB", (W, H), cfg.background)
    boxes = np.asarray(rec.boxes, dtype=np.float64) * np.array([W, H, W, H], dtype=np.float64)
    rects = []
    for b, rel, eid in zip(boxes, plan.crops, rec.element_ids, strict=True):
        with _open(root, rel, eid) as im:
            rgb = im.convert("RGB")
        r = fit_rect(b, rgb.size, cfg.keep_aspect)
        x0, y0, x1, y1 = (round(float(v)) for v in r)
        w, h = max(1, x1 - x0), max(1, y1 - y0)
        canvas.paste(rgb.resize((w, h), Image.Resampling.LANCZOS), (x0, y0))
        rects.append(r)
    return canvas, np.asarray(rects)
