from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from PIL import Image, ImageDraw

from .schema import Element, LayoutRecord

CELL = 600
PALETTE = [
    "#e6194b",
    "#3cb44b",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#008080",
    "#9a6324",
    "#800000",
    "#808000",
    "#000075",
    "#469990",
    "#f032e6",
]


def draw_grid(
    records: Iterable[LayoutRecord],
    rows: int,
    cols: int,
    *,
    labels: bool = False,
    elements: dict[str, Element] | None = None,
    root: Path | None = None,
) -> Image.Image:
    """First rows*cols layouts as boxes (or pool crops when `elements` is given) on white cells."""
    grid = Image.new("RGB", (cols * CELL, rows * CELL), "white")
    for n, rec in enumerate(records):
        if n >= rows * cols:
            break
        cell = Image.new("RGB", (CELL, CELL), "white")
        d = ImageDraw.Draw(cell)
        for box, cat, eid in zip(rec.boxes, rec.category_ids, rec.element_ids, strict=True):
            x0, y0, x1, y1 = (round(v * CELL) for v in box)
            if elements is not None and root is not None:
                with Image.open(root / elements[eid].path) as im:
                    crop = im.convert("RGB").resize((max(1, x1 - x0), max(1, y1 - y0)))
                cell.paste(crop, (x0, y0))
            color = PALETTE[cat % len(PALETTE)]
            d.rectangle((x0, y0, x1, y1), outline=color, width=2)
            if labels:
                d.text((x0 + 3, y0 + 2), str(cat), fill=color)
        d.rectangle((0, 0, CELL - 1, CELL - 1), outline="#888888")
        grid.paste(cell, ((n % cols) * CELL, (n // cols) * CELL))
    return grid
