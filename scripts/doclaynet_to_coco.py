"""Convert a DocLayNet-small split (annotations/*.json + images/*.png) to a COCO dataset folder.

Each page JSON has `form[]` rows; rows sharing `id_box` are the lines of one layout block, and
`box` = [x, y, w, h], already in the page PNG's pixel space (`coco_width` x `coco_height`,
origin top-left), so no rescaling. Category ids follow DocLayNet's alphabetical order. The folder
is the layout `mesh-bestfit pool build --dataset` reads; the subset is the split's folder name.

    python scripts/doclaynet_to_coco.py datasets/doclaynet/train --out work/doclaynet-train
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CATEGORIES = [
    "Caption",
    "Footnote",
    "Formula",
    "List-item",
    "Page-footer",
    "Page-header",
    "Picture",
    "Section-header",
    "Table",
    "Text",
    "Title",
]
CAT_ID = {n: i for i, n in enumerate(CATEGORIES, start=1)}


def convert(split: Path) -> dict[str, object]:
    images, annotations = [], []
    for n, ann_file in enumerate(sorted((split / "annotations").glob("*.json")), start=1):
        d = json.loads(ann_file.read_text(encoding="utf-8"))
        m = d["metadata"]
        images.append(
            {"id": n, "file_name": f"{ann_file.stem}.png", "width": m["coco_width"], "height": m["coco_height"]}
        )
        seen = set()
        for row in d["form"]:
            if row["id_box"] in seen:
                continue
            seen.add(row["id_box"])
            annotations.append(
                {
                    "id": row["id_box"],
                    "image_id": n,
                    "iscrowd": 0,
                    "category_id": CAT_ID[row["category"]],
                    "bbox": row["box"],
                }
            )
    return {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": i, "name": n} for n, i in CAT_ID.items()],
    }


def write_dataset(split: Path, out: Path) -> dict[str, object]:
    """Write out/annotations/instances_<split>.json and symlink out/images/<split> to the PNGs."""
    coco = convert(split)
    ann, link = out / "annotations" / f"instances_{split.name}.json", out / "images" / split.name
    ann.parent.mkdir(parents=True, exist_ok=True)
    link.parent.mkdir(parents=True, exist_ok=True)
    ann.write_text(json.dumps(coco), encoding="utf-8")
    link.unlink(missing_ok=True)
    link.symlink_to((split / "images").resolve(), target_is_directory=True)
    return coco


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("split", type=Path, help="e.g. datasets/doclaynet/train")
    ap.add_argument("--out", type=Path, required=True, help="dataset folder to write")
    a = ap.parse_args()
    coco = write_dataset(a.split, a.out)
    print(f"{len(coco['images'])} images, {len(coco['annotations'])} annotations -> {a.out}")  # type: ignore[arg-type]
