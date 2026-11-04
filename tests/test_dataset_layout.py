"""The rendered dataset follows CVAT's COCO 1.0 layout and the cocodataset.org field list.

Sources: https://docs.cvat.ai/docs/dataset_management/formats/format-coco/ (images/<subset>/,
annotations/instances_<subset>.json; boxes import when `segmentation` is empty) and
https://cocodataset.org/#format-data (info, licenses, images, annotations, categories).
"""

import json
from pathlib import Path

from conftest import POOL_FLAGS, RENDER_FLAGS


def _tree(root: Path) -> list[str]:
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())


def test_cvat_coco_tree_and_fields(cli, pool_dir, tmp_path):
    out = tmp_path / "ds"
    r = cli(
        "run",
        "--pool",
        pool_dir,
        "--count",
        3,
        "--out",
        out,
        "--seed",
        1,
        "--subset",
        "val",
        "--workers",
        0,
        *POOL_FLAGS,
        *RENDER_FLAGS,
    )
    assert r.exit_code == 0, r.output
    assert _tree(out) == sorted(
        [
            "annotations/instances_val.json",
            "classes.txt",
            "layouts.jsonl",
            "run.json",
            *(f"images/val/{i:08d}.jpg" for i in range(3)),
            *(f"labels/val/{i:08d}.txt" for i in range(3)),
        ]
    )

    d = json.loads((out / "annotations/instances_val.json").read_text())
    assert set(d) == {"info", "licenses", "images", "annotations", "categories"}
    assert d["licenses"] == [] and "version" in d["info"]
    ids = [i["id"] for i in d["images"]]
    assert ids == [1, 2, 3] and all(
        set(i) == {"id", "width", "height", "file_name"} and i["file_name"] == f"{i['id'] - 1:08d}.jpg"
        for i in d["images"]
    )
    cats = {c["id"] for c in d["categories"]}
    assert all(set(c) == {"id", "name", "supercategory"} and c["id"] > 0 for c in d["categories"])
    anns = d["annotations"]
    assert [a["id"] for a in anns] == list(range(1, len(anns) + 1))
    for a in anns:
        assert set(a) == {"id", "image_id", "category_id", "segmentation", "area", "bbox", "iscrowd"}
        assert a["segmentation"] == [] and a["iscrowd"] == 0  # CVAT imports these as boxes
        assert a["image_id"] in ids and a["category_id"] in cats
        assert a["area"] == a["bbox"][2] * a["bbox"][3] > 0


def test_subset_must_be_a_safe_folder_name(cli, pool_dir, tmp_path):
    for bad in ("..", "a/b", ".hidden", ""):
        r = cli(
            "run", "--pool", pool_dir, "--count", 1, "--out", tmp_path / "x", "--subset", bad, "--dry-run", *POOL_FLAGS
        )
        assert r.exit_code == 2, bad


def test_dry_run_lists_the_cvat_paths(cli, pool_dir, tmp_path):
    r = cli("run", "--pool", pool_dir, "--count", 1, "--out", tmp_path / "x", "--dry-run", *POOL_FLAGS)
    outs = [Path(p).relative_to(tmp_path / "x").as_posix() for p in json.loads(r.stdout)["outputs"]]
    assert outs == [
        "layouts.jsonl",
        "images/default",
        "labels/default",
        "annotations/instances_default.json",
        "classes.txt",
        "run.json",
    ]
