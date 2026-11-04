"""Smoke test against the real DocLayNet-small val split (skipped when it is not checked out)."""

import importlib.util
import json
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).parent.parent
VAL = ROOT / "datasets/doclaynet/val"
pytestmark = pytest.mark.skipif(not VAL.is_dir(), reason="datasets/doclaynet not present")


def _mod():
    spec = importlib.util.spec_from_file_location("d2c", ROOT / "scripts/doclaynet_to_coco.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _convert():
    return _mod().convert(VAL)


def test_converted_boxes_lie_inside_their_images():
    coco = _convert()
    sizes = {i["id"]: (i["width"], i["height"]) for i in coco["images"]}
    assert len(coco["annotations"]) == 1034 and len({a["id"] for a in coco["annotations"]}) == 1034
    for a in coco["annotations"]:  # observed overshoot in val is 1.31 px; pool build clips it
        x, y, w, h = a["bbox"]
        W, H = sizes[a["image_id"]]
        assert -2 <= x and -2 <= y and x + w <= W + 2 and y + h <= H + 2, a


def test_pool_from_doclaynet_val(cli, tmp_path):
    _mod().write_dataset(VAL, tmp_path / "ds")
    r = cli("pool", "build", "--dataset", tmp_path / "ds", "--out", tmp_path / "p")
    assert r.exit_code == 0, r.output
    m = json.loads((tmp_path / "p/pool.json").read_text())
    assert len(m["categories"]) == 11 and 900 < len(m["elements"]) <= 1034
    e = m["elements"][0]
    with Image.open(tmp_path / "p" / e["path"]) as im:
        assert im.size[0] > 0
