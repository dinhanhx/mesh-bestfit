import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))  # `import reference`

import json

import numpy as np
import pytest
from PIL import Image
from typer.testing import CliRunner

from meshbestfit.cli import app

POOL_FLAGS = ["--candidates", "8", "--large-ratio", "0.75"]  # the fixture pool is tiny
RENDER_FLAGS = ["--width", "200", "300"]


@pytest.fixture
def cli():
    runner = CliRunner()
    return lambda *args: runner.invoke(app, [str(a) for a in args])


@pytest.fixture
def coco_fixture(tmp_path):
    """3 images, 12 instances each (10 large, 2 small), 2 categories."""
    rng = np.random.default_rng(0)
    (tmp_path / "images").mkdir()
    images, anns = [], []
    for i in range(3):
        Image.fromarray(rng.integers(0, 255, (500, 400, 3), dtype=np.uint8)).save(tmp_path / "images" / f"{i}.png")
        images.append({"id": i, "file_name": f"{i}.png", "width": 400, "height": 500})
        for k in range(12):
            small = k >= 10
            w, h = (rng.uniform(8, 15), rng.uniform(8, 15)) if small else (rng.uniform(60, 200), rng.uniform(40, 120))
            anns.append(
                {
                    "id": len(anns) + 1,
                    "image_id": i,
                    "category_id": 1 + k % 2,
                    "bbox": [rng.uniform(0, 400 - w), rng.uniform(0, 500 - h), w, h],
                }
            )
    coco = {"images": images, "annotations": anns, "categories": [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}]}
    path = tmp_path / "coco.json"
    path.write_text(json.dumps(coco))
    return path


@pytest.fixture
def pool_dir(tmp_path, coco_fixture, cli):
    out = tmp_path / "pool"
    r = cli("pool", "build", "--coco", coco_fixture, "--images-dir", tmp_path / "images", "--out", out)
    assert r.exit_code == 0, r.output
    return out
