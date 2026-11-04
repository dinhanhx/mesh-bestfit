import json
import os
import signal
from pathlib import Path

from conftest import POOL_FLAGS, RENDER_FLAGS
from PIL import Image

from meshbestfit import runner


def _run(cli, pool_dir: Path, out: Path, *extra: object):
    return cli("run", "--pool", pool_dir, "--count", 5, "--out", out, "--seed", 3, *POOL_FLAGS, *RENDER_FLAGS, *extra)


def test_pool_build(pool_dir):
    m = json.loads((pool_dir / "pool.json").read_text())
    assert len(m["elements"]) == 36 and [c["name"] for c in m["categories"]] == ["a", "b"]
    assert all(0 < e["w"] <= 1 and 0 < e["h"] <= 1 for e in m["elements"])
    assert all((pool_dir / e["path"]).is_file() for e in m["elements"])


def test_run_labels_match_layouts(cli, pool_dir, tmp_path):
    out = tmp_path / "ds"
    r = _run(cli, pool_dir, out, "--label-format", "yolo", "--label-format", "coco")
    assert r.exit_code == 0, r.output
    layouts = [json.loads(line) for line in (out / "layouts.jsonl").read_text().splitlines()]
    coco = json.loads((out / "annotations/instances_default.json").read_text())
    assert len(layouts) == len(coco["images"]) == 5
    assert (out / "classes.txt").read_text() == "a\nb\n"
    for rec in layouts:
        name = f"{rec['page']:08d}"
        lines = (out / "labels/default" / f"{name}.txt").read_text().splitlines()
        assert len(lines) == len(rec["boxes"])
        for line in lines:
            c, cx, cy, w, h = line.split()
            assert int(c) in (0, 1)
            assert 0 <= float(cx) - float(w) / 2 and float(cx) + float(w) / 2 <= 1
            assert 0 <= float(cy) - float(h) / 2 and float(cy) + float(h) / 2 <= 1
        img = next(i for i in coco["images"] if i["id"] == rec["page"] + 1)
        assert Image.open(out / "images/default" / img["file_name"]).size == (img["width"], img["height"])
        anns = [a for a in coco["annotations"] if a["image_id"] == rec["page"] + 1]
        assert len(anns) == len(rec["boxes"])
        for a in anns:
            x, y, w, h = a["bbox"]
            assert 0 <= x and 0 <= y and x + w <= img["width"] + 1e-6 and y + h <= img["height"] + 1e-6
    assert json.loads((out / "run.json").read_text())["seed"] == 3


def test_workers_do_not_change_output(cli, pool_dir, tmp_path):
    a, b = tmp_path / "w0", tmp_path / "w2"
    assert _run(cli, pool_dir, a, "--workers", 0).exit_code == 0
    assert _run(cli, pool_dir, b, "--workers", 2).exit_code == 0
    for rel in (
        "layouts.jsonl",
        "annotations/instances_default.json",
        "labels/default/00000004.txt",
        "images/default/00000004.jpg",
    ):
        assert (a / rel).read_bytes() == (b / rel).read_bytes(), rel


def test_render_command_matches_run(cli, pool_dir, tmp_path):
    ds, rd = tmp_path / "ds", tmp_path / "rd"
    assert _run(cli, pool_dir, ds, "--workers", 0).exit_code == 0
    r = cli("render", "--layouts", ds / "layouts.jsonl", "--pool", pool_dir, "--out", rd, "--workers", 0, *RENDER_FLAGS)
    assert r.exit_code == 0, r.output
    for rel in ("annotations/instances_default.json", "labels/default/00000002.txt", "images/default/00000002.jpg"):
        assert (ds / rel).read_bytes() == (rd / rel).read_bytes(), rel


def test_keep_aspect_label_boxes_stay_inside_layout_boxes(cli, pool_dir, tmp_path):
    out = tmp_path / "ka"
    assert _run(cli, pool_dir, out, "--keep-aspect", "--label-format", "coco").exit_code == 0
    layouts = {json.loads(line)["page"]: json.loads(line) for line in (out / "layouts.jsonl").read_text().splitlines()}
    coco = json.loads((out / "annotations/instances_default.json").read_text())
    for a in coco["annotations"]:
        im = next(i for i in coco["images"] if i["id"] == a["image_id"])
        x, y, w, h = a["bbox"]
        assert any(
            x0 * im["width"] - 1e-6 <= x
            and x + w <= x1 * im["width"] + 1e-6
            and y0 * im["height"] - 1e-6 <= y
            and y + h <= y1 * im["height"] + 1e-6
            for x0, y0, x1, y1 in layouts[a["image_id"] - 1]["boxes"]
        )


def test_resume_after_truncation_equals_uninterrupted(cli, pool_dir, tmp_path):
    full, part = tmp_path / "full.jsonl", tmp_path / "part" / "layouts.jsonl"
    args = ["layout", "generate", "--pool", pool_dir, "--count", 6, "--seed", 9, "--workers", 0, *POOL_FLAGS]
    assert cli(*args, "--out", full).exit_code == 0
    assert cli(*args, "--out", part).exit_code == 0
    data = part.read_bytes()
    cut = data.index(b"\n", data.index(b"\n") + 1) + 1 + 17  # 2 lines + a partial third
    part.write_bytes(data[:cut])
    assert cli(*args, "--out", part).exit_code == 2  # refuses to clobber without a flag
    assert cli(*args, "--out", part, "--resume").exit_code == 0
    assert part.read_bytes() == full.read_bytes()


def test_sigint_flushes_then_resume_completes(cli, pool_dir, tmp_path, monkeypatch):
    out = tmp_path / "ds"
    real = runner.run_task

    def interrupt_at_2(page):
        if page == 2:
            os.kill(os.getpid(), signal.SIGINT)
        return real(page)

    monkeypatch.setattr(runner, "run_task", interrupt_at_2)
    assert _run(cli, pool_dir, out, "--workers", 0).exit_code == 130
    assert len((out / "layouts.jsonl").read_text().splitlines()) == 3
    monkeypatch.setattr(runner, "run_task", real)
    assert _run(cli, pool_dir, out, "--workers", 0, "--resume").exit_code == 0
    ref = tmp_path / "ref"
    assert _run(cli, pool_dir, ref, "--workers", 0).exit_code == 0
    for rel in ("layouts.jsonl", "annotations/instances_default.json"):
        assert (out / rel).read_bytes() == (ref / rel).read_bytes()


def test_failed_pages_are_skipped_then_counted(cli, pool_dir, tmp_path):
    for p in pool_dir.glob("*/*.jpg"):
        p.unlink()
    out = tmp_path / "bad"
    r = _run(cli, pool_dir, out, "--workers", 0)
    assert r.exit_code == 1 and "page 0 skipped" in r.output + (r.stderr or "")
    r = _run(cli, pool_dir, tmp_path / "tolerant", "--workers", 0, "--max-failures", 10)
    assert r.exit_code == 0
    assert not (tmp_path / "tolerant" / "layouts.jsonl").read_text()
