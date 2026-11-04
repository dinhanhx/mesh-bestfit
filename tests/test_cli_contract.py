import json
import shutil

from conftest import POOL_FLAGS


def test_pool_build_dataset_folder(cli, pool_dir, coco_fixture, tmp_path):
    ds = tmp_path / "ds"
    (ds / "annotations").mkdir(parents=True)
    shutil.copytree(tmp_path / "images", ds / "images/train")
    shutil.copy(coco_fixture, ds / "annotations/instances_train.json")
    r = cli("pool", "build", "--dataset", ds, "--out", tmp_path / "p")
    assert r.exit_code == 0, r.output
    a, b = (json.loads((d / "pool.json").read_text()) for d in (pool_dir, tmp_path / "p"))
    assert (a["categories"], a["elements"]) == (b["categories"], b["elements"])
    shutil.copy(coco_fixture, ds / "annotations/instances_val.json")
    shutil.copytree(ds / "images/train", ds / "images/val")
    assert cli("pool", "build", "--dataset", ds, "--out", tmp_path / "p2").exit_code == 3
    assert cli("pool", "build", "--dataset", ds, "--subset", "val", "--out", tmp_path / "p2").exit_code == 0
    assert cli("pool", "build", "--dataset", ds, "--coco", coco_fixture, "--out", tmp_path / "p3").exit_code == 2
    assert (
        cli(
            "pool",
            "build",
            "--coco",
            coco_fixture,
            "--images-dir",
            tmp_path / "images",
            "--subset",
            "val",
            "--out",
            tmp_path / "p3",
        ).exit_code
        == 2
    )


def test_version_and_help(cli):
    assert cli("--version").exit_code == 0
    assert cli().exit_code in (0, 2)  # no_args_is_help
    assert cli("layout", "generate", "--help").exit_code == 0


def test_dry_run_writes_nothing(cli, pool_dir, tmp_path):
    out = tmp_path / "never"
    r = cli("run", "--pool", pool_dir, "--count", 3, "--out", out, "--dry-run", *POOL_FLAGS)
    assert r.exit_code == 0 and not out.exists()
    assert json.loads(r.stdout)["settings"]["sampling"]["candidates"] == 8
    r = cli(
        "layout", "generate", "--pool", pool_dir, "--count", 3, "--out", tmp_path / "l.jsonl", "--dry-run", *POOL_FLAGS
    )
    assert r.exit_code == 0 and not (tmp_path / "l.jsonl").exists()


def test_exit_2_unknown_toml_key_and_bad_flags(cli, pool_dir, tmp_path):
    bad = tmp_path / "bad.toml"
    bad.write_text("[layout]\nnot_a_key = 1\n")
    base = ["layout", "generate", "--pool", pool_dir, "--count", 1, "--out", tmp_path / "o.jsonl"]
    assert cli("--config", bad, *base, *POOL_FLAGS).exit_code == 2
    assert cli("--preset", "nope", *base).exit_code == 2
    assert cli(*base, "--workers", -1).exit_code == 2
    assert cli(*base, "--resume", "--overwrite").exit_code == 2
    assert cli(*base, "--large-ratio", 2).exit_code == 2


def test_cli_beats_toml_beats_preset(cli, pool_dir, tmp_path):
    cfg = tmp_path / "c.toml"
    cfg.write_text("[sampling]\ncandidates = 8\nlarge_ratio = 0.75\n[layout]\nmax_elements = 7\n")
    base = ["--config", cfg, "run", "--pool", pool_dir, "--count", 1, "--out", tmp_path / "d", "--dry-run"]
    s = json.loads(cli(*base).stdout)["settings"]
    assert (s["sampling"]["candidates"], s["layout"]["max_elements"]) == (8, 7)
    s = json.loads(cli(*base, "--max-elements", 0).stdout)["settings"]  # CLI wins; 0 = no limit
    assert s["layout"]["max_elements"] is None
    s = json.loads(
        cli(
            "--preset",
            "repo",
            "run",
            "--pool",
            pool_dir,
            "--count",
            1,
            "--out",
            tmp_path / "d",
            "--dry-run",
            *POOL_FLAGS,
        ).stdout
    )["settings"]
    assert (s["layout"]["max_elements"], s["layout"]["max_small"]) == (None, 5)


def test_exit_3_invalid_input(cli, tmp_path, coco_fixture, pool_dir):
    (tmp_path / "images").mkdir(exist_ok=True)
    junk = tmp_path / "junk.json"
    junk.write_text("{}")
    assert (
        cli("pool", "build", "--coco", junk, "--images-dir", tmp_path / "images", "--out", tmp_path / "p2").exit_code
        == 3
    )
    d = json.loads(coco_fixture.read_text())
    d["annotations"][1]["id"] = d["annotations"][0]["id"]
    dup = tmp_path / "dup.json"
    dup.write_text(json.dumps(d))
    assert (
        cli("pool", "build", "--coco", dup, "--images-dir", tmp_path / "images", "--out", tmp_path / "p3").exit_code
        == 3
    )
    m = json.loads((pool_dir / "pool.json").read_text())
    m["elements"][0]["category_id"] = 99
    (pool_dir / "pool.json").write_text(json.dumps(m))
    assert cli("layout", "generate", "--pool", pool_dir, "--count", 1, "--out", tmp_path / "o.jsonl").exit_code == 3


def test_exit_4_pool_too_small_and_take_all(cli, pool_dir, tmp_path):
    base = ["layout", "generate", "--pool", pool_dir, "--count", 2, "--workers", 0]
    assert cli(*base, "--out", tmp_path / "a.jsonl").exit_code == 4  # default K=500
    r = cli(*base, "--out", tmp_path / "b.jsonl", "--on-short-stratum", "take-all")
    assert r.exit_code == 0 and len((tmp_path / "b.jsonl").read_text().splitlines()) == 2


def test_stats_and_viz(cli, pool_dir, tmp_path):
    lay = tmp_path / "l.jsonl"
    assert (
        cli(
            "layout",
            "generate",
            "--pool",
            pool_dir,
            "--count",
            4,
            "--out",
            lay,
            "--seed",
            1,
            "--workers",
            0,
            *POOL_FLAGS,
        ).exit_code
        == 0
    )
    r = cli("stats", "--layouts", lay, "--json")
    s = json.loads(r.stdout)
    assert s["pages"] == 4 and 0 < s["density"] <= 1
    assert cli("viz", "--layouts", lay, "--out", tmp_path / "g.png", "--labels").exit_code == 0
    assert cli("viz", "--layouts", lay, "--out", tmp_path / "c.png", "--pool", pool_dir).exit_code == 0
    assert (tmp_path / "g.png").stat().st_size > 0


def test_check_files_exit_3(cli, pool_dir, tmp_path):
    next(pool_dir.glob("*/*.jpg")).unlink()
    base = [
        "layout",
        "generate",
        "--pool",
        pool_dir,
        "--count",
        1,
        "--out",
        tmp_path / "o.jsonl",
        "--workers",
        0,
        *POOL_FLAGS,
    ]
    assert cli(*base, "--check-files").exit_code == 3


def test_resume_reads_seed_from_run_json(cli, pool_dir, tmp_path):
    args = ["layout", "generate", "--pool", pool_dir, "--count", 4, "--workers", 0, *POOL_FLAGS]
    ref, part = tmp_path / "ref" / "l.jsonl", tmp_path / "part" / "l.jsonl"
    assert cli(*args, "--out", ref, "--seed", 5).exit_code == 0
    assert cli(*args, "--out", part, "--seed", 5).exit_code == 0
    lines = part.read_bytes().splitlines(keepends=True)
    part.write_bytes(b"".join(lines[:2]))
    assert cli(*args, "--out", part, "--resume").exit_code == 0  # no --seed given
    assert part.read_bytes() == ref.read_bytes()
    assert cli(*args, "--out", tmp_path / "x" / "l.jsonl", "--resume").exit_code == 2  # no run.json
