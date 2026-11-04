# mesh-bestfit

Mesh-candidate BestFit synthetic document-layout generator: COCO dataset in, synthetic
layout-detection pages (images + YOLO/COCO labels) out. Spec:
`SPEC.md`.

```bash
uv sync --extra augment
uv run pytest && uv run ruff check . && uv run mypy src
```

## Quickstart on DocLayNet-small (`datasets/doclaynet/`)

The local DocLayNet-small is per-page JSON, not COCO, so convert one split to a COCO dataset
folder first (`annotations/instances_train.json` plus `images/train`, a symlink to the PNGs):

```bash
uv run python scripts/doclaynet_to_coco.py datasets/doclaynet/train --out work/doclaynet-train
mesh-bestfit pool build --dataset work/doclaynet-train --out work/pool
mesh-bestfit pool augment --pool work/pool --min-count 100 --aug-times 50 --seed 0   # optional
mesh-bestfit run --pool work/pool --count 1000 --out work/dataset --seed 0 --workers 4
mesh-bestfit viz --layouts work/dataset/layouts.jsonl --out grid.png --labels
mesh-bestfit stats --layouts work/dataset/layouts.jsonl --json
```

`pool build --dataset DIR [--subset NAME]` reads any such folder (`images/<subset>/` +
`annotations/instances_<subset>.json`), e.g. a CVAT export or a previous `run` output, so those
skip the converter. `--coco FILE --images-dir DIR` still takes a single json and a flat images folder.

The dataset follows CVAT's COCO 1.0 layout (`--subset` names the folder, default `default`):

```text
work/dataset/
  images/<subset>/00000000.jpg
  labels/<subset>/00000000.txt          # YOLO, mirrors images/
  annotations/instances_<subset>.json   # COCO
  classes.txt  layouts.jsonl  run.json
```

Global flags (`-v`, `-q`, `--preset`, `--config`, `--log-format`) go before the subcommand.
`--dry-run` validates and prints the plan; `--resume` continues after Ctrl+C (exit 130).
