# mesh-bestfit: usage guide

`mesh-bestfit` turns a COCO-annotated document dataset into synthetic layout-detection pages
(images + YOLO / COCO labels). It crops every annotated element into a pool, packs elements onto
blank pages with the Mesh-candidate BestFit algorithm, and renders the result.

For the design and algorithm details see [SPEC.md](SPEC.md).

## Contents

1. [Install](#install)
2. [Quickstart](#quickstart)
3. [How it works](#how-it-works)
4. [Commands](#commands)
5. [Output layout](#output-layout)
6. [Settings: presets, TOML, flags](#settings-presets-toml-flags)
7. [Reproducibility, resume, interrupt](#reproducibility-resume-interrupt)
8. [Exit codes](#exit-codes)
9. [Troubleshooting](#troubleshooting)

## Install

Requires Python 3.10 or newer. From a clone of the repo:

```bash
uv sync                    # core
uv sync --extra augment    # also installs OpenCV, needed only by `pool augment`
uv run mesh-bestfit --help
```

[Unverified] Only Linux has been tested; macOS and Windows have not been run.

## Quickstart

Input: a COCO dataset folder with `images/<subset>/` and `annotations/instances_<subset>.json`
(a CVAT COCO 1.0 export has this shape, and so does the output of a previous `run`).

```bash
# 1. Crop every instance into an element pool
mesh-bestfit pool build --dataset ./coco-dataset --out ./pool

# 2. (optional) add appearance variants for rare categories
mesh-bestfit pool augment --pool ./pool --min-count 100 --aug-times 50 --seed 0

# 3. Generate 1000 pages
mesh-bestfit run --pool ./pool --count 1000 --out ./dataset --seed 0 --workers 4

# 4. Inspect
mesh-bestfit viz   --layouts ./dataset/layouts.jsonl --out grid.png --labels
mesh-bestfit stats --layouts ./dataset/layouts.jsonl --json
```

Preview any generation command without writing anything:

```bash
mesh-bestfit run --pool ./pool --count 1000 --out ./dataset --dry-run
```

## How it works

| Stage | Command | Reads | Writes |
| --- | --- | --- | --- |
| 1. Pool | `pool build` | COCO JSON + source images | `pool/<category>/<id>.jpg`, `pool/pool.json` |
| 2. Augment (optional) | `pool augment` | `pool.json` | variant crops, updated `pool.json` |
| 3. Layout | `layout generate` | `pool.json` | `layouts.jsonl` |
| 4. Render | `render` | `layouts.jsonl`, `pool.json` | images + labels |
| 3+4 | `run` | `pool.json` | images + labels + `layouts.jsonl` |
| Inspect | `viz`, `stats` | `layouts.jsonl` | PNG grid, metrics |

Use `run` unless you want to generate layouts once and render them several times with different
render settings (size, label format, `--keep-aspect`, subset name).

## Commands

Global flags go **before** the subcommand:

```text
mesh-bestfit [--version] [-v | -q] [--log-format text|json] [--preset paper2|repo] [--config FILE] COMMAND ...
```

| Flag | Meaning |
| --- | --- |
| `-v` / `-q` | DEBUG / WARNING-only logging |
| `--log-format json` | one JSON object per log line on stderr |
| `--preset` | `paper2` (default) or `repo`, see [Presets](#presets) |
| `--config FILE` | TOML settings file, see [Config file](#config-file) |

Logs, the seed and progress bars go to stderr. stdout carries only machine-readable output
(`--dry-run`, `stats --json`). Progress bars are off when stderr is not a terminal.

### `pool build`

Crops every COCO instance into `<out>/<category>/<id>.jpg` and writes `<out>/pool.json`.

```bash
# from a dataset folder
mesh-bestfit pool build --dataset ./coco-dataset [--subset train] --out ./pool

# or from a single json + a flat image folder
mesh-bestfit pool build --coco annotations.json --images-dir ./images --out ./pool \
    --min-size 0.01 --exclude-category Date --exclude-category Section
```

| Flag | Meaning |
| --- | --- |
| `--out DIR` | required; created if missing |
| `--dataset DIR` | dataset folder; replaces `--coco` and `--images-dir` |
| `--subset NAME` | with `--dataset` only; default: the only `instances_*.json` found (none or several: exit 3) |
| `--coco FILE`, `--images-dir DIR` | use both together when not using `--dataset` |
| `--min-size X` | drop elements whose width or height is below this fraction of the page (default `0.01`) |
| `--exclude-category NAME` | repeatable; a name not in the COCO file logs a warning |

Annotation `id`s must be unique across all categories, and each image's real size must match its
COCO `width` / `height`; otherwise the command exits 3.

### `pool augment`

Adds appearance variants for rare categories. Needs the `augment` extra
(without it: exit 2). Rewrites `pool.json` in place.

```bash
mesh-bestfit pool augment --pool ./pool --min-count 100 --aug-times 50 --seed 0 [--no-flip]
```

| Flag | Meaning |
| --- | --- |
| `--pool PATH` | `pool.json` or its folder |
| `--min-count N` | a category is rare when its element count is `<= N` (default 100) |
| `--aug-times N` | variants per element (default 50) |
| `--seed INT` | default: random, printed to stderr |
| `--flip` / `--no-flip` | random mirror; default on for `paper2`, off for `repo` |

Variants are picked at render time (one of original or variants, uniformly) when
`--use-variants` is on. They do not change how often an element is sampled.

### `layout generate`

Writes `layouts.jsonl`, one page per line, coordinates normalized to [0, 1].

```bash
mesh-bestfit layout generate --pool ./pool --count 1000 --out ./work/layouts.jsonl \
    --seed 7 --workers 8
```

`run.json` is written next to `--out`.

### `render`

Renders a saved `layouts.jsonl`.

```bash
mesh-bestfit render --layouts ./work/layouts.jsonl --pool ./pool --out ./dataset \
    --width 1200 2000 --aspect 0.7 1.5 --max-height 3000 \
    --label-format yolo --label-format coco
```

`--seed` overrides the render seed stored in each layout record.

### `run`

`layout generate` + `render` in one streamed pass. Takes all layout and render flags and also
writes `<out>/layouts.jsonl`.

```bash
mesh-bestfit run --pool ./pool --count 100000 --out ./dataset --seed 0 --workers 16
```

### `viz`

PNG grid of the first `rows × cols` layouts (600 px cells).

```bash
mesh-bestfit viz --layouts layouts.jsonl --out grid.png --rows 2 --cols 2 --labels [--pool ./pool]
```

`--labels` prints category ids; `--pool` pastes pool crops instead of bare boxes.

### `stats`

Density, alignment and category counts of a layout file.

```bash
mesh-bestfit stats --layouts layouts.jsonl          # table on stderr
mesh-bestfit stats --layouts layouts.jsonl --json   # JSON on stdout
```

### Flags shared by `layout generate`, `render`, `run`

| Flag | Meaning |
| --- | --- |
| `--seed INT` | default: random 63-bit integer, printed to stderr and saved in `run.json` |
| `--workers N` | worker processes; default `cpu_count - 1`; `0` runs inline |
| `--resume` | skip finished pages and continue; without `--seed` the seed is read from `run.json` |
| `--overwrite` | allow writing into an existing output |
| `--dry-run` | validate, print `{"settings": ..., "outputs": [...]}` as JSON, write nothing |
| `--check-files` | require every file listed in `pool.json` to exist |
| `--max-failures N` | abort (exit 1) once more than N pages fail; default 0, so the first failed page aborts |

An existing output without `--resume` or `--overwrite` is a usage error, and the two flags cannot
be combined. A page that fails (missing or corrupt crop, unreadable image) is logged with its page
index and element id, then skipped and counted.

### Layout flags (`layout generate`, `run`)

| Flag | Default (`paper2`) | Meaning |
| --- | --- | --- |
| `--candidates K` | 500 | candidate elements drawn per page |
| `--large-ratio R` | 0.99 | share of large candidates; the rest are small |
| `--on-short-stratum error\|take-all` | `error` | when the pool has fewer large or small elements than the quota: `error` exits 4, `take-all` uses what exists |
| `--max-elements N` | 35 | element cap per page; `0` = no limit |
| `--fill-threshold X` | 1e-4 | stop the page when the best fill rate falls below this |
| `--max-small N` | no cap | cap on small elements per page; `-1` = no cap |

A page ends with one `stop_reason` in `layouts.jsonl`: `max_elements`, `no_fit`,
`below_threshold`, `small_cap` or `pool_exhausted`.

### Render flags (`render`, `run`)

| Flag | Default | Meaning |
| --- | --- | --- |
| `--width LO HI` | `1200 2000` | page width range, px |
| `--aspect LO HI` | `0.7 1.5` | height / width range |
| `--max-height PX` | 3000 | page height cap |
| `--label-format F` | `yolo`, `coco` | repeatable: `yolo`, `yolo-poly`, `coco` |
| `--keep-aspect` / `--stretch` | `--stretch` | fit each crop inside its box (centered) instead of stretching it |
| `--category-offset N` | none | class id = COCO category id + N; default: contiguous 0-based ids |
| `--use-variants` / `--no-variants` | on | pick a random variant per element |
| `--subset NAME` | `default` | folder name under `images/`, `labels/` and in `instances_<name>.json` |

Label formats:

- `yolo`: `class cx cy w h`, normalized.
- `yolo-poly`: `class x0 y0 x1 y0 x1 y1 x0 y1`, normalized (the reference repo's format). Cannot be combined with `yolo`: both write `labels/<subset>/<page>.txt`.
- `coco`: `annotations/instances_<subset>.json`, bbox `[x, y, w, h]` in pixels.

With `--keep-aspect`, labels cover the pasted rectangle, not the layout box.
`--subset` must start with a letter or digit and contain only letters, digits, `.`, `_`, `-`.
Use one `--out` per subset.

## Output layout

```text
dataset/
  images/<subset>/00000012.jpg         # zero-padded page index
  labels/<subset>/00000012.txt         # yolo / yolo-poly only
  annotations/instances_<subset>.json  # coco only
  classes.txt                          # one category name per line; line number = class id
  run.json                             # versions, resolved settings, seed, pool sha256, counts
  layouts.jsonl                        # `run` only
```

- `images/` + `annotations/` follow CVAT's COCO 1.0 layout. pycocotools and Datumaro loaded a 20-page sample of the output; [Unverified] uploading it to a CVAT instance was not tried.
- With `--category-offset`, `classes.txt` has a blank line for every id gap so that line number = class id.
- `classes.txt` and the COCO file are written only after a run completes; an interrupted run has neither.

## Settings: presets, TOML, flags

Precedence, highest first: **CLI flag > `--config` TOML > preset**. A flag you do not pass never
overrides anything.

### Presets

| | `paper2` (default) | `repo` |
| --- | --- | --- |
| `--max-elements` | 35 | no limit |
| `--fill-threshold` | 1e-4 | 0 |
| `--max-small` | no cap | 5 |
| `pool augment` flip | on | off |

`repo` matches the behaviour of the original reference scripts.

### Config file

`--config mesh-bestfit.toml`, shown with the `paper2` defaults. Omit a key to keep the preset's
value; unknown keys are an error (exit 2). TOML has no null, so `max_elements = 0` means no limit
and `max_small = -1` means no cap.

```toml
[pool]
min_size = 0.01              # drop elements with w or h below this
small_threshold = 0.05       # large iff both w and h >= this
exclude_categories = []

[sampling]
candidates = 500
large_ratio = 0.99
on_short_stratum = "error"   # or "take-all"

[layout]
max_elements = 35
fill_threshold = 1e-4
shrink = [0.8, 0.95]         # per-axis contraction of each box about its center

[render]
width = [1200, 2000]
aspect = [0.7, 1.5]          # H / W
max_height = 3000
background = "white"         # any color Pillow parses
image_format = "jpg"         # or "png"
jpeg_quality = 95
use_variants = true
subset = "default"
keep_aspect = false
# category_offset = -1
label_formats = ["yolo", "coco"]

[augment]
flip = true
```

Constraints: `0 < min_size < small_threshold <= 1`, `0 < shrink[0] <= shrink[1] <= 1`,
`0 < lo <= hi` for `width` and `aspect`.

```bash
mesh-bestfit --config mesh-bestfit.toml run --pool ./pool --count 500 --out ./dataset
mesh-bestfit --preset repo run --pool ./pool --count 500 --out ./dataset
```

## Reproducibility, resume, interrupt

- The same seed, settings, pool and dependency versions give the same output for any `--workers` value. Across different NumPy/Pillow versions this is not assumed; `run.json` records the versions.
- Save the printed seed (or read it from `run.json`) to repeat a run that used a random seed.
- **Ctrl+C once**: stops submitting new pages, finishes the in-flight ones, flushes, exits 130. **Twice**: aborts immediately.
- Continue an interrupted run by repeating the same command with `--resume`:

```bash
mesh-bestfit run --pool ./pool --count 1000 --out ./dataset --resume
```

  `--resume` skips pages already done (for `render`: pages whose image exists) and reads the seed from `run.json` unless you pass `--seed`.
- Images and labels are written to `*.tmp` and renamed, so a half-written file is not left under its final name.

## Exit codes

| Code | Meaning |
| --- | --- |
| 0 | success |
| 1 | unexpected error, or failed pages exceed `--max-failures` |
| 2 | usage or config error: unknown TOML key, bad flag value, `--resume` with `--overwrite`, output exists without either flag, `pool augment` without the `augment` extra |
| 3 | invalid input data: bad COCO file, duplicate annotation ids, corrupt `pool.json`, image size differs from COCO `width` / `height`, missing file under `--check-files` |
| 4 | pool too small for the requested sampling (`--on-short-stratum error`) |
| 130 | interrupted; continue with `--resume` |

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| Exit 4: `pool has N large / M small elements, need 495 / 5` | the pool is smaller than one page's candidate draw. Add `--on-short-stratum take-all`, lower `--candidates`, or lower `--min-size` / add more source data |
| `... already exists (use --resume or --overwrite)` | pick a new `--out`, or pass `--resume` / `--overwrite` |
| `--resume without --seed needs a seed in .../run.json` | `run.json` is missing or has no seed; pass `--seed` |
| Exit 2 from `pool augment` | install the extra: `uv sync --extra augment` |
| Exit 3 on `pool build` | check the message: duplicate annotation ids, or an image whose size differs from the COCO file |
| Run aborts on the first bad page | default `--max-failures 0`; raise it to skip failures |
| Pages look sparse | pages stop on `no_fit` / `below_threshold` when the pool has few elements that fit; see `stop_reason` in `layouts.jsonl` and `mesh-bestfit stats` |
| Need to debug a worker error | `--workers 0` runs inline; add `-v` for DEBUG logs |
