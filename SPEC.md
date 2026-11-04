# mesh-bestfit: Python CLI Specification (Mesh-candidate Bestfit)

Oct 31, 2026 · @Anh Vu Dinh

## 1. Scope and goals

`mesh-bestfit` is a Python CLI that turns a COCO-annotated document dataset into synthetic layout-detection pages: it crops an element pool, places elements with Mesh-candidate BestFit, renders the pages, and writes labels. It replaces the seven scripts and one notebook in the uploaded repo snapshot.

&#91;Inference\] Sections 2 and 3 cite the repo files or the two papers directly. Everything else (command names, schemas, the speed-up in section 6) is a design proposal. Items I could not verify are tagged \[Unverified\] or \[Inference\].

**Goals**

- G1: reproduce the reference algorithm, with a `paper2` preset (default) and a `repo` preset that matches the snapshot's behavior.
- G2: run on Python 3.10 and newer; every dependency must resolve on 3.10.
- G3: same seed gives the same output for any `--workers` value.
- G4: stream, resume, and parallelize by process; no torch or torchvision dependency (the repo imports both only for `box_iou`).
- G5: one installable package, typed, with tests.

**Non-goals:** model training, downloading M6Doc or DocBank, HTML-based synthesis, learned layout generators, text rendering.

| Stage | Command | Reads | Writes | Repo equivalent |
| --- | --- | --- | --- | --- |
| 1. Pool | `pool build` | COCO JSON, source images (or a COCO dataset folder) | `pool/` crops, `pool.json` | none (README says to crop manually) |
| 2. Augment (optional) | `pool augment` | `pool.json` | variant crops, updated `pool.json` | `augmentation.py` |
| 3. Layout | `layout generate` | `pool.json` | `layouts.jsonl` | `bestfit_generator.py`, `combine_layouts.py` |
| 4. Render | `render` | `layouts.jsonl`, `pool.json` | `images/`, labels | `rendering.py` |
| 5. End to end | `run` | `pool.json` | `images/`, labels, `layouts.jsonl` | stages 3 and 4, streamed |
| 6. Inspect | `viz`, `stats` | `layouts.jsonl` | PNG grid, metrics JSON | `visualize.ipynb` |

## 2. Algorithm source

The repo code and Paper 2 (Appendix E) agree on the core loop; they differ on stop rules and pool source, so the spec ships two presets. Sources: the uploaded repo snapshot, Paper 1 (*DocLayout-YOLO: Enhancing Document Layout Analysis…*, file `…Arxiv241012628`) and Paper 2 (*DocLayout-YOLO: Axis-Attentive … Mesh-Guided Synthetic Pretraining*, file `…Arxiv261028171`).

| Item | Repo snapshot | Paper 2 (App. E) | Paper 1 (App. A) |
| --- | --- | --- | --- |
| Element source | M6Doc-test annotations (README) | DocBank train split, minus Date and Section | M6Doc test, 74 categories |
| Min size ε | skip if w/W or h/H < 0.01 (`process.py:19`) | ε = 0.01 | not stated in text read |
| Large/small split | large iff both ≥ 0.05 (`process.py:23`) | τs = 0.05 | stratified by size |
| Candidate set | 500: `int(500*0.99)` large + `int(500*0.01)` small (`bestfit_generator.py:25-27`) | K = 500, ρ = 0.99, ⌊ρK⌋ large | stratified sample, K not stated |
| Start element | random candidate, random center inside page (`:32-35`) | random draw; position not stated | random element, "a certain position" |
| Score | fits ⇒ element area ÷ grid area (`:68-73`) | same, Eq. 13 | max fill rate |
| Stop | no fitting pair, or small-element cap | N = 35 elements, or best fill < 1e-4 | threshold fr\_thr named, value not stated in text read |
| Small-element cap | at most 5 small elements (`:38`, `:82-85`) | not mentioned \[Unverified whether kept\] | Mininum = 5 (A.2.3) |
| Placement | grid center (`:91-92`) | grid center | grid center |
| Post-scale | w and h × U(0.8, 0.95), independent (`base.py:16`) | same | random central scaling |
| Canvas | W \~ U\_int(1200, 2000), H = min(3000, int(W·r)), r \~ U(0.7, 1.5) (`rendering.py:22-26`, `process.py:39-41`) | same | not stated in text read |
| Render | PyMuPDF 1.23.7, `keep_proportion=False` (stretch) | resize to box | not stated in text read |
| Labels | per-image `.txt`, 8 values per box, class = COCO id − 1 | not stated | not stated |
| Augmentation | brightness/contrast, resized crop, Sobel, elastic, Gaussian noise; no flip (`augmentation.py:31-37`) | not described in App. E | adds random flip, p = 0.5 |

**Repo behaviors the spec changes on purpose**

- The mesh is every lattice rectangle from the tick product, filtered by IoU sum = 0 in float32 (`bestfit_generator.py:52-61`). The spec uses float64 and an equivalent faster enumeration (section 6).
- `random.seed(timestamp)` runs at import (`:13`) and workers come from `multiprocessing.Pool`. \[Inference\] Under the fork start method, workers may share one RNG state and emit duplicate layouts. The spec seeds each page from a `SeedSequence`.
- Output names are `time.time()` strings (`:116`). \[Inference\] Parallel workers can collide on a name. The spec uses zero-padded page indices.
- `OUTPUT_DIR` and `INSTANCE2PATHLIST` are globals set under `__main__`, which only works when workers inherit them. The spec passes explicit, picklable config.
- `random.sample` raises `ValueError` when a stratum is smaller than its quota. The spec makes that policy explicit (section 4).

## 3. Requirements

`requires-python = ">=3.10"`. On 3.10 several libraries stop at older releases than on 3.12+ (NumPy 2.2.6 instead of 2.5.4, Pillow 12.3.0 instead of 13.0.0), so the dependency floors below are the newest versions I confirmed on 3.10, and newer Pythons resolve to newer releases automatically.

| Package | Role | Newest, any Python | Newest on 3.10 | Spec constraint |
| --- | --- | --- | --- | --- |
| numpy | arrays, mesh, fit table | 2.5.4 (needs ≥3.12) | 2.2.6 | `numpy>=2.2.6` |
| pillow | crop, resize, composite | 13.0.0 (needs ≥3.11) | 12.3.0 | `pillow>=12.3` |
| typer | CLI | 0.27.3 | 0.27.3 | `typer>=0.27` |
| pydantic | config and schema validation | 2.14.2 | 2.14.2 | `pydantic>=2.14` |
| orjson | COCO and JSONL I/O | 3.13.0 | 3.13.0 | `orjson>=3.13` |
| rich | progress bars, tables | 15.0.0 | 15.0.0 | `rich>=15` |
| tomli | TOML config, 3.10 only | 2.5.0 | 2.5.0 | `tomli>=2.5; python_version < "3.11"` |
| opencv-python-headless | `[augment]` extra | 5.0.0.93 | 5.0.0.93 | `>=5.0,<6` \[Unverified: OpenCV 5 API vs 4.x\] |
| pytest, hypothesis, pytest-benchmark | tests | 9.2.0, 6.170.14 (≥3.11), 5.3.0 | 9.2.0, 6.168.5, 5.3.0 | dev group |
| ruff, mypy | lint, types | 0.17.3, 2.4.1 | same | dev group |

Source: PyPI JSON API, e.g. [numpy](https://pypi.org/pypi/numpy/json), queried 2026-10-31. `tomllib` is in the standard library from 3.11, hence `tomli` on 3.10.

**Deliberately not used**

- PyMuPDF: the repo pins 1.23.7 only to place images on a blank page; Pillow does the same. The newest PyMuPDF on 3.10 is 1.28.2 and 2.0 needs ≥3.11.
- albumentations: the repo calls `alpha_affine` and `var_limit` (`augmentation.py:35-36`), arguments from an older API. \[Unverified\] whether they exist in 2.x; the newest release is 2.0.8 (2025-05-27). Augmentation is reimplemented with NumPy and OpenCV, so it cannot break on a signature change. The operation strengths are the implementation's own choice, because the repo's `augmentation.py` was not available then; they are not the repo's values \[Unverified\].
- torch, torchvision: used only for `box_iou`; the SAT mesh in section 6 replaces it.
- numba, scipy: not needed. In my prototype the mesh and fit steps took milliseconds on a synthetic pool (section 6).

**Platforms:** Linux, macOS, Windows; CPU only. Multiprocessing uses the `spawn` context, so nothing may rely on inherited globals.

## 4. CLI interface

One `mesh-bestfit` entry point with subcommands, built on Typer. Option precedence is CLI flag, then `--config` TOML, then preset defaults (`paper2` unless `--preset repo`).

```text
mesh-bestfit [--version] [-v|-q] [--log-format text|json] [--config PATH] [--preset paper2|repo] COMMAND

COMMAND:
  pool build       COCO + source images  -> element pool + pool.json
  pool augment     rare-category variants (optional)
  layout generate  pool.json             -> layouts.jsonl
  render           layouts.jsonl + pool  -> images + labels
  run              layout generate + render, streamed
  viz              layouts.jsonl         -> PNG grid of boxes (or crops)
  stats            layouts.jsonl         -> density / alignment / category counts
```

Shared flags on `layout generate`, `render`, `run`: `--seed INT` (default: a random 63-bit integer, printed and saved to `run.json`; on `render` it overrides the render seed stored in each layout record), `--workers INT` (default `os.cpu_count() - 1`, `0` = run inline), `--resume` (skip finished page indices; without `--seed` the seed is read back from `run.json`), `--overwrite`, `--dry-run`, `--check-files` (require every pool file to exist), `--max-failures N` (default 0). `--resume` and `--overwrite` are mutually exclusive, and an existing output without either flag is a usage error. `--pool` accepts `pool.json` or the folder that holds it. Global flags (`-v`, `-q`, `--preset`, `--config`, `--log-format`) go before the subcommand.

```bash
# 1. pool: crop every COCO instance, normalized w/h by its source image
mesh-bestfit pool build --coco annotations.json --images-dir ./images --out ./pool \
    --min-size 0.01 --exclude-category Date --exclude-category Section
# or, from a COCO dataset folder (images/<subset>/, annotations/instances_<subset>.json):
mesh-bestfit pool build --dataset ./coco-dataset [--subset train] --out ./pool

# 2. optional: variants for categories with <= 100 elements
mesh-bestfit pool augment --pool ./pool --min-count 100 --aug-times 50 --seed 0

# 3. layouts only (JSONL, one page per line)
mesh-bestfit layout generate --pool ./pool/pool.json --count 1000 --out layouts.jsonl \
    --seed 7 --workers 8 --candidates 500 --large-ratio 0.99 \
    --max-elements 35 --fill-threshold 1e-4

# 4. render a saved layout file
mesh-bestfit render --layouts layouts.jsonl --pool ./pool/pool.json --out ./dataset \
    --width 1200 2000 --aspect 0.7 1.5 --max-height 3000 --label-format yolo --label-format coco

# or 3+4 in one pass
mesh-bestfit run --pool ./pool/pool.json --count 100000 --out ./dataset --seed 0 --workers 16

# inspect
mesh-bestfit viz --layouts layouts.jsonl --out grid.png --rows 2 --cols 2 --labels
mesh-bestfit stats --layouts layouts.jsonl --json
```

**Flag notes**

- `--preset repo` sets `--max-elements` to none, `--fill-threshold 0` with strict `>`, `--max-small 5`, and `--no-flip` in `pool augment`. `--preset paper2` sets 35, `1e-4`, no small cap, and keeps `--max-small` settable. A TOML merge cannot unset a preset value, so `--max-elements 0` means no limit and `--max-small -1` means no cap; the same values work in the config file (section 5).
- `--on-short-stratum error|take-all` (default `error`): what to do when the large or small stratum has fewer elements than its quota. `error` matches the repo's `random.sample` failure; `take-all` uses what exists and logs one warning. The choice reaches `generate_layout` as `LayoutParams.on_short_stratum` (section 6); under `take-all` each quota is capped at its stratum size.
- `--label-format` is repeatable: `yolo` (`class cx cy w h`, normalized), `yolo-poly` (repo-compatible `class x0 y0 x1 y0 x1 y1 x0 y1`), `coco` (`annotations/instances_<subset>.json`). Default: `yolo` and `coco`.
- `pool augment` treats a category as rare when its count is `<= --min-count`, as the repo does (`augmentation.py:75`). Paper 1 says "less than 100"; the boundary differs by one.
- Class ids written to labels are 0-based and listed in `classes.txt`. By default they are contiguous, in category-id order. The repo writes `COCO id - 1` (`bestfit_generator.py:106`); `--category-offset -1` reproduces that (class = COCO id + N), and `classes.txt` then gets a blank line for every id gap, so line number = class id.
- `render` and `run` stretch each crop to its box by default (`--stretch`, the repo's behavior). `--keep-aspect` instead fits each crop inside its box, centered, keeping its aspect ratio; labels then cover the pasted rectangle, not the layout box.
- `--subset NAME` (`render` and `run`; default `default`) names the subset folder: `images/<subset>/`, `labels/<subset>/` and `annotations/instances_<subset>.json` (section 5).
- `pool build --dataset DIR` replaces `--coco` and `--images-dir` (giving both is a usage error, exit 2): it reads `DIR/annotations/instances_<subset>.json` and `DIR/images/<subset>/`. Without `--subset` the one `instances_*.json` is used; none or several exit 3. `--subset` without `--dataset` is a usage error.
- `pool build` writes `<category>/<id>.jpg` (JPEG quality 95; the category name is made filename-safe). It clips each bbox to its image, normalizes by the COCO `width` and `height` (a mismatch with the real image size exits 3), skips instances below `--min-size`, and warns about `--exclude-category` names that are not in the COCO file. `pool.json` keeps every non-excluded category, including one with no elements.
- `pool augment` seeds each element from `SeedSequence(seed, spawn_key=(element_index,))`, writes variants to `<category>/aug/<id>/<id>_NNNN.jpg`, replaces that element's `variants` list, and rewrites `pool.json` in place. Each of the five operations runs with p = 0.5 and at least one always runs; with flip enabled, p = 0.5 mirrors the crop.
- `viz` draws the first `rows × cols` layouts in 600 px cells; `--pool` pastes pool crops instead of bare boxes. `stats` prints its table to stderr unless `--json` is set, so stdout stays machine-readable.

**Exit codes**

| Code | Meaning |
| --- | --- |
| 0 | success |
| 1 | unexpected error, or failed pages exceed `--max-failures` |
| 2 | usage or config error (Typer/Click default): unknown TOML key, bad flag value, `--resume` with `--overwrite`, existing output without either flag, `pool augment` without the `[augment]` extra |
| 3 | invalid input data (bad COCO file, duplicate annotation ids, corrupt pool manifest, image size different from its COCO `width` and `height`) |
| 4 | pool too small for the requested sampling (`--on-short-stratum error`) |
| 130 | interrupted; completed pages are flushed and `--resume` continues |

## 5. Data model and I/O formats

Three files carry state between stages: `pool.json` (what can be placed), `layouts.jsonl` (where it goes, normalized), and the rendered dataset. All coordinates in pool and layout files are normalized to \[0, 1\]; pixels exist only in `render` output.

**`pool.json`** replaces the repo's folder walk plus `map_dict.json`. `w` and `h` are divided by the source image's width and height, as in `process.py:18-22`. `id` is the COCO annotation id and must be unique across all categories: the repo's `map_dict` is keyed by file stem alone (`map_dict.py:26,34`), so a repeated id across categories silently overwrites.

```json
{
  "version": 1,
  "source": {"coco": "annotations.json", "sha256": "…"},
  "categories": [{"id": 1, "name": "paragraph"}],
  "elements": [
    {"id": "727", "category_id": 1, "w": 0.31, "h": 0.04,
     "path": "paragraph/727.jpg",
     "variants": ["paragraph/aug/727/727_0000.jpg"]}
  ]
}
```

**`layouts.jsonl`**, one page per line, boxes after the final shrink, `xyxy`:

```json
{"page": 12, "seed": {"entropy": 7, "spawn_key": [12]}, "preset": "paper2",
 "boxes": [[0.12, 0.08, 0.61, 0.33], [0.64, 0.08, 0.90, 0.51]],
 "category_ids": [3, 1], "element_ids": ["727", "919"],
 "stop_reason": "max_elements"}
```

`stop_reason` is one of `max_elements`, `no_fit`, `below_threshold`, `small_cap`, `pool_exhausted`. The repo's combined JSON (`boxes`, `categories`, `relpaths`) is not produced.

**Rendered dataset**

```text
dataset/
  images/<subset>/00000012.jpg         # zero-padded page index; <subset> defaults to "default"
  labels/<subset>/00000012.txt         # per --label-format yolo | yolo-poly; mirrors images/
  annotations/instances_<subset>.json  # per --label-format coco; bbox = [x, y, w, h] in pixels
  classes.txt                # one category name per line, 0-based order
  run.json                   # tool + dependency versions, resolved config, seed, pool sha256, counts
  layouts.jsonl              # `run` only: the layouts it rendered (resume, COCO replay, viz, stats need it)
```

`images/` and `annotations/` follow CVAT's COCO 1.0 layout (`images/<subset_name>/`, `annotations/instances_<subset_name>.json`; Datumaro, which CVAT imports with, reads the same tree). `labels/<subset>/` mirrors `images/<subset>/`, which is where YOLO trainers look for labels. `--subset` takes a name that starts with a letter or digit and holds only letters, digits, `.`, `_` and `-`. One `--out` holds one subset: `classes.txt`, `run.json` and `layouts.jsonl` sit at the root, so a second subset needs its own `--out`, and the two subset paths can then be copied together.

The COCO file has the cocodataset.org top-level keys `info`, `licenses`, `images`, `annotations` and `categories`. `info` holds `description`, `version` and `contributor` and no timestamp, so the bytes depend only on the inputs. Each image has `id` (page index + 1), `width`, `height` and `file_name`, the bare name inside `images/<subset>/`. Each annotation has `id` (from 1), `image_id`, `category_id`, `segmentation`, `area` (= w·h), `bbox` and `iscrowd` (= 0). `segmentation` is `[]`: CVAT's page lists "Bounding Boxes (if the `segmentation` field is empty)" as importable, so the instances arrive as boxes, not polygons. Each category has `id`, `name` and `supercategory` (`""`); ids are the source COCO ids, not the 0-based class ids, and an id of 0 logs a warning because Datumaro counts category id 0 as "not specified". Only pages whose image exists are listed. The file is rebuilt at the end from `layouts.jsonl` by replaying each page's render draws (section 7), so memory stays flat and `--resume` needs no per-page state; crop headers are read only with `keep_aspect`. `classes.txt` and the COCO file are not written when a run is interrupted.

Checked by loading a 20-page dataset rendered from the DocLayNet-small train pool: pycocotools (`COCO(...)`) read 20 images, 223 annotations and 11 categories, and Datumaro's `coco_instances` importer read subset `train` with 20 items, 11 labels, images found under `images/train/` and bbox annotations with the right labels. Uploading a zip to a CVAT instance was not tried \[Unverified\].

**Config file** (`--config mesh-bestfit.toml`), shown with `paper2` defaults. TOML has no null and a merge cannot unset a preset value, so `max_elements = 0` means no limit and `max_small = -1` means no cap; omitting a key keeps the preset's value.

```toml
[pool]
min_size = 0.01              # epsilon: drop elements with w < or h < this
small_threshold = 0.05       # tau_s: large iff both w and h >= this
exclude_categories = []        # Paper 2 drops "Date" and "Section"

[sampling]
candidates = 500             # K
large_ratio = 0.99           # rho
on_short_stratum = "error"   # or "take-all"

[layout]
max_elements = 35            # N
fill_threshold = 1e-4        # fr_thr; best fill below this stops the page
shrink = [0.8, 0.95]         # per-axis contraction about the box center

[render]
width = [1200, 2000]
aspect = [0.7, 1.5]          # H / W
max_height = 3000
background = "white"
image_format = "jpg"
jpeg_quality = 95
use_variants = true          # pick a random variant (original or augmented) per element
subset = "default"           # images/<subset>/ and annotations/instances_<subset>.json
keep_aspect = false          # true: fit each crop inside its box, centered, instead of stretching
# category_offset = -1       # omit: contiguous 0-based class ids; N: class = COCO id + N
label_formats = ["yolo", "coco"]

[augment]
flip = true                  # paper2 default; the repo preset sets false
```

Validation: Pydantic v2 models, `extra="forbid"`, frozen, with range checks (`0 < min_size < small_threshold <= 1`, `0 < shrink[0] <= shrink[1] <= 1`, `0 < lo <= hi` for `width` and `aspect`, `max_height >= 1`), a `background` that Pillow parses as a color, a `subset` that is a safe folder name, and `yolo` and `yolo-poly` not both in `label_formats` (they write the same `labels/<subset>/<page>.txt`). Unknown keys exit with code 2.

## 6. Algorithm specification

Each page repeats one step: rebuild the mesh of empty grids, pick the single best candidate–grid pair by fill rate, place it at the grid center, and stop when a stop rule fires. Coordinates are normalized; the page is \[0, 1\] × \[0, 1\].

&#91;embedded content: mesh-bestfit page loop · 7 steps, 1 decision\]

A pass ends at the stop check; otherwise the chosen element is centered in its grid and the mesh is rebuilt.

**Definitions**

- Element *e*: width *w*, height *h*, category, id. Placed box: (x1, y1, x2, y2).
- Ticks: X = sorted unique {0, 1} ∪ {all placed x1, x2}; Y likewise.
- Grid: any lattice rectangle \[xa, xb\] × \[ya, yb\] with xa < xb in X and ya < yb in Y that has zero overlap area with every placed box. Touching an edge is allowed (repo: IoU sum = 0, `bestfit_generator.py:61`).
- *e* fits *g* iff w ≤ w\_g and h ≤ h\_g. fill(e, g) = (w·h) / (w\_g·h\_g) if it fits, else 0 (Paper 2, Eq. 13).

**Per-page procedure**

1. Filter the pool by ε, split it into large (both sides ≥ τs) and small, draw k\_large = ⌊Kρ⌋ and k\_small = K − k\_large without replacement. `⌊Kρ + 1e-9⌋` guards against float error; for K = 500, ρ = 0.99 it gives 495 and 5. Under `--on-short-stratum take-all` both counts are capped at their stratum sizes; under `error` a short stratum raises `ShortStratumError` (exit 4).
2. Draw the start element uniformly from the candidates. Center: cx \~ U(w/2, 1 − w/2), cy \~ U(h/2, 1 − h/2) (`bestfit_generator.py:33-34`). Paper 2 gives no position rule \[Unverified\]; the repo rule is used.
3. While fewer than N elements are placed and candidates remain: build the mesh, find the best pair, apply the stop rules below, place the element at the grid center, drop it from the candidates.
4. Shrink every box about its center by independent w and h factors from U(0.8, 0.95).
5. Emit boxes, category ids, element ids and `stop_reason`.

**Stop rules**, checked in this order each iteration:

| `stop_reason` | Condition | `repo` preset | `paper2` preset |
| --- | --- | --- | --- |
| `max_elements` | placed count ≥ N | off | N = 35 |
| `pool_exhausted` | no candidates left | on | on |
| `no_fit` | no element fits any grid | on | on |
| `below_threshold` | best fill < fr\_thr | fr\_thr = 0 (never fires) | 1e-4 |
| `small_cap` | placing a small element would exceed `max_small` | 5 | off |

&#91;Inference\] With K = 500 and ρ = 0.99 the candidate set holds exactly 5 small elements (`bestfit_generator.py:27`), and the repo breaks only when its small count exceeds 5 (`:84`), so the repo's small cap cannot fire at those defaults. It matters only when K·(1 − ρ) > 5.

**Determinism.** One `numpy.random.Generator` per page, seeded from `SeedSequence(seed, spawn_key=(page_index,))`. Draw order is fixed: large stratum, small stratum, start index, start cx then cy, then one (n, 2) draw for the shrink factors. The loop itself uses no randomness. Tie-breaks: equal fill goes to the lowest candidate index, then the lexicographically lowest (x0, y0, x1, y1). The repo's tie order follows `list(set(...))` iteration (`:52`) and is not reproduced. Any change to this order bumps `algorithm_version` in `run.json`.

**Cost and the exact speed-up.** The repo builds every point pair from the tick product (`:52-58`): with 34 placed boxes there are 70 ticks per axis, C(70,2)² = 5,832,225 candidate rectangles, then it loops candidates × grids in Python (`:66-75`). The spec keeps the same grid set and the same argmax, computed in two vectorized steps:

- Mesh: mark occupied lattice cells, take a summed-area table, and keep each index-pair rectangle whose summed occupancy is 0.
- Best pair: for the valid grids, build a table T\[a, b\] = smallest grid area among grids whose width rank ≥ a and height rank ≥ b (a 2-D suffix minimum). One `searchsorted` per candidate gives its smallest fitting area, hence its best fill. Then recover the grid with that exact area.

Verified in this session: the tested code below matched a pure-Python port of the repo's lattice-and-loop logic (float64, strict zero-overlap test) on 60/60 random grid-set cases and 40/40 random best-pair cases; the same seed gave identical boxes; 200 generated pages had no overlapping boxes. I did not run the repo's torch code. Measured on a synthetic 4,000-element pool, single process: a constructed 34-box page had 301,471 valid grids; the mesh took 149 ms, the K = 500 search 163 ms, and peak RSS was 154 MB. \[Inference\] Real pools and real layouts will differ. The implementation's tests (section 9) repeat the comparison on 20 random grid-set cases and 20 best-pair cases, and run the Hypothesis invariants on 50 pages.

```python
# meshbestfit/mesh.py
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

F64 = NDArray[np.float64]
Grids = tuple[F64, F64, F64, F64]  # x0, y0, x1, y1 (normalized)

_BLOCK = 512  # X-pair rows per block; caps the `covered` temporaries when many boxes are placed


def empty_grids(boxes: F64) -> Grids:
    """All lattice rectangles with zero overlap area against `boxes` ((n, 4) xyxy in [0, 1])."""
    xs = np.unique(np.concatenate(([0.0, 1.0], boxes[:, 0], boxes[:, 2])))
    ys = np.unique(np.concatenate(([0.0, 1.0], boxes[:, 1], boxes[:, 3])))
    occ = np.zeros((xs.size - 1, ys.size - 1), dtype=np.int32)
    for x1, y1, x2, y2 in boxes:  # edges are ticks, so searchsorted hits exactly
        occ[np.searchsorted(xs, x1) : np.searchsorted(xs, x2), np.searchsorted(ys, y1) : np.searchsorted(ys, y2)] = 1
    sat = np.zeros((xs.size, ys.size), dtype=np.int32)
    sat[1:, 1:] = occ.cumsum(0).cumsum(1)  # summed-area table over cells
    i0, i1 = np.triu_indices(xs.size, 1)
    j0, j1 = np.triu_indices(ys.size, 1)
    rows: list[NDArray[np.intp]] = []
    cols: list[NDArray[np.intp]] = []
    for s in range(0, i0.size, _BLOCK):
        r0, r1 = i0[s : s + _BLOCK, None], i1[s : s + _BLOCK, None]
        covered = sat[r1, j1] - sat[r0, j1] - sat[r1, j0] + sat[r0, j0]
        a, b = np.nonzero(covered == 0)  # touching an element's edge is allowed
        rows.append(a + s)
        cols.append(b)
    a, b = np.concatenate(rows), np.concatenate(cols)
    return xs[i0[a]], ys[j0[b]], xs[i1[a]], ys[j1[b]]


def best_pair(cw: F64, ch: F64, grids: Grids) -> tuple[float, int, int]:
    """Max over (candidate, grid) of w*h / grid_area subject to w <= gw and h <= gh.

    Returns (fill, candidate_index, grid_index); candidate_index is -1 when nothing fits.
    Ties: lowest candidate index, then lexicographically lowest (x0, y0, x1, y1).
    """
    x0, y0, x1, y1 = grids
    if x0.size == 0:
        return 0.0, -1, -1
    gw, gh = x1 - x0, y1 - y0
    area = gw * gh
    uw, wi = np.unique(gw, return_inverse=True)
    uh, hi = np.unique(gh, return_inverse=True)
    table = np.full((uw.size, uh.size), np.inf)
    np.minimum.at(table, (wi, hi), area)
    # table[a, b] = smallest grid area with width rank >= a and height rank >= b
    table = np.minimum.accumulate(table[::-1], axis=0)[::-1]
    table = np.minimum.accumulate(table[:, ::-1], axis=1)[:, ::-1]
    a = np.searchsorted(uw, cw, side="left")
    b = np.searchsorted(uh, ch, side="left")
    ok = (a < uw.size) & (b < uh.size)
    min_area = np.full(cw.size, np.inf)
    min_area[ok] = table[a[ok], b[ok]]
    fill = np.where(np.isfinite(min_area), cw * ch / min_area, 0.0)
    ci = int(np.argmax(fill))  # first maximum -> lowest candidate index
    if fill[ci] <= 0.0:
        return 0.0, -1, -1
    hits = np.flatnonzero((gw >= cw[ci]) & (gh >= ch[ci]) & (area == min_area[ci]))
    gi = int(hits[np.lexsort((y1[hits], x1[hits], y0[hits], x0[hits]))[0]])
    return float(fill[ci]), ci, gi
```

```python
# meshbestfit/layout.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .mesh import best_pair, empty_grids

ALGORITHM_VERSION = 1  # bump on any change to draw order or tie-breaks (spec section 6)

StopReason = Literal["max_elements", "no_fit", "below_threshold", "small_cap", "pool_exhausted"]


class ShortStratumError(ValueError):
    """A large or small stratum has fewer elements than its quota (exit code 4)."""


@dataclass(frozen=True, slots=True, kw_only=True)
class LayoutParams:
    candidates: int = 500
    large_ratio: float = 0.99
    max_elements: int | None = 35  # repo preset: None
    fill_threshold: float = 1e-4  # repo preset: 0.0
    max_small: int | None = None  # repo preset: 5
    shrink: tuple[float, float] = (0.8, 0.95)
    on_short_stratum: Literal["error", "take-all"] = "error"


@dataclass(frozen=True, slots=True)
class Layout:
    boxes: NDArray[np.float64]  # (n, 4) xyxy after shrink
    element_idx: NDArray[np.intp]  # row indices into the pool arrays
    stop_reason: StopReason


def nominal_quotas(p: LayoutParams) -> tuple[int, int]:
    k_large = int(np.floor(p.candidates * p.large_ratio + 1e-9))
    return k_large, p.candidates - k_large


def quotas(small: NDArray[np.bool_], p: LayoutParams) -> tuple[int, int]:
    """(k_large, k_small) to draw; ShortStratumError under `error` when a stratum is short."""
    k_large, k_small = nominal_quotas(p)
    n_small = int(small.sum())
    n_large = small.size - n_small
    if p.on_short_stratum == "take-all":
        k_large, k_small = min(k_large, n_large), min(k_small, n_small)
    elif k_large > n_large or k_small > n_small:
        raise ShortStratumError(
            f"pool has {n_large} large / {n_small} small elements, "
            f"need {k_large} / {k_small} (use --on-short-stratum take-all)"
        )
    if k_large + k_small == 0:
        raise ShortStratumError("pool has no elements to draw")
    return k_large, k_small


def generate_layout(
    w: NDArray[np.float64],
    h: NDArray[np.float64],
    small: NDArray[np.bool_],
    p: LayoutParams,
    rng: np.random.Generator,
) -> Layout:
    """w, h, small: pool arrays already filtered by epsilon. All randomness comes from `rng`."""
    k_large, k_small = quotas(small, p)
    large_pool, small_pool = np.flatnonzero(~small), np.flatnonzero(small)
    cand = np.concatenate(
        [
            rng.choice(large_pool, size=k_large, replace=False),
            rng.choice(small_pool, size=k_small, replace=False),
        ]
    )
    alive = np.ones(cand.size, dtype=bool)

    s = int(rng.integers(cand.size))  # start element
    alive[s] = False
    e = cand[s]
    cx = rng.uniform(w[e] / 2, 1 - w[e] / 2)
    cy = rng.uniform(h[e] / 2, 1 - h[e] / 2)
    boxes = [(cx - w[e] / 2, cy - h[e] / 2, cx + w[e] / 2, cy + h[e] / 2)]
    chosen = [e]
    small_count = int(small[e])
    reason: StopReason

    while True:
        if p.max_elements is not None and len(chosen) >= p.max_elements:
            reason = "max_elements"
            break
        if not alive.any():
            reason = "pool_exhausted"
            break
        live = np.flatnonzero(alive)
        grids = empty_grids(np.asarray(boxes))
        fill, ci, gi = best_pair(w[cand[live]], h[cand[live]], grids)
        if ci < 0:
            reason = "no_fit"
            break
        if fill < p.fill_threshold:
            reason = "below_threshold"
            break
        e = cand[live[ci]]
        if small[e] and p.max_small is not None and small_count + 1 > p.max_small:
            reason = "small_cap"
            break
        small_count += int(small[e])
        alive[live[ci]] = False
        gx = (grids[0][gi] + grids[2][gi]) / 2
        gy = (grids[1][gi] + grids[3][gi]) / 2
        boxes.append((gx - w[e] / 2, gy - h[e] / 2, gx + w[e] / 2, gy + h[e] / 2))
        chosen.append(e)

    b = np.clip(np.asarray(boxes), 0.0, 1.0)
    c = (b[:, :2] + b[:, 2:]) / 2
    half = (b[:, 2:] - b[:, :2]) / 2 * rng.uniform(p.shrink[0], p.shrink[1], size=(len(b), 2))
    return Layout(np.hstack([c - half, c + half]), np.asarray(chosen, dtype=np.intp), reason)
```

The two blocks above are the implementation's `mesh.py` and `layout.py`. They differ from the prototype run described above in four ways: `empty_grids` works in blocks of X-pair rows, `best_pair` returns early on an empty grid set, short strata are handled by `quotas` and `ShortStratumError`, and the stop checks run in the table order (`max_elements` before `pool_exhausted`). The tests in section 9 run against this code.

Pre-shrink boxes can overlap by float rounding at the last digit (a box centered in a grid it exactly fills); after the shrink they cannot, so the no-overlap test applies to final boxes with no tolerance.

## 7. Architecture

One src-layout package, `meshbestfit`. The algorithm (`mesh.py`, `layout.py`) is pure NumPy with no file or CLI imports, so it is tested on arrays; everything that touches disk or processes sits around it.

```text
mesh-bestfit/
├── pyproject.toml
├── README.md
├── uv.lock
├── scripts/
│   └── doclaynet_to_coco.py   # DocLayNet-small -> COCO dataset folder (development dataset, section 9)
├── src/meshbestfit/
│   ├── __init__.py       # __version__
│   ├── cli.py            # Typer app; parses flags, builds Settings, calls one stage
│   ├── config.py         # Pydantic models, presets, TOML loading
│   ├── schema.py         # PoolManifest, LayoutRecord (Pydantic, JSON in/out)
│   ├── mesh.py           # empty_grids, best_pair            (section 6)
│   ├── layout.py         # generate_layout                   (section 6)
│   ├── pool.py           # COCO -> crops + pool.json; manifest -> arrays
│   ├── augment.py        # rare-category variants (needs the [augment] extra)
│   ├── render.py         # layout -> Pillow canvas
│   ├── labels.py         # yolo, yolo-poly, coco writers
│   ├── runner.py         # process pool, seeding, ordered streaming, resume
│   ├── viz.py            # box/crop grid image
│   ├── stats.py          # density, alignment, category counts
│   └── py.typed
└── tests/
    ├── conftest.py, reference.py
    ├── test_mesh.py, test_layout_properties.py, test_labels.py
    ├── test_pipeline.py, test_cli_contract.py, test_dataset_layout.py
    └── test_doclaynet.py      # skipped when datasets/doclaynet is absent
```

**Python 3.10 rules.** Allowed: `X | Y` unions in annotations, `match`, `zip(strict=True)`, `dataclass(slots=True, kw_only=True)`, `from __future__ import annotations`. Not allowed because they need 3.11 or later: `tomllib` (use the `tomli` fallback below), `typing.Self`, `enum.StrEnum`, `ExceptionGroup`, `asyncio.TaskGroup`, `typing.override`, PEP 695 generics. Type-check with `mypy --strict`; ship `py.typed`.

**Config and presets** (Pydantic v2; frozen, `extra="forbid"`):

```python
# meshbestfit/config.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from PIL import ImageColor
from pydantic import BaseModel, ConfigDict, Field, model_validator

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


def _ordered(lo: float, hi: float, name: str) -> None:
    if not 0 < lo <= hi:
        raise ValueError(f"{name} must satisfy 0 < lo <= hi")


class PoolConfig(_Model):
    min_size: float = Field(default=0.01, gt=0, lt=1)
    small_threshold: float = Field(default=0.05, gt=0, le=1)
    exclude_categories: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _order(self) -> PoolConfig:
        if self.min_size >= self.small_threshold:
            raise ValueError("min_size must be below small_threshold")
        return self


class SamplingConfig(_Model):
    candidates: int = Field(default=500, ge=2)
    large_ratio: float = Field(default=0.99, ge=0, le=1)
    on_short_stratum: Literal["error", "take-all"] = "error"


class LayoutConfig(_Model):
    max_elements: int | None = Field(default=35, ge=1)
    fill_threshold: float = Field(default=1e-4, ge=0)
    max_small: int | None = Field(default=None, ge=0)
    shrink: tuple[float, float] = (0.8, 0.95)

    @model_validator(mode="after")
    def _order(self) -> LayoutConfig:
        if not 0 < self.shrink[0] <= self.shrink[1] <= 1:
            raise ValueError("shrink must satisfy 0 < lo <= hi <= 1")
        return self


class RenderConfig(_Model):
    width: tuple[int, int] = (1200, 2000)
    aspect: tuple[float, float] = (0.7, 1.5)  # H / W
    max_height: int = Field(default=3000, ge=1)
    background: str = "white"
    image_format: Literal["jpg", "png"] = "jpg"
    jpeg_quality: int = Field(default=95, ge=1, le=100)
    use_variants: bool = True
    subset: str = Field(default="default", pattern=r"^[A-Za-z0-9][\w.-]*$")  # images/<subset>/
    keep_aspect: bool = False  # fit crops inside their box instead of stretching
    category_offset: int | None = None  # None: contiguous 0-based; else class = id + offset
    label_formats: tuple[Literal["yolo", "yolo-poly", "coco"], ...] = ("yolo", "coco")

    @model_validator(mode="after")
    def _check(self) -> RenderConfig:
        _ordered(*self.width, "width")
        _ordered(*self.aspect, "aspect")
        ImageColor.getrgb(self.background)  # raises ValueError on an unknown color
        if "yolo" in self.label_formats and "yolo-poly" in self.label_formats:
            raise ValueError("label_formats: yolo and yolo-poly both write labels/<page>.txt")
        return self


class AugmentConfig(_Model):
    flip: bool = True  # Paper 1 adds a random flip; the repo does not


class Settings(_Model):
    pool: PoolConfig = PoolConfig()
    sampling: SamplingConfig = SamplingConfig()
    layout: LayoutConfig = LayoutConfig()
    render: RenderConfig = RenderConfig()
    augment: AugmentConfig = AugmentConfig()


PRESETS: dict[str, Settings] = {
    "paper2": Settings(),
    "repo": Settings(
        layout=LayoutConfig(max_elements=None, fill_threshold=0.0, max_small=5), augment=AugmentConfig(flip=False)
    ),
}


def _merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    return {
        k: _merge(a[k], v) if isinstance(v, dict) and isinstance(a.get(k), dict) else v for k, v in {**a, **b}.items()
    }


def load_settings(preset: str, config: Path | None, cli: dict[str, Any]) -> Settings:
    data = PRESETS[preset].model_dump()  # 1. preset
    if config is not None:  # 2. TOML file
        data = _merge(data, tomllib.loads(config.read_text(encoding="utf-8")))
    data = _merge(data, cli)  # 3. explicit CLI flags
    # TOML has no null and a merge cannot unset a preset value: 0 / negative mean "no limit".
    lay = data["layout"]
    if lay.get("max_elements") == 0:
        lay["max_elements"] = None
    if lay.get("max_small") is not None and lay["max_small"] < 0:
        lay["max_small"] = None
    return Settings.model_validate(data)
```

**CLI** (Typer; `Optional[...]` rather than `X | None` in command signatures, because I have not verified Typer's handling of the `|` form on 3.10 \[Unverified\]):

```python
# meshbestfit/cli.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Optional

import typer

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)
layout_app = typer.Typer(no_args_is_help=True)
app.add_typer(layout_app, name="layout")


@app.callback()
def main(
    ctx: typer.Context,
    preset: Annotated[str, typer.Option(help="paper2 or repo")] = "paper2",
    config: Annotated[Optional[Path], typer.Option(exists=True, dir_okay=False)] = None,
) -> None:
    ctx.obj = {"preset": preset, "config": config}


@layout_app.command("generate")
@_guard  # DataError -> 3, ShortStratumError -> 4, TooManyFailures -> 1, ImportError -> 2
def layout_generate(
    ctx: typer.Context,
    pool: Annotated[Path, typer.Option(exists=True)],  # pool.json or its folder
    out: Annotated[Path, typer.Option()],
    count: Annotated[int, typer.Option(min=1)],
    seed: Annotated[Optional[int], typer.Option()] = None,
    workers: Annotated[int, typer.Option(min=0)] = max(1, (os.cpu_count() or 2) - 1),
    resume: Annotated[bool, typer.Option("--resume")] = False,
) -> None:
    from .runner import generate_layouts  # lazy: keeps `--help` fast

    ...
```

The skeleton shows the pattern; the module has every command from section 4. Flags the user did not pass stay `None` and are dropped before `load_settings`, so only explicit flags override the TOML file and the preset. A `ValueError` from Pydantic or the TOML parser becomes `typer.BadParameter` (exit 2). Boolean flags are single-name (`--resume`) except `--keep-aspect/--stretch`, `--use-variants/--no-variants` and `--flip/--no-flip`. `Optional[tuple[int, int]]` (`--width LO HI`) and `Optional[list[...]]` (repeatable flags) work on Python 3.10.21; the `X | None` form was not tried.

**Runner.** Spawn context, pool loaded once per worker, results written by the parent in page-index order through a bounded window of in-flight futures, so output bytes do not depend on `--workers`:

```python
# meshbestfit/runner.py
from __future__ import annotations

import multiprocessing as mp
import signal
import threading
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future, ProcessPoolExecutor
from typing import Any, TypeVar

A = TypeVar("A")
T = TypeVar("T")


def _child_init(initializer: Callable[..., None] | None, initargs: tuple[Any, ...]) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # the parent drains in-flight pages on Ctrl+C
    if initializer:
        initializer(*initargs)


def ordered_map(
    fn: Callable[[A], T],
    items: Iterable[A],
    workers: int,
    *,
    initializer: Callable[..., None] | None = None,
    initargs: tuple[Any, ...] = (),
    window: int = 4,
    stop: threading.Event | None = None,
) -> Iterator[T]:
    """Yield fn(x) in input order; at most workers * window tasks are in flight.

    Once `stop` is set nothing more is submitted and the in-flight tasks are still yielded.
    """
    stop = stop or threading.Event()
    if workers == 0:  # inline, for debugging and tests
        if initializer:
            initializer(*initargs)
        for x in items:
            if stop.is_set():
                return
            yield fn(x)
        return
    ex = ProcessPoolExecutor(
        workers, mp_context=mp.get_context("spawn"), initializer=_child_init, initargs=(initializer, initargs)
    )
    inflight: deque[Future[T]] = deque()
    try:
        for x in items:
            if stop.is_set():
                break
            inflight.append(ex.submit(fn, x))
            if len(inflight) >= workers * window:
                yield inflight.popleft().result()
        while inflight:
            yield inflight.popleft().result()
    finally:
        ex.shutdown(wait=True, cancel_futures=True)
```

`fn` and the initializer are module-level functions, so they pickle under `spawn`. A worker builds its `Generator` from `SeedSequence(seed, spawn_key=(page_index,))`. Workers ignore SIGINT; the parent's handler sets `stop`, so `ordered_map` stops submitting and still yields what is in flight (section 8). A task takes a page index for `layout generate` and `run`, and the raw `layouts.jsonl` line for `render`, so no worker loads the whole file. It returns `(page, layouts line or None, error or None)`: an exception inside one page becomes the error string. `ShortStratumError` is checked in the parent before any worker starts, so it never surfaces inside a page.

**Render semantics** (one page; the stream is `SeedSequence(entropy, spawn_key=(*spawn_key, 1))` from the page's own layout record, so `render` needs no seed and `--seed` only overrides `entropy`; independent of the layout stream. Draw order: W, r, then one crop choice per box):

1. W = integer in \[width\_lo, width\_hi\]; r \~ U(aspect\_lo, aspect\_hi); H = min(max\_height, int(W·r)). White RGB canvas.
2. For each box, pixel rectangle = (x1·W, y1·H, x2·W, y2·H). Pick the crop uniformly from the original plus its variants when `use_variants`; the repo does the same with `random.choice` over the path list (`rendering.py:37`).
3. Resize the crop to the rectangle with no aspect-ratio correction (the repo passes `keep_proportion=False`, `rendering.py:39`) and paste at rounded integer coordinates. Pillow resampling is `LANCZOS`; MuPDF's resampling is not documented in the repo \[Unverified\], so pixels will not match the repo byte for byte. With `keep_aspect`, the crop is instead scaled by min(box width / crop width, box height / crop height) and centered in the rectangle, and step 4 uses the pasted rectangle as the label box.
4. Labels use the unrounded float box in pixels, normalized for YOLO formats.

## 8. Determinism, performance, logging, errors

Reproducibility holds per environment: the same seed, config, pool, and locked dependency versions give the same output for any `--workers` value. Across environments it is not assumed. \[Unverified\] NumPy may change `Generator` streams between releases, and Python 3.10 resolves to NumPy 2.2.6 while newer Pythons resolve to newer NumPy. `run.json` therefore records the NumPy and Pillow versions, and the repo ships a `uv.lock`.

**Determinism rules**

- All layout math is float64. No iteration over `set` or `dict` order feeds a decision.
- Page *i* uses `SeedSequence(seed, spawn_key=(i,))` for layout and `(i, 1)` for render. Nothing reads the clock, a global RNG, or `os.getpid()`.
- The parent writes results in page-index order (`ordered_map`, section 7).

**Performance**

- Task = one page. Workers load the pool arrays once in the initializer and open crops lazily.
- Mesh memory grows with the square of the tick pairs: at 34 placed boxes, C(70,2)² = 5,832,225 entries, about 23 MB per int32 array (arithmetic), and I measured 154 MB peak RSS in the prototype. `empty_grids` computes `covered` in blocks of 512 X-pair rows (`mesh._BLOCK`), so memory stays bounded when `max_elements` is raised or unset (the `repo` preset). A test checks that blocking does not change the grid set; the memory saving itself was not measured \[Unverified\].
- &#91;Inference\] Render, not layout, should dominate per-page time for typical pages (decode, resize, JPEG encode of up to 35 crops at 1200–2000 px width). Benchmark both with `pytest-benchmark` before tuning.
- Measured on DocLayNet-small train (pool of 8,427 elements, `paper2` defaults, 4 workers, a machine with 20 logical CPUs): `run --count 100` took 1.7 s wall including worker start-up and `layout generate --count 100` took 0.8 s. Those pages averaged 9.7 boxes, far below N = 35, so the timings do not carry over to dense pages \[Inference\]. No `pytest-benchmark` test exists yet.

**Logging.** Standard `logging` with a Rich handler on stderr; `-v` is DEBUG, `-q` is WARNING. `--log-format json` writes one JSON object per line (formatted with `orjson`). Stdout carries only machine-readable output (`stats --json`, `--dry-run`). Progress bars go to stderr and turn off when stderr is not a TTY.

**Errors**

- Page failures (missing or corrupt crop, unreadable image): log page index and element id, skip the page, count it. `--max-failures N` (default 0) aborts with exit code 1 once the count exceeds N.
- Load-time validation (exit 3): duplicate annotation ids, `w` or `h` outside (0, 1\], a manifest path that does not exist when `--check-files` is set, an unknown category id.
- Output safety: images and labels are written to `*.tmp` and renamed. `layouts.jsonl` is appended and flushed per page, after that page's image and labels are on disk; `--resume` truncates a trailing partial line, then skips page indices already present (for `render`: pages whose image file exists).
- SIGINT: stop submitting, drain in-flight pages, flush, exit 130. Workers ignore SIGINT; the first Ctrl+C sets a stop flag in the parent and a second one aborts. `run.json` is written when a run starts (seed and zero counts, so `--resume` can read the seed back) and again when it finishes; `layout generate` writes it next to its `--out` file. `classes.txt` and the COCO file are written only after a complete run.
- `--dry-run`: validate config and pool (including the sampling quotas, exit 4), print `{"settings": …, "outputs": […]}` as JSON on stdout, exit 0 without writing.

## 9. Testing and validation

The algorithm is tested against a slow, literal port of the repo's lattice-and-loop logic, plus property tests on invariants. The prototype of sections 6 and 7 was run on Python 3.12.3 only. The implementation's 69 tests pass on Python 3.10.21 (NumPy 2.2.6, Pillow 12.3.0), on 3.10 at the dependency floors, and on the default uv Python 3.14.10; `ruff check` and `mypy --strict` are clean. No CI is set up.

| Layer | What it checks | Tool |
| --- | --- | --- |
| Reference equivalence | `empty_grids` returns the same grid set as the brute-force port; `best_pair` returns the same fill and candidate | pytest, `tests/reference.py` |
| Layout invariants | boxes inside \[0, 1\]; no two final boxes overlap; count ≤ N; `stop_reason` is valid; small count ≤ `max_small`; same seed gives identical boxes | Hypothesis |
| Determinism | `layouts.jsonl`, COCO, label and image bytes identical for `--workers 0` and `--workers 2`; `render` from a saved layouts file equals `run`; `--resume` after a truncated file (or after SIGINT) equals an uninterrupted run, with the seed read back from `run.json` | pytest, tmp dirs |
| Pipeline | 3-image COCO fixture: `pool build`, `run --count 5`, then every label box lies inside its image and counts match the layout | pytest, Typer `CliRunner` |
| CLI contract | exit codes 0, 1, 2, 3, 4, 130; `--dry-run` writes nothing; unknown TOML key exits 2; CLI beats TOML beats preset; `--check-files`; `stats` and `viz` run | `CliRunner` |
| Labels | class ids contiguous or offset, `classes.txt` gaps, yolo and yolo-poly text, `--keep-aspect` boxes stay inside layout boxes | pytest |
| Dataset layout | the file tree equals the CVAT COCO 1.0 layout (`images/<subset>/`, `annotations/instances_<subset>.json`, `labels/<subset>/`); COCO keys and fields per cocodataset.org, empty `segmentation`, unique ids; an unsafe `--subset` exits 2; `--dry-run` lists the same paths | pytest |
| Development dataset | DocLayNet-small `val` converts to COCO with every box inside its image; `pool build` runs on it | pytest, skipped when absent |
| Dependency floors | tests pass on the lowest allowed versions | `uv pip install --resolution lowest-direct` job |
| Python matrix | 3.10, 3.12 and the newest stable Python | CI (not set up), `uv run --python <ver> pytest` |

**Development dataset (DocLayNet-small).** `datasets/doclaynet/{train,val,test}` hold per-page `annotations/*.json` and `images/*.png` (1025 × 1025), 691, 64 and 49 pages; the `pdfs/` folders are unused. Each JSON has `metadata` and `form[]` rows; rows sharing `id_box` are the lines of one layout block, and `box` = [x, y, w, h] is already in the PNG's pixel space, not PDF points, origin top-left. `val` boxes overshoot the image by up to 1.31 px. `scripts/doclaynet_to_coco.py <split> --out DIR` writes a COCO dataset folder (`DIR/annotations/instances_<split>.json`, `DIR/images/<split>` as a symlink to the PNGs; `pool build --dataset DIR` reads it) with 11 categories in DocLayNet's alphabetical order and `id_box` as the annotation id (unique within each split). `train` has 9,017 blocks, 8,427 of which pass `min_size = 0.01`, enough for the default K = 500 draw (495 large, 5 small). `val` and `test` were not used to build a pool beyond the `val` test.

**Property test**

```python
# tests/test_layout_properties.py
import itertools

import numpy as np
from hypothesis import HealthCheck, given, settings, strategies as st

from meshbestfit.layout import LayoutParams, generate_layout

rng0 = np.random.default_rng(0)
W = np.concatenate([rng0.uniform(0.05, 0.9, 3500), rng0.uniform(0.01, 0.05, 500)])
H = np.concatenate([rng0.uniform(0.05, 0.5, 3500), rng0.uniform(0.01, 0.05, 500)])
SMALL = (W < 0.05) | (H < 0.05)


@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 2**32 - 1), page=st.integers(0, 10_000))
def test_invariants(seed: int, page: int) -> None:
    p = LayoutParams(max_elements=35, fill_threshold=1e-4)
    ss = np.random.SeedSequence(seed, spawn_key=(page,))
    a = generate_layout(W, H, SMALL, p, np.random.default_rng(ss))
    b = generate_layout(W, H, SMALL, p, np.random.default_rng(ss))
    assert np.array_equal(a.boxes, b.boxes)  # determinism
    bx = a.boxes
    assert 1 <= len(bx) <= 35 and (bx >= 0).all() and (bx <= 1).all()
    for u, v in itertools.combinations(range(len(bx)), 2):  # no overlap after shrink
        ox = min(bx[u, 2], bx[v, 2]) - max(bx[u, 0], bx[v, 0])
        oy = min(bx[u, 3], bx[v, 3]) - max(bx[u, 1], bx[v, 1])
        assert not (ox > 0 and oy > 0)
```

**Reference port** (kept in `tests/reference.py`; deliberately unoptimized, float64, strict zero-overlap test):

```python
import itertools


def ref_grids(boxes):
    xs = sorted({0.0, 1.0, *boxes[:, 0], *boxes[:, 2]})
    ys = sorted({0.0, 1.0, *boxes[:, 1], *boxes[:, 3]})
    out = set()
    for xa, xb in itertools.combinations(xs, 2):
        for ya, yb in itertools.combinations(ys, 2):
            if all(not (min(xb, b[2]) - max(xa, b[0]) > 0 and min(yb, b[3]) - max(ya, b[1]) > 0) for b in boxes):
                out.add((xa, ya, xb, yb))
    return out


def ref_best(cw, ch, grids):
    best = (0.0, -1)
    for ei, (w, h) in enumerate(zip(cw, ch)):
        for g in grids:
            gw, gh = g[2] - g[0], g[3] - g[1]
            if w > gw or h > gh:
                continue
            f = (w * h) / (gw * gh)
            if f > best[0]:
                best = (f, ei)
    return best
```

**Quality sanity check, not a gate.** `stats` reports density (sum of element areas ÷ page area, Paper 1 Eq. 7) and alignment (Paper 1 Eq. 5–6, where the sum's normalization is unclear in the text I read \[Unverified\]). The papers report Align 0.0009 and Density 0.645 for Mesh-candidate BestFit against 0.0171 and 0.259 for random placement (Paper 1 Table 9; Paper 2 repeats the method's pair). Those numbers depend on the paper's pool, so a regression test should compare against random placement on the same pool, not against 0.645. That baseline test is not implemented. The implemented `stats` alignment is the mean over boxes of −log(1 − d), with d the smallest gap on any of the six alignment lines (left, center, right, top, middle, bottom) to any other box \[Unverified: normalization\]. Measured on 100 pages from the DocLayNet-small train pool: density 0.652 and alignment 0.0014, against the papers' 0.645 and 0.0009 on their own pool \[Inference: not comparable\].

## 10. Packaging and dependency pins

Managed with uv, built with Hatchling (1.33.0 requires Python ≥3.10 per PyPI). Floors in `dependencies` are the section 3 versions; the only upper bound is on OpenCV, because its 5.x API against 4.x is unverified. `uv.lock` pins exact versions. \[Inference\] A universal lock should hold separate NumPy and Pillow releases for 3.10 and for 3.12 and newer.

```toml
# pyproject.toml
[build-system]
requires = ["hatchling>=1.33"]
build-backend = "hatchling.build"

[project]
name = "mesh-bestfit"
version = "0.1.0"
description = "Mesh-candidate BestFit synthetic document-layout generator"
readme = "README.md"
requires-python = ">=3.10"
license = "BSD-3-Clause"
dependencies = [
  "numpy>=2.2.6",
  "pillow>=12.3",
  "typer>=0.27",
  "pydantic>=2.14",
  "orjson>=3.13",
  "rich>=15",
  "tomli>=2.5; python_version < '3.11'",
]

[project.optional-dependencies]
augment = ["opencv-python-headless>=5.0,<6"]

[project.scripts]
mesh-bestfit = "meshbestfit.cli:app"

[dependency-groups]
dev = [
  "pytest>=9.2",
  "hypothesis>=6.168",
  "pytest-benchmark>=5.3",
  "ruff>=0.17",
  "mypy>=2.4",
]

[tool.hatch.build.targets.wheel]
packages = ["src/meshbestfit"]

[tool.ruff]
target-version = "py310"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM", "NPY", "PERF"]

[tool.ruff.lint.per-file-ignores]
"src/meshbestfit/cli.py" = ["UP007", "UP045"]  # spec: Optional[...] in Typer signatures
"tests/*" = ["E501", "SIM300", "B905"]

[tool.mypy]
strict = true
python_version = "3.10"

[[tool.mypy.overrides]]
module = ["tomllib", "tomli"]  # not importable on every Python the checker targets
ignore_missing_imports = true

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra"
```

```bash
uv sync --extra augment                 # .venv + lockfile
uv run pytest
uv run ruff check . && uv run mypy src
uv run --python 3.10 pytest             # the floor Python
uv pip install --resolution lowest-direct -e ".[augment]" "pytest>=9.2" "hypothesis>=6.168"   # floor-version job
uv build                                # sdist + wheel
```

I confirmed that `uv pip install --resolution` accepts `lowest-direct`, and that `uv sync` has `--extra`, `--group`, `--locked`, `--frozen` and `--resolution`, on the uv release available in my sandbox (0.11.7). On uv 0.13.6 the implementation's `pyproject.toml` resolved with `uv sync --extra augment` (NumPy 2.5.4 and Pillow 13.0.0 on Python 3.14.10) and built with `uv build`. The floor command needs floors on the test tools too: with `pytest hypothesis` unconstrained, `lowest-direct` picked ancient releases whose build failed. With the floors above it installed numpy 2.2.6, pillow 12.3.0, typer 0.27.0, pydantic 2.14.0, orjson 3.13.0, rich 15.0.0 and opencv-python-headless 5.0.0.93 on Python 3.10, and all 69 tests passed. The `tomllib` and `tomli` imports need a mypy override because neither exists on every Python the checker targets.

## 11. Open questions and unverified items

**Decisions for you**

1. Pool source: M6Doc test (repo README, Paper 1) or DocBank train (Paper 2)? The spec accepts any COCO file; `--exclude-category` covers Paper 2's removal of Date and Section. The choice changes four things. Classes: pool categories become classes.txt, so M6Doc-test gives 74 types from about 2,800 pages (Paper 1, Section 3.1), while DocBank has 13 types and Paper 2 drops Date and Section to reach 11 (Paper 2, Table 1 and App. E.1). Leakage: Paper 2 states it takes elements only from DocBank's training split, not from any downstream benchmark, while Paper 1 uses M6Doc test. Pool size: each page draws 495 large and 5 small elements, so a pool with fewer exits with code 4. Balance: layouts sample elements uniformly, so a category's share in the output follows its share in the pool; in the repo, augmentation only adds appearance variants chosen at render time (map\_dict.py, rendering.py:37), not extra sampling entries. Development currently uses DocLayNet-small (section 9).
2. Label default: `yolo` plus `coco`. Settled: your trainer (Ultralytics) reads both. The repo writes 8-value polygon lines; yolo-poly stays available for parity.
3. Aspect distortion: add a `--keep-aspect` mode. Implemented as fit-inside-and-center, with labels on the pasted rectangle (section 4 flag notes). \[Inference\] That is the implementation's reading of the request; confirm it is the mode you meant.
4. Standalone package (mesh-bestfit takes COCO as inputs, docsynth can grow big due to other new features, mesh-bestfit can grow big with more generation algorithms)
5. License: BSD-3-Clause. `pyproject.toml` declares it; no LICENSE file exists yet because it needs the copyright holder's name.
6. Name: `mesh-bestfit` (package `meshbestfit`)

**Unverified items**

| Item | Where it matters | How to settle |
| --- | --- | --- |
| Start-element position in Paper 2 | section 6, step 2 | read the authors' released code or ask them; repo rule used |
| Whether Paper 2 keeps the small-element cap | `paper2` preset | same; cap is inert at K = 500, ρ = 0.99 anyway |
| Paper 1 `fr_thr` value | `repo` preset uses 0 | not stated in the text I read |
| Flip step in augmentation (Paper 1 yes, repo no) | `pool augment` default | choose `--flip` or `--no-flip` per preset |
| Align metric normalization | `stats` alignment value (implemented as mean −log(1 − d)) | check LayoutGAN++ definition |
| MuPDF vs Pillow resampling | pixel-level parity with the repo | render one page both ways and diff |
| Typer on 3.10 with `X \| None` | `cli.py` signatures | `Optional[...]` verified on 3.10.21 by the CLI tests; the `X \| None` form was not tried |
| NumPy `Generator` stream stability across releases | cross-environment reproducibility | read NumPy's stream-compatibility policy; record versions meanwhile |
| OpenCV 5.x API against 4.x; albumentations 2.x argument names | `[augment]` extra | CI on the `[augment]` extra |
| Augmentation strengths (brightness, crop, Sobel, elastic, noise) | `pool augment` output | compare with the repo's `augmentation.py` parameters |
| CI matrix (3.10, 3.12, newest) | G2 | not set up; local runs on 3.10.21 and 3.14.10 pass |
| Windows and macOS (`spawn`, paths) | section 3, Platforms | not run; Linux only |
| COCO `file_name` rule (bare name or path) | `annotations/instances_<subset>.json` | neither the CVAT page nor the COCO page states it; the two importers resolved bare names in the 20-page check |
| Uploading the dataset zip to a CVAT instance | section 5 | not tried; only pycocotools and Datumaro loaded the output |
| Speed on dense pages and on M6Doc or DocBank pools | section 8 | DocLayNet-small timings exist; add `pytest-benchmark` tests |
| Local DocLayNet-small equals the Hugging Face `pierreguillou/DocLayNet-small` dataset | section 9 | the dataset page was not opened; compare file lists and counts |

**Sources**

- Uploaded repo snapshot `opendatalab_DocLayout-YOLO_main_mesh-candidate_bestfit.zip` (README, `bestfit_generator.py`, `rendering.py`, `augmentation.py`, `map_dict.py`, `combine_layouts.py`, `utils/`, `visualize.ipynb`).
- Paper 1: *DocLayout-YOLO: Enhancing Document Layout Analysis through Diverse Synthetic Data and Global-to-Local Adaptive Perception* (project file, 15 pages): Section 3, Appendix A.2–A.3.
- Paper 2: *DocLayout-YOLO: Axis-Attentive Document Layout Detector with Mesh-Guided Synthetic Pretraining* (project file, 35 pages): Section 3, Appendix E.
- Dependency versions: PyPI JSON API, queried 2026-10-31, e.g. [numpy](https://pypi.org/pypi/numpy/json), [pillow](https://pypi.org/pypi/pillow/json), [pymupdf](https://pypi.org/pypi/pymupdf/json); same endpoint for each package in section 3.
- Development dataset: `datasets/doclaynet/` (DocLayNet-small layout: `annotations/`, `images/`, `pdfs/` per split); reference given with the task: https://huggingface.co/datasets/pierreguillou/DocLayNet-small (not opened).
- Dataset format: CVAT, https://docs.cvat.ai/docs/dataset_management/formats/format-coco/ (tree, `instances_<subset>.json`, boxes when `segmentation` is empty); COCO data format, https://cocodataset.org/#format-data (that page renders no content when fetched, so the field list was read from its source file `dataset/format-data.htm` in the `cocodataset/cocodataset.github.io` repository); Datumaro COCO page, https://open-edge-platform.github.io/datumaro/latest/docs/data-formats/formats/coco.html (`images/<subset>/`, `<task>_<subset>.json`, category id 0).
- Prototype checks (equivalence, determinism, timing) were run in this session on Python 3.12.3 with a synthetic pool.
