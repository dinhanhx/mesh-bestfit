from __future__ import annotations

import hashlib
import logging
import multiprocessing as mp
import os
import platform
import signal
import threading
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future, ProcessPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, TypeVar

import numpy as np
import orjson
import PIL

from . import __version__
from .config import Settings
from .labels import (
    atomic_write,
    class_index,
    classes_txt,
    image_name,
    label_name,
    write_coco,
    yolo_text,
)
from .layout import (
    ALGORITHM_VERSION,
    LayoutParams,
    ShortStratumError,
    generate_layout,
    nominal_quotas,
    quotas,
)
from .pool import PoolArrays, pool_arrays
from .render import plan_page, render_page
from .schema import LayoutRecord, PoolManifest, SeedInfo, load_manifest

log = logging.getLogger(__name__)

A = TypeVar("A")
T = TypeVar("T")
Result = tuple[int, bytes | None, str | None]  # page, layouts.jsonl line, error message


class TooManyFailures(Exception):
    """Failed pages exceeded --max-failures (exit code 1)."""


def _no_progress() -> None:
    return None


# --- ordered process map -------------------------------------------------------------------


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


# --- worker side ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class WorkerSpec:
    """Everything a worker needs; plain data so it pickles under `spawn`."""

    pool_path: str
    settings_json: str
    preset: str
    seed: int = 0  # layout stream: SeedSequence(seed, spawn_key=(page,))
    render_seed: int | None = None  # None: render with each layout record's own seed
    out_dir: str = ""


class _State:
    def __init__(self, spec: WorkerSpec) -> None:
        self.spec = spec
        self.settings = Settings.model_validate_json(spec.settings_json)
        self.manifest = load_manifest(Path(spec.pool_path))
        self.root = Path(spec.pool_path).parent
        self.elements = {e.id: e for e in self.manifest.elements}
        self.arrays = pool_arrays(self.manifest, self.settings.pool)
        self.params = layout_params(self.settings)
        self.classes = class_index(self.manifest.categories, self.settings.render.category_offset)
        self.out = Path(spec.out_dir)


_STATE: _State | None = None


def layout_params(s: Settings) -> LayoutParams:
    return LayoutParams(
        candidates=s.sampling.candidates,
        large_ratio=s.sampling.large_ratio,
        on_short_stratum=s.sampling.on_short_stratum,
        max_elements=s.layout.max_elements,
        fill_threshold=s.layout.fill_threshold,
        max_small=s.layout.max_small,
        shrink=s.layout.shrink,
    )


def init_worker(spec: WorkerSpec) -> None:
    global _STATE  # noqa: PLW0603  # per-process worker state
    _STATE = _State(spec)


def _state() -> _State:
    assert _STATE is not None, "init_worker was not called"
    return _STATE


def _layout(page: int) -> LayoutRecord:
    st = _state()
    ss = np.random.SeedSequence(st.spec.seed, spawn_key=(page,))
    lay = generate_layout(st.arrays.w, st.arrays.h, st.arrays.small, st.params, np.random.default_rng(ss))
    idx = lay.element_idx
    return LayoutRecord(
        page=page,
        seed=SeedInfo(entropy=st.spec.seed, spawn_key=(page,)),
        preset=st.spec.preset,
        boxes=tuple(map(tuple, lay.boxes.tolist())),
        category_ids=tuple(int(c) for c in st.arrays.category_ids[idx]),
        element_ids=tuple(st.arrays.ids[i] for i in idx),
        stop_reason=lay.stop_reason,
    )


def _write_page(rec: LayoutRecord) -> None:
    st = _state()
    cfg = st.settings.render
    plan = plan_page(rec, st.elements, cfg, st.spec.render_seed)
    img, rects = render_page(rec, plan, st.root, cfg)
    name = st.out / image_name(rec.page, cfg)
    tmp = name.with_name(name.name + ".tmp")
    img.save(tmp, format="JPEG" if cfg.image_format == "jpg" else "PNG", quality=cfg.jpeg_quality)
    os.replace(tmp, name)
    for fmt in ("yolo", "yolo-poly"):
        if fmt in cfg.label_formats:
            text = yolo_text(rects, plan.size, [st.classes[c] for c in rec.category_ids], poly=fmt == "yolo-poly")
            atomic_write(st.out / label_name(rec.page, cfg), text.encode())


def _line(rec: LayoutRecord) -> bytes:
    return rec.model_dump_json().encode()


def layout_task(page: int) -> Result:
    try:
        return page, _line(_layout(page)), None
    except ShortStratumError:
        raise
    except Exception as e:  # noqa: BLE001 - one bad page must not kill the run
        return page, None, f"{type(e).__name__}: {e}"


def render_task(line: bytes) -> Result:
    rec = LayoutRecord.model_validate_json(line)
    try:
        _write_page(rec)
        return rec.page, None, None
    except Exception as e:  # noqa: BLE001
        return rec.page, None, f"{type(e).__name__}: {e}"


def run_task(page: int) -> Result:
    try:
        rec = _layout(page)
        _write_page(rec)
        return page, _line(rec), None
    except ShortStratumError:
        raise
    except Exception as e:  # noqa: BLE001
        return page, None, f"{type(e).__name__}: {e}"


# --- parent side ---------------------------------------------------------------------------


@dataclass(slots=True)
class Counts:
    done: int = 0
    failures: int = 0
    skipped: int = 0  # already finished (--resume)


def check_pool(pool_path: Path, settings: Settings, *, check_files: bool = False) -> PoolArrays:
    """Load and validate the pool; raise ShortStratumError now rather than inside a worker."""
    m = load_manifest(pool_path, check_files=check_files)
    arr = pool_arrays(m, settings.pool)
    p = layout_params(settings)
    got = quotas(arr.small, p)
    if got != nominal_quotas(p):
        log.warning(
            "short stratum: drawing %d large + %d small candidates instead of %d + %d", *got, *nominal_quotas(p)
        )
    return arr


def iter_lines(path: Path) -> Iterator[bytes]:
    with path.open("rb") as f:
        for line in f:
            if line.strip():
                yield line


def iter_records(path: Path) -> Iterator[LayoutRecord]:
    for line in iter_lines(path):
        yield LayoutRecord.model_validate_json(line)


def finished_pages(path: Path) -> set[int]:
    """Pages already in a layouts file. Drops a trailing partial line first."""
    if not path.exists():
        return set()
    data = path.read_bytes()
    keep = data[: data.rfind(b"\n") + 1]
    if len(keep) != len(data):
        path.write_bytes(keep)
    return {int(orjson.loads(line)["page"]) for line in keep.splitlines() if line.strip()}


def _drive(
    fn: Callable[[A], Result],
    items: Iterable[A],
    spec: WorkerSpec,
    workers: int,
    *,
    sink: Path | None,
    append: bool,
    stop: threading.Event,
    progress: Callable[[], None],
    max_failures: int,
) -> Counts:
    c = Counts()
    out: IO[bytes] | Any = sink.open("ab" if append else "wb") if sink else nullcontext()
    with out as f:
        for page, line, err in ordered_map(fn, items, workers, initializer=init_worker, initargs=(spec,), stop=stop):
            if err is not None:
                c.failures += 1
                log.error("page %d skipped: %s", page, err)
                if c.failures > max_failures:
                    raise TooManyFailures(f"{c.failures} failed pages (--max-failures {max_failures})")
            else:
                c.done += 1
                if line is not None and f is not None:
                    f.write(line + b"\n")
                    f.flush()
            progress()
    return c


def generate_layouts(
    spec: WorkerSpec,
    count: int,
    out_file: Path,
    *,
    workers: int,
    resume: bool,
    stop: threading.Event,
    progress: Callable[[], None] = _no_progress,
    max_failures: int = 0,
) -> Counts:
    done = finished_pages(out_file) if resume else set()
    todo = (i for i in range(count) if i not in done)
    c = _drive(
        layout_task,
        todo,
        spec,
        workers,
        sink=out_file,
        append=resume,
        stop=stop,
        progress=progress,
        max_failures=max_failures,
    )
    c.skipped = len(done)
    return c


def render_layouts(
    spec: WorkerSpec,
    layouts: Path,
    *,
    workers: int,
    resume: bool,
    stop: threading.Event,
    progress: Callable[[], None] = _no_progress,
    max_failures: int = 0,
) -> Counts:
    out = Path(spec.out_dir)
    cfg = Settings.model_validate_json(spec.settings_json).render
    skipped = 0

    def todo() -> Iterator[bytes]:
        nonlocal skipped
        for line in iter_lines(layouts):
            page = int(orjson.loads(line)["page"])
            if resume and (out / image_name(page, cfg)).is_file():
                skipped += 1
            else:
                yield line

    c = _drive(
        render_task,
        todo(),
        spec,
        workers,
        sink=None,
        append=False,
        stop=stop,
        progress=progress,
        max_failures=max_failures,
    )
    c.skipped = skipped
    return c


def run_pages(
    spec: WorkerSpec,
    count: int,
    *,
    workers: int,
    resume: bool,
    stop: threading.Event,
    progress: Callable[[], None] = _no_progress,
    max_failures: int = 0,
) -> Counts:
    layouts = Path(spec.out_dir) / "layouts.jsonl"
    done = finished_pages(layouts) if resume else set()
    todo = (i for i in range(count) if i not in done)
    c = _drive(
        run_task,
        todo,
        spec,
        workers,
        sink=layouts,
        append=resume,
        stop=stop,
        progress=progress,
        max_failures=max_failures,
    )
    c.skipped = len(done)
    return c


def finalize_dataset(spec: WorkerSpec, layouts: Path, manifest: PoolManifest) -> int:
    """classes.txt and, if requested, annotations/instances_<subset>.json. Returns COCO pages."""
    st = Settings.model_validate_json(spec.settings_json)
    out = Path(spec.out_dir)
    cfg = st.render
    classes = classes_txt(manifest.categories, cfg.category_offset)
    atomic_write(out / "classes.txt", classes.encode())
    if "coco" not in cfg.label_formats:
        return 0
    return write_coco(
        out,
        iter_records(layouts),
        manifest.categories,
        {e.id: e for e in manifest.elements},
        Path(spec.pool_path).parent,
        cfg,
        spec.render_seed,
    )


def write_run_json(
    path: Path, *, settings: Settings, preset: str, seed: int | None, pool_path: Path, counts: Counts, requested: int
) -> None:
    data = {
        "tool": "mesh-bestfit",
        "version": __version__,
        "algorithm_version": ALGORITHM_VERSION,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pillow": PIL.__version__,
        "preset": preset,
        "seed": seed,
        "settings": settings.model_dump(mode="json"),
        "pool": str(pool_path),
        "pool_sha256": hashlib.sha256(pool_path.read_bytes()).hexdigest(),
        "counts": {
            "requested": requested,
            "done": counts.done,
            "failed": counts.failures,
            "skipped_resume": counts.skipped,
        },
    }
    atomic_write(path, orjson.dumps(data, option=orjson.OPT_INDENT_2))
