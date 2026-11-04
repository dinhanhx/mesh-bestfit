from __future__ import annotations

import functools
import logging
import os
import secrets
import signal
import sys
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Optional, TypeVar

import orjson
import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress

from . import __version__
from .config import PRESETS, Settings, load_settings
from .labels import coco_name
from .layout import ShortStratumError
from .schema import DataError, load_manifest

log = logging.getLogger("meshbestfit")

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)
pool_app = typer.Typer(no_args_is_help=True, help="Element pool.")
layout_app = typer.Typer(no_args_is_help=True, help="Layout generation.")
app.add_typer(pool_app, name="pool")
app.add_typer(layout_app, name="layout")

F = TypeVar("F", bound=Callable[..., Any])


class LogFormat(str, Enum):
    text = "text"
    json = "json"


class LabelFormat(str, Enum):
    yolo = "yolo"
    yolo_poly = "yolo-poly"
    coco = "coco"


Seed = Annotated[Optional[int], typer.Option(help="Default: random, printed and saved to run.json.")]
Workers = Annotated[int, typer.Option(min=0, help="0 = run inline.")]
Resume = Annotated[bool, typer.Option("--resume", help="Skip finished page indices.")]
Overwrite = Annotated[bool, typer.Option("--overwrite", help="Replace existing output.")]
DryRun = Annotated[bool, typer.Option("--dry-run", help="Validate, print the plan, write nothing.")]
CheckFiles = Annotated[bool, typer.Option("--check-files", help="Require every pool file to exist.")]
MaxFailures = Annotated[int, typer.Option(min=0, help="Abort once more pages than this fail.")]
DEFAULT_WORKERS = max(1, (os.cpu_count() or 2) - 1)


# --- plumbing ------------------------------------------------------------------------------


def _die(code: int, e: object) -> typer.Exit:
    typer.echo(f"error: {e}", err=True)
    return typer.Exit(code)


def _guard(fn: F) -> F:
    """Map domain errors to the documented exit codes."""
    from .runner import TooManyFailures

    @functools.wraps(fn)
    def wrapper(*a: Any, **kw: Any) -> Any:
        try:
            return fn(*a, **kw)
        except DataError as e:
            raise _die(3, e) from e
        except ShortStratumError as e:
            raise _die(4, e) from e
        except TooManyFailures as e:
            raise _die(1, e) from e
        except ImportError as e:
            raise _die(2, e) from e

    return wrapper  # type: ignore[return-value]


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return orjson.dumps({"level": record.levelname, "logger": record.name, "msg": record.getMessage()}).decode()


def _setup_logging(verbose: bool, quiet: bool, fmt: LogFormat) -> None:
    handler: logging.Handler
    if fmt is LogFormat.json:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(_JsonFormatter())
    else:
        handler = RichHandler(console=Console(stderr=True), show_path=False)
    root = logging.getLogger("meshbestfit")
    root.handlers[:] = [handler]
    root.setLevel(logging.DEBUG if verbose else logging.WARNING if quiet else logging.INFO)


def _version(value: bool) -> None:
    if value:
        typer.echo(f"mesh-bestfit {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    ctx: typer.Context,
    version: Annotated[bool, typer.Option("--version", callback=_version, is_eager=True)] = False,
    verbose: Annotated[bool, typer.Option("-v", "--verbose")] = False,
    quiet: Annotated[bool, typer.Option("-q", "--quiet")] = False,
    log_format: Annotated[LogFormat, typer.Option(help="Log line format.")] = LogFormat.text,
    preset: Annotated[str, typer.Option(help="paper2 or repo")] = "paper2",
    config: Annotated[Optional[Path], typer.Option(exists=True, dir_okay=False, help="TOML settings file.")] = None,
) -> None:
    if preset not in PRESETS:
        raise typer.BadParameter(f"{preset!r}: choose from {', '.join(PRESETS)}", param_hint="--preset")
    _setup_logging(verbose, quiet, log_format)
    ctx.obj = {"preset": preset, "config": config}


def _settings(ctx: typer.Context, **sections: dict[str, Any]) -> Settings:
    """Preset < TOML < CLI. Only flags the user passed (not None) override."""
    cli = {s: {k: v for k, v in d.items() if v is not None} for s, d in sections.items()}
    try:
        return load_settings(ctx.obj["preset"], ctx.obj["config"], cli)
    except ValueError as e:  # pydantic ValidationError and TOMLDecodeError are both ValueErrors
        raise typer.BadParameter(str(e)) from e


def _pool_file(p: Path) -> Path:
    return p / "pool.json" if p.is_dir() else p


def _occupied(p: Path) -> bool:
    return any(p.iterdir()) if p.is_dir() else p.exists()


def _check_output(paths: list[Path], resume: bool, overwrite: bool) -> None:
    if resume and overwrite:
        raise typer.BadParameter("--resume and --overwrite are mutually exclusive")
    if not (resume or overwrite):
        for p in paths:
            if _occupied(p):
                raise typer.BadParameter(f"{p} already exists (use --resume or --overwrite)")


def _resolve_seed(seed: int | None, resume: bool, run_json: Path) -> int:
    if seed is not None:
        return seed
    if resume:
        try:
            return int(orjson.loads(run_json.read_bytes())["seed"])
        except (OSError, KeyError, TypeError, ValueError) as e:
            raise typer.BadParameter(f"--resume without --seed needs a seed in {run_json}") from e
    return secrets.randbits(63)


def _run_json(
    path: Path, ctx: typer.Context, s: Settings, seed: int | None, pool: Path, counts: Any, requested: int
) -> None:
    from .runner import write_run_json

    write_run_json(
        path, settings=s, preset=ctx.obj["preset"], seed=seed, pool_path=pool, counts=counts, requested=requested
    )


def _dry_run(settings: Settings, outputs: list[Path]) -> None:
    typer.echo(
        orjson.dumps(
            {"settings": settings.model_dump(mode="json"), "outputs": [str(p) for p in outputs]},
            option=orjson.OPT_INDENT_2,
        ).decode()
    )


@contextmanager
def _interruptible() -> Iterator[threading.Event]:
    """First Ctrl+C: stop submitting and drain in-flight pages. Second: raise."""
    stop = threading.Event()

    def handler(signum: int, frame: Any) -> None:
        if stop.is_set():
            raise KeyboardInterrupt
        log.warning("interrupted: finishing in-flight pages (Ctrl+C again to abort)")
        stop.set()

    old = signal.signal(signal.SIGINT, handler)
    try:
        yield stop
    finally:
        signal.signal(signal.SIGINT, old)


@contextmanager
def _bar(total: int | None, desc: str) -> Iterator[Callable[[], None]]:
    console = Console(stderr=True)
    if not console.is_terminal:
        yield lambda: None
        return
    with Progress(console=console) as p:
        task = p.add_task(desc, total=total)
        yield lambda: p.advance(task)


def _layout_overrides(
    candidates: int | None,
    large_ratio: float | None,
    on_short: str | None,
    max_elements: int | None,
    fill_threshold: float | None,
    max_small: int | None,
) -> dict[str, dict[str, Any]]:
    return {
        "sampling": {"candidates": candidates, "large_ratio": large_ratio, "on_short_stratum": on_short},
        "layout": {"max_elements": max_elements, "fill_threshold": fill_threshold, "max_small": max_small},
    }


Candidates = Annotated[Optional[int], typer.Option(help="K candidates per page.")]
LargeRatio = Annotated[Optional[float], typer.Option(help="rho: share of large candidates.")]
OnShort = Annotated[Optional[str], typer.Option(help="error | take-all when a stratum is short.")]
MaxElements = Annotated[Optional[int], typer.Option(min=0, help="N per page; 0 = no limit.")]
FillThreshold = Annotated[Optional[float], typer.Option(help="Best fill below this stops the page.")]
MaxSmall = Annotated[Optional[int], typer.Option(min=-1, help="Small-element cap; -1 = no cap.")]
Width = Annotated[Optional[tuple[int, int]], typer.Option(help="Page width range, px: LO HI.")]
Aspect = Annotated[Optional[tuple[float, float]], typer.Option(help="H/W range: LO HI.")]
MaxHeight = Annotated[Optional[int], typer.Option(help="Page height cap, px.")]
LabelFormats = Annotated[Optional[list[LabelFormat]], typer.Option(help="Repeatable: yolo | yolo-poly | coco.")]
KeepAspect = Annotated[
    Optional[bool], typer.Option("--keep-aspect/--stretch", help="Fit crops in their box, not stretch.")
]
CategoryOffset = Annotated[
    Optional[int], typer.Option(help="class = category id + N (repo: -1). Default: contiguous 0-based.")
]
UseVariants = Annotated[Optional[bool], typer.Option("--use-variants/--no-variants")]
Subset = Annotated[
    Optional[str],
    typer.Option(help="Subset name: images/<subset>/, annotations/instances_<subset>.json. Default: default."),
]


def _render_overrides(
    width: Any,
    aspect: Any,
    max_height: Any,
    label_format: Any,
    keep_aspect: Any,
    category_offset: Any,
    use_variants: Any,
    subset: Any,
) -> dict[str, Any]:
    return {
        "width": width,
        "aspect": aspect,
        "max_height": max_height,
        "label_formats": [f.value for f in label_format] if label_format else None,
        "keep_aspect": keep_aspect,
        "category_offset": category_offset,
        "use_variants": use_variants,
        "subset": subset,
    }


def _print_seed(seed: int) -> None:
    typer.echo(f"seed: {seed}", err=True)


# --- pool ----------------------------------------------------------------------------------


@pool_app.command("build")
@_guard
def pool_build(
    ctx: typer.Context,
    out: Annotated[Path, typer.Option()],
    coco: Annotated[Optional[Path], typer.Option(exists=True, dir_okay=False)] = None,
    images_dir: Annotated[Optional[Path], typer.Option(exists=True, file_okay=False)] = None,
    dataset: Annotated[
        Optional[Path],
        typer.Option(
            exists=True,
            file_okay=False,
            help="COCO dataset folder: images/<subset>/ + annotations/instances_<subset>.json. "
            "Replaces --coco and --images-dir.",
        ),
    ] = None,
    subset: Annotated[Optional[str], typer.Option(help="With --dataset. Default: the only one found.")] = None,
    min_size: Annotated[Optional[float], typer.Option(help="epsilon: drop w or h below this.")] = None,
    exclude_category: Annotated[Optional[list[str]], typer.Option(help="Repeatable.")] = None,
) -> None:
    """Crop every COCO instance into an element pool."""
    from .pool import build_pool, resolve_dataset

    if dataset is not None:
        if coco or images_dir:
            raise typer.BadParameter("--dataset replaces --coco and --images-dir")
        coco, images_dir = resolve_dataset(dataset, subset)
    elif coco is None or images_dir is None or subset is not None:
        raise typer.BadParameter("give --dataset, or --coco with --images-dir (--subset needs --dataset)")
    s = _settings(ctx, pool={"min_size": min_size, "exclude_categories": exclude_category or None})
    out.mkdir(parents=True, exist_ok=True)
    with _bar(None, "cropping") as tick:
        m = build_pool(coco, images_dir, out, s.pool, tick)
    log.info("wrote %s (%d elements)", out / "pool.json", len(m.elements))


@pool_app.command("augment")
@_guard
def pool_augment(
    ctx: typer.Context,
    pool: Annotated[Path, typer.Option(exists=True, help="pool.json or its folder.")],
    min_count: Annotated[int, typer.Option(min=0, help="Rare = category count <= this.")] = 100,
    aug_times: Annotated[int, typer.Option(min=1)] = 50,
    seed: Seed = None,
    flip: Annotated[Optional[bool], typer.Option("--flip/--no-flip")] = None,
) -> None:
    """Add appearance variants for rare categories (needs the [augment] extra)."""
    from .augment import augment_pool

    s = _settings(ctx, augment={"flip": flip})
    seed = secrets.randbits(63) if seed is None else seed
    _print_seed(seed)
    with _bar(None, "augmenting") as tick:
        augment_pool(
            _pool_file(pool), min_count=min_count, aug_times=aug_times, seed=seed, flip=s.augment.flip, tick=tick
        )


# --- layout / render / run -----------------------------------------------------------------


@layout_app.command("generate")
@_guard
def layout_generate(
    ctx: typer.Context,
    pool: Annotated[Path, typer.Option(exists=True, help="pool.json or its folder.")],
    out: Annotated[Path, typer.Option(help="layouts.jsonl")],
    count: Annotated[int, typer.Option(min=1)],
    seed: Seed = None,
    workers: Workers = DEFAULT_WORKERS,
    resume: Resume = False,
    overwrite: Overwrite = False,
    dry_run: DryRun = False,
    candidates: Candidates = None,
    large_ratio: LargeRatio = None,
    on_short_stratum: OnShort = None,
    max_elements: MaxElements = None,
    fill_threshold: FillThreshold = None,
    max_small: MaxSmall = None,
    check_files: CheckFiles = False,
    max_failures: MaxFailures = 0,
) -> None:
    """pool.json -> layouts.jsonl"""
    from .runner import Counts, WorkerSpec, check_pool, generate_layouts

    s = _settings(
        ctx, **_layout_overrides(candidates, large_ratio, on_short_stratum, max_elements, fill_threshold, max_small)
    )
    pool = _pool_file(pool)
    run_json = out.parent / "run.json"
    _check_output([out], resume, overwrite)
    check_pool(pool, s, check_files=check_files)
    if dry_run:
        return _dry_run(s, [out, run_json])
    seed = _resolve_seed(seed, resume, run_json)
    _print_seed(seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    spec = WorkerSpec(str(pool), s.model_dump_json(), ctx.obj["preset"], seed)
    _run_json(run_json, ctx, s, seed, pool, Counts(), count)
    with _interruptible() as stop, _bar(count, "layouts") as tick:
        c = generate_layouts(
            spec, count, out, workers=workers, resume=resume, stop=stop, progress=tick, max_failures=max_failures
        )
    _run_json(run_json, ctx, s, seed, pool, c, count)
    log.info("layouts: %d done, %d failed, %d already done", c.done, c.failures, c.skipped)
    if stop.is_set():
        raise typer.Exit(130)


def _dataset_dirs(out: Path, s: Settings) -> list[Path]:
    """Directories of the dataset (CVAT COCO 1.0): images/<subset>/ and, for YOLO, labels/<subset>/
    mirroring it. The COCO file lands in annotations/ when the run finishes."""
    r = s.render
    dirs = [out / "images" / r.subset]
    if any(f in r.label_formats for f in ("yolo", "yolo-poly")):
        dirs.append(out / "labels" / r.subset)
    return dirs


def _dataset_paths(out: Path, s: Settings) -> list[Path]:
    """Everything a render or run writes, for --dry-run. Run files sit at the root."""
    coco = [out / coco_name(s.render)] if "coco" in s.render.label_formats else []
    return [*_dataset_dirs(out, s), *coco, out / "classes.txt", out / "run.json"]


@app.command("render")
@_guard
def render(
    ctx: typer.Context,
    layouts: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    pool: Annotated[Path, typer.Option(exists=True, help="pool.json or its folder.")],
    out: Annotated[Path, typer.Option()],
    seed: Annotated[Optional[int], typer.Option(help="Override the render seed stored in layouts.")] = None,
    workers: Workers = DEFAULT_WORKERS,
    resume: Resume = False,
    overwrite: Overwrite = False,
    dry_run: DryRun = False,
    width: Width = None,
    aspect: Aspect = None,
    max_height: MaxHeight = None,
    label_format: LabelFormats = None,
    keep_aspect: KeepAspect = None,
    category_offset: CategoryOffset = None,
    use_variants: UseVariants = None,
    subset: Subset = None,
    check_files: CheckFiles = False,
    max_failures: MaxFailures = 0,
) -> None:
    """layouts.jsonl + pool -> images + labels"""
    from .runner import WorkerSpec, finalize_dataset, iter_lines, render_layouts

    s = _settings(
        ctx,
        render=_render_overrides(
            width, aspect, max_height, label_format, keep_aspect, category_offset, use_variants, subset
        ),
    )
    pool = _pool_file(pool)
    manifest = load_manifest(pool, check_files=check_files)
    _check_output([out / "run.json", out / "images" / s.render.subset], resume, overwrite)
    if dry_run:
        return _dry_run(s, _dataset_paths(out, s))
    for d in _dataset_dirs(out, s):
        d.mkdir(parents=True, exist_ok=True)
    spec = WorkerSpec(str(pool), s.model_dump_json(), ctx.obj["preset"], 0, seed, str(out))
    total = sum(1 for _ in iter_lines(layouts))
    with _interruptible() as stop, _bar(total, "render") as tick:
        c = render_layouts(
            spec, layouts, workers=workers, resume=resume, stop=stop, progress=tick, max_failures=max_failures
        )
    if stop.is_set():
        raise typer.Exit(130)
    finalize_dataset(spec, layouts, manifest)
    _run_json(out / "run.json", ctx, s, seed, pool, c, total)
    log.info("render: %d done, %d failed, %d already done", c.done, c.failures, c.skipped)


@app.command("run")
@_guard
def run(
    ctx: typer.Context,
    pool: Annotated[Path, typer.Option(exists=True, help="pool.json or its folder.")],
    count: Annotated[int, typer.Option(min=1)],
    out: Annotated[Path, typer.Option()],
    seed: Seed = None,
    workers: Workers = DEFAULT_WORKERS,
    resume: Resume = False,
    overwrite: Overwrite = False,
    dry_run: DryRun = False,
    candidates: Candidates = None,
    large_ratio: LargeRatio = None,
    on_short_stratum: OnShort = None,
    max_elements: MaxElements = None,
    fill_threshold: FillThreshold = None,
    max_small: MaxSmall = None,
    width: Width = None,
    aspect: Aspect = None,
    max_height: MaxHeight = None,
    label_format: LabelFormats = None,
    keep_aspect: KeepAspect = None,
    category_offset: CategoryOffset = None,
    use_variants: UseVariants = None,
    subset: Subset = None,
    check_files: CheckFiles = False,
    max_failures: MaxFailures = 0,
) -> None:
    """layout generate + render, streamed (also writes layouts.jsonl)."""
    from .runner import Counts, WorkerSpec, check_pool, finalize_dataset, run_pages

    s = _settings(
        ctx,
        **_layout_overrides(candidates, large_ratio, on_short_stratum, max_elements, fill_threshold, max_small),
        render=_render_overrides(
            width, aspect, max_height, label_format, keep_aspect, category_offset, use_variants, subset
        ),
    )
    pool = _pool_file(pool)
    layouts = out / "layouts.jsonl"
    _check_output([layouts, out / "run.json", out / "images" / s.render.subset], resume, overwrite)
    check_pool(pool, s, check_files=check_files)
    if dry_run:
        return _dry_run(s, [layouts, *_dataset_paths(out, s)])
    seed = _resolve_seed(seed, resume, out / "run.json")
    _print_seed(seed)
    for d in _dataset_dirs(out, s):
        d.mkdir(parents=True, exist_ok=True)
    spec = WorkerSpec(str(pool), s.model_dump_json(), ctx.obj["preset"], seed, None, str(out))
    _run_json(out / "run.json", ctx, s, seed, pool, Counts(), count)
    with _interruptible() as stop, _bar(count, "pages") as tick:
        c = run_pages(spec, count, workers=workers, resume=resume, stop=stop, progress=tick, max_failures=max_failures)
    if stop.is_set():
        raise typer.Exit(130)
    finalize_dataset(spec, layouts, load_manifest(pool))
    _run_json(out / "run.json", ctx, s, seed, pool, c, count)
    log.info("run: %d done, %d failed, %d already done", c.done, c.failures, c.skipped)


# --- inspect -------------------------------------------------------------------------------


@app.command("viz")
@_guard
def viz(
    layouts: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    out: Annotated[Path, typer.Option(help="PNG file.")],
    rows: Annotated[int, typer.Option(min=1)] = 2,
    cols: Annotated[int, typer.Option(min=1)] = 2,
    labels: Annotated[bool, typer.Option("--labels", help="Print category ids.")] = False,
    pool: Annotated[Optional[Path], typer.Option(help="Draw pool crops instead of bare boxes.")] = None,
) -> None:
    """layouts.jsonl -> PNG grid of boxes (or crops)."""
    from .runner import iter_records
    from .viz import draw_grid

    elements, root = None, None
    if pool is not None:
        pool = _pool_file(pool)
        elements, root = {e.id: e for e in load_manifest(pool).elements}, pool.parent
    out.parent.mkdir(parents=True, exist_ok=True)
    draw_grid(iter_records(layouts), rows, cols, labels=labels, elements=elements, root=root).save(out, format="PNG")


@app.command("stats")
@_guard
def stats(
    layouts: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    as_json: Annotated[bool, typer.Option("--json", help="JSON on stdout.")] = False,
) -> None:
    """layouts.jsonl -> density / alignment / category counts."""
    from .runner import iter_records
    from .stats import compute_stats

    result = compute_stats(iter_records(layouts))
    if as_json:
        typer.echo(orjson.dumps(result, option=orjson.OPT_INDENT_2).decode())
        return
    from rich.table import Table

    t = Table("metric", "value")
    for k, v in result.items():
        t.add_row(k, f"{v:.4f}" if isinstance(v, float) else str(v))
    Console(stderr=True).print(t)
