from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DataError(Exception):
    """Invalid input data: bad COCO file, duplicate ids, corrupt pool manifest (exit code 3)."""


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Source(_Model):
    coco: str
    sha256: str


class Category(_Model):
    id: int
    name: str


class Element(_Model):
    id: str
    category_id: int
    w: float = Field(gt=0, le=1)
    h: float = Field(gt=0, le=1)
    path: str
    variants: tuple[str, ...] = ()


class PoolManifest(_Model):
    version: Literal[1] = 1
    source: Source
    categories: tuple[Category, ...]
    elements: tuple[Element, ...]

    @model_validator(mode="after")
    def _consistent(self) -> PoolManifest:
        cats = {c.id for c in self.categories}
        if len(cats) != len(self.categories):
            raise ValueError("duplicate category id")
        ids = {e.id for e in self.elements}
        if len(ids) != len(self.elements):
            raise ValueError("duplicate element id")
        unknown = {e.category_id for e in self.elements} - cats
        if unknown:
            raise ValueError(f"unknown category id(s): {sorted(unknown)}")
        return self


class SeedInfo(_Model):
    entropy: int
    spawn_key: tuple[int, ...]


class LayoutRecord(_Model):
    page: int
    seed: SeedInfo
    preset: str
    boxes: tuple[tuple[float, float, float, float], ...]
    category_ids: tuple[int, ...]
    element_ids: tuple[str, ...]
    stop_reason: Literal["max_elements", "no_fit", "below_threshold", "small_cap", "pool_exhausted"]


def load_manifest(path: Path, *, check_files: bool = False) -> PoolManifest:
    try:
        m = PoolManifest.model_validate_json(path.read_bytes())
    except ValueError as e:  # includes pydantic.ValidationError
        raise DataError(f"{path}: {e}") from e
    if check_files:
        for el in m.elements:
            for rel in (el.path, *el.variants):
                if not (path.parent / rel).is_file():
                    raise DataError(f"{path}: element {el.id}: missing file {rel}")
    return m
