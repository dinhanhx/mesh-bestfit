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
    flip: bool = True  # Paper 1 (arxiv.org/abs/2410.12628) adds a random flip; the repo does not


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
