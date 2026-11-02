"""Stable data contracts shared by detector, segmenter, and exporters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class BoxXYXY:
    """A bounding box in absolute image pixels: left, top, right, bottom."""

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        values = (self.x1, self.y1, self.x2, self.y2)
        if not all(np.isfinite(value) for value in values):
            raise ValueError(f"Box contains a non-finite value: {values}")
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError(f"Box must have positive width and height: {values}")

    @classmethod
    def from_normalized_cxcywh(
        cls,
        values: list[float] | tuple[float, float, float, float],
        image_width: int,
        image_height: int,
    ) -> "BoxXYXY":
        """Convert Grounding DINO's normalized cxcywh box to pixel xyxy."""

        if len(values) != 4:
            raise ValueError(f"Expected four box values, received: {values}")
        if image_width <= 0 or image_height <= 0:
            raise ValueError("Image dimensions must be positive")

        center_x, center_y, width, height = map(float, values)
        x1 = (center_x - width / 2.0) * image_width
        y1 = (center_y - height / 2.0) * image_height
        x2 = (center_x + width / 2.0) * image_width
        y2 = (center_y + height / 2.0) * image_height
        return cls(x1, y1, x2, y2).clip(image_width, image_height)

    def clip(self, image_width: int, image_height: int) -> "BoxXYXY":
        x1 = min(max(self.x1, 0.0), float(image_width))
        y1 = min(max(self.y1, 0.0), float(image_height))
        x2 = min(max(self.x2, 0.0), float(image_width))
        y2 = min(max(self.y2, 0.0), float(image_height))
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Box lies outside the image after clipping")
        return BoxXYXY(x1, y1, x2, y2)

    def as_list(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]


@dataclass(frozen=True)
class Detection:
    label: str
    score: float
    box: BoxXYXY
    raw_phrase: str

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("Detection label cannot be empty")
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"Detection score must be in [0, 1], got {self.score}")


@dataclass(frozen=True)
class DetectionBatch:
    image_path: Path
    image_rgb: NDArray[np.uint8]
    prompt: str
    detections: tuple[Detection, ...]


@dataclass(frozen=True)
class SegmentedInstance:
    detection: Detection
    mask: NDArray[np.bool_]
    mask_score: float

    def __post_init__(self) -> None:
        if self.mask.ndim != 2:
            raise ValueError(f"Mask must be HxW, got shape {self.mask.shape}")
        if not np.isfinite(self.mask_score):
            raise ValueError("Mask score must be finite")


@dataclass(frozen=True)
class PredictionResult:
    image_path: Path
    image_rgb: NDArray[np.uint8]
    prompt: str
    instances: tuple[SegmentedInstance, ...]

