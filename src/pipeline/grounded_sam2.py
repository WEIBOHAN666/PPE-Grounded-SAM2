"""Compose open-vocabulary detection with box-prompted segmentation."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
from numpy.typing import NDArray

from src.schema import Detection, DetectionBatch, PredictionResult, SegmentedInstance


class Detector(Protocol):
    def detect(self, image_path: str | Path) -> DetectionBatch: ...


class Segmenter(Protocol):
    def segment(
        self,
        image_rgb: NDArray[np.uint8],
        detections: Sequence[Detection],
    ) -> tuple[SegmentedInstance, ...]: ...


class GroundedSam2Pipeline:
    def __init__(self, detector: Detector, segmenter: Segmenter) -> None:
        self.detector = detector
        self.segmenter = segmenter

    def predict(self, image_path: str | Path) -> PredictionResult:
        detection_batch = self.detector.detect(image_path)
        instances = self.segmenter.segment(
            detection_batch.image_rgb,
            detection_batch.detections,
        )
        return PredictionResult(
            image_path=detection_batch.image_path,
            image_rgb=detection_batch.image_rgb,
            prompt=detection_batch.prompt,
            instances=instances,
        )

