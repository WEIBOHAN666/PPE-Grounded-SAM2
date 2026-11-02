"""Adapter around the official Grounding DINO inference API."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.schema import BoxXYXY, Detection, DetectionBatch


DEFAULT_CLASSES = ("person", "head", "helmet")


def normalize_prompt(prompt: str) -> str:
    normalized = prompt.lower().strip()
    if not normalized:
        raise ValueError("Grounding DINO prompt cannot be empty")
    return normalized if normalized.endswith(".") else normalized + "."


def normalize_phrase(phrase: str, classes: Sequence[str] = DEFAULT_CLASSES) -> str:
    """Map a free-form Grounding DINO phrase to one project class."""

    cleaned = re.sub(r"[^a-z0-9_ -]+", " ", phrase.lower()).strip()
    normalized_classes = [item.lower().strip() for item in classes]
    if cleaned in normalized_classes:
        return cleaned

    tokens = set(cleaned.replace("-", " ").split())
    matches = [item for item in normalized_classes if item in tokens]
    if len(matches) == 1:
        return matches[0]
    return cleaned or "unknown"


def _as_float(value: Any) -> float:
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


class GroundingDinoDetector:
    """Load the official model lazily and return project-native detections."""

    def __init__(
        self,
        model_config_path: str | Path,
        model_checkpoint_path: str | Path,
        *,
        prompt: str = "person. head. helmet.",
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        device: str = "cuda",
        classes: Sequence[str] = DEFAULT_CLASSES,
        api: Any | None = None,
        model: Any | None = None,
    ) -> None:
        self.model_config_path = Path(model_config_path).expanduser().resolve()
        self.model_checkpoint_path = Path(model_checkpoint_path).expanduser().resolve()
        self.prompt = normalize_prompt(prompt)
        self.box_threshold = float(box_threshold)
        self.text_threshold = float(text_threshold)
        self.device = device
        self.classes = tuple(classes)

        if not 0.0 <= self.box_threshold <= 1.0:
            raise ValueError("box_threshold must be in [0, 1]")
        if not 0.0 <= self.text_threshold <= 1.0:
            raise ValueError("text_threshold must be in [0, 1]")

        if api is None:
            if not self.model_config_path.is_file():
                raise FileNotFoundError(
                    f"Grounding DINO config not found: {self.model_config_path}"
                )
            if not self.model_checkpoint_path.is_file():
                raise FileNotFoundError(
                    f"Grounding DINO checkpoint not found: {self.model_checkpoint_path}"
                )
            try:
                from groundingdino.util import inference as official_api
            except ImportError as exc:
                raise RuntimeError(
                    "Grounding DINO is not installed. Install the official repository "
                    "with: pip install -e third_party/GroundingDINO"
                ) from exc
            api = official_api

        self._api = api
        self._model = model or self._api.load_model(
            str(self.model_config_path),
            str(self.model_checkpoint_path),
            device=self.device,
        )

    def detect(self, image_path: str | Path) -> DetectionBatch:
        path = Path(image_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Input image not found: {path}")

        image_rgb, image_tensor = self._api.load_image(str(path))
        image_rgb = np.asarray(image_rgb, dtype=np.uint8)
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError(f"Expected RGB image, got shape {image_rgb.shape}")

        boxes, scores, phrases = self._api.predict(
            model=self._model,
            image=image_tensor,
            caption=self.prompt,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            device=self.device,
        )
        box_values = boxes.tolist() if hasattr(boxes, "tolist") else list(boxes)
        score_values = list(scores)
        phrase_values = list(phrases)
        if not (len(box_values) == len(score_values) == len(phrase_values)):
            raise RuntimeError("Grounding DINO returned mismatched boxes, scores, and labels")

        height, width = image_rgb.shape[:2]
        detections: list[Detection] = []
        for box, score, phrase in zip(box_values, score_values, phrase_values):
            try:
                pixel_box = BoxXYXY.from_normalized_cxcywh(box, width, height)
            except ValueError:
                continue
            raw_phrase = str(phrase)
            detections.append(
                Detection(
                    label=normalize_phrase(raw_phrase, self.classes),
                    score=_as_float(score),
                    box=pixel_box,
                    raw_phrase=raw_phrase,
                )
            )

        return DetectionBatch(
            image_path=path,
            image_rgb=image_rgb,
            prompt=self.prompt,
            detections=tuple(detections),
        )

