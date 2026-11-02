"""Adapt prediction.json records without changing the learner's algorithm."""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any

from association_practice import associate_ppe_boxes
from src.schema import BoxXYXY


LABELS = ("person", "head", "helmet")


def _finite_number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def associate_prediction(payload: dict, min_score: float = 0.6) -> dict:
    """Return a copy with associations expressed as original instance IDs.

    All original instances and model metadata are retained. Only detections
    meeting min_score participate in matching, so a low-confidence box cannot
    consume a head or helmet needed by a qualified detection.
    """
    if not _finite_number(min_score) or not 0 <= min_score <= 1:
        raise ValueError("min_score must be a finite number in [0, 1]")
    if not isinstance(payload, dict):
        raise ValueError("Prediction must be a JSON object")
    if not isinstance(payload.get("image"), str) or not payload["image"]:
        raise ValueError("Prediction requires a non-empty image path")
    if any(type(payload.get(key)) is not int or payload[key] <= 0
           for key in ("width", "height")):
        raise ValueError("Prediction requires positive integer width and height")
    if not isinstance(payload.get("instances"), list):
        raise ValueError("Prediction requires an instances list")

    boxes: dict[str, list[BoxXYXY]] = {label: [] for label in LABELS}
    instance_ids: dict[str, list[int]] = {label: [] for label in LABELS}
    seen_ids: set[int] = set()
    for position, instance in enumerate(payload["instances"]):
        if not isinstance(instance, dict):
            raise ValueError(f"Instance {position} must be an object")
        instance_id = instance.get("instance_id")
        if type(instance_id) is not int:
            raise ValueError(f"Instance {position} requires an integer instance_id")
        if instance_id in seen_ids:
            raise ValueError(f"Duplicate instance_id: {instance_id}")
        seen_ids.add(instance_id)
        label = instance.get("label")
        if label not in LABELS:
            raise ValueError(f"Unsupported label for instance {instance_id}: {label!r}")
        score = instance.get("detection_score")
        if not _finite_number(score) or not 0 <= score <= 1:
            raise ValueError(f"Invalid detection_score for instance {instance_id}")
        values = instance.get("box_xyxy")
        if not isinstance(values, list) or len(values) != 4 or not all(
            _finite_number(value) for value in values
        ):
            raise ValueError(f"Invalid box_xyxy for instance {instance_id}")
        box = BoxXYXY(*values)
        if box.x1 < 0 or box.y1 < 0 or box.x2 > payload["width"] or box.y2 > payload["height"]:
            raise ValueError(f"Box outside image for instance {instance_id}")
        if score >= min_score:
            boxes[label].append(box)
            instance_ids[label].append(instance_id)

    rows = associate_ppe_boxes(boxes["person"], boxes["head"], boxes["helmet"])

    def original_id(label: str, index: int | None) -> int | None:
        return None if index is None else instance_ids[label][index]

    result = deepcopy(payload)
    result["association_min_score"] = min_score
    result["associations"] = [
        {
            "person_id": original_id("person", row["person_index"]),
            "head_id": original_id("head", row["head_index"]),
            "helmet_id": original_id("helmet", row["helmet_index"]),
        }
        for row in rows
    ]
    return result
