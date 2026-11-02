"""Render COCO bounding boxes on a reproducible sample of images."""

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path
from typing import Any

from src.datasets.coco_dataset import CocoSplit


COLORS = {
    "head": "#ff8c00",
    "helmet": "#00c853",
    "person": "#2979ff",
}


def _draw_label(draw: Any, position: tuple[float, float], label: str, color: str) -> None:
    x, y = position
    try:
        left, top, right, bottom = draw.textbbox((x, y), label)
        draw.rectangle((left - 2, top - 2, right + 2, bottom + 2), fill=color)
    except AttributeError:
        draw.rectangle((x, y, x + 8 * len(label), y + 13), fill=color)
    draw.text((x, y), label, fill="white")


def visualize_split(
    split_dir: str | Path,
    output_dir: str | Path,
    count: int = 8,
    seed: int = 42,
) -> list[Path]:
    """Draw ground-truth boxes for a sample and return written image paths."""

    try:
        from PIL import Image, ImageDraw, ImageOps
    except ImportError as exc:
        raise RuntimeError(
            "Pillow is required for visualization. Run: pip install -r requirements.txt"
        ) from exc

    coco = CocoSplit.load(split_dir)
    annotations_by_image = coco.annotations_by_image()
    category_names = coco.category_names
    candidates = [
        image
        for image in coco.images
        if isinstance(image.get("id"), int)
        and annotations_by_image.get(image["id"])
        and isinstance(image.get("file_name"), str)
        and (coco.directory / image["file_name"]).is_file()
    ]

    if count <= 0:
        raise ValueError("count must be greater than zero")
    if not candidates:
        raise ValueError(f"No annotated images found in {coco.directory}")

    rng = random.Random(seed)
    selected = rng.sample(candidates, k=min(count, len(candidates)))
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for index, image_record in enumerate(selected, start=1):
        source_path = coco.directory / image_record["file_name"]
        with Image.open(source_path) as source:
            canvas = ImageOps.exif_transpose(source).convert("RGB")

        draw = ImageDraw.Draw(canvas)
        per_image_counts: Counter[str] = Counter()
        for annotation in annotations_by_image[image_record["id"]]:
            bbox = annotation.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            x, y, width, height = map(float, bbox)
            if width <= 0 or height <= 0:
                continue

            name = category_names.get(annotation.get("category_id"), "unknown")
            color = COLORS.get(name.lower(), "#e91e63")
            per_image_counts[name] += 1
            draw.rectangle((x, y, x + width, y + height), outline=color, width=3)
            _draw_label(draw, (x + 2, max(0, y - 15)), name, color)

        summary = " | ".join(f"{name}: {value}" for name, value in per_image_counts.items())
        _draw_label(draw, (6, 6), summary or "no annotations", "#212121")

        safe_name = Path(image_record["file_name"]).stem
        output_path = destination / f"{coco.name}_{index:02d}_{safe_name}.jpg"
        canvas.save(output_path, quality=92)
        written.append(output_path)

    return written

