"""Save masks, metadata, and an overlay for one prediction."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from src.schema import PredictionResult


COLORS = {
    "person": (41, 121, 255),
    "head": (255, 140, 0),
    "helmet": (0, 200, 83),
}


def export_prediction(result: PredictionResult, output_dir: str | Path) -> dict[str, Path]:
    destination = Path(output_dir).expanduser().resolve()
    mask_dir = destination / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    overlay = result.image_rgb.astype(np.float32).copy()
    records: list[dict] = []

    for index, instance in enumerate(result.instances, start=1):
        detection = instance.detection
        color = COLORS.get(detection.label, (233, 30, 99))
        color_array = np.asarray(color, dtype=np.float32)
        overlay[instance.mask] = overlay[instance.mask] * 0.55 + color_array * 0.45

        mask_name = f"{index:03d}_{detection.label}.png"
        mask_path = mask_dir / mask_name
        Image.fromarray(instance.mask.astype(np.uint8) * 255).save(mask_path)
        records.append(
            {
                "instance_id": index,
                "label": detection.label,
                "raw_phrase": detection.raw_phrase,
                "detection_score": round(detection.score, 6),
                "mask_score": round(instance.mask_score, 6),
                "box_xyxy": [round(value, 2) for value in detection.box.as_list()],
                "mask_area_pixels": int(instance.mask.sum()),
                "mask_file": str(Path("masks") / mask_name),
            }
        )

    overlay_image = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(overlay_image)
    for instance in result.instances:
        detection = instance.detection
        color = COLORS.get(detection.label, (233, 30, 99))
        box = detection.box.as_list()
        draw.rectangle(box, outline=color, width=3)
        label = f"{detection.label} {detection.score:.2f} / mask {instance.mask_score:.2f}"
        x, y = box[0] + 2, max(0, box[1] - 14)
        draw.rectangle((x - 2, y - 2, x + 7 * len(label), y + 13), fill=color)
        draw.text((x, y), label, fill="white")

    overlay_path = destination / "overlay.jpg"
    overlay_image.save(overlay_path, quality=92)

    metadata = {
        "image": str(result.image_path),
        "width": int(result.image_rgb.shape[1]),
        "height": int(result.image_rgb.shape[0]),
        "prompt": result.prompt,
        "instance_count": len(records),
        "instances": records,
    }
    json_path = destination / "prediction.json"
    json_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"json": json_path, "overlay": overlay_path, "masks": mask_dir}
