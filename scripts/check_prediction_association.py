"""Stage exported predictions and run the original review tool in isolation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image

from src.association import associate_prediction


def _image_path(prediction_path: Path, value: str, override: Path | None) -> Path:
    if override is not None:
        candidates = [override.expanduser()]
    else:
        image = Path(value).expanduser()
        candidates = [image] if image.is_absolute() else [Path.cwd() / image, prediction_path.parent / image]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Source image not found: {value!r}; provide --image with its local path")


def run_association_check(
    prediction_paths: list[Path], output_dir: Path, image_override: Path | None = None
) -> dict:
    """Run a batch at the review tool's existing default threshold of 0.6."""
    if not prediction_paths:
        raise ValueError("Provide at least one prediction JSON")
    if image_override is not None and len(prediction_paths) != 1:
        raise ValueError("--image is only available for a single prediction")
    destination = output_dir.expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(f"Use a new or empty output directory: {destination}")

    prepared = []
    image_names = set()
    for prediction_path in prediction_paths:
        source = prediction_path.expanduser().resolve()
        payload = json.loads(source.read_text(encoding="utf-8-sig"))
        associated = associate_prediction(payload, min_score=0.6)
        image = _image_path(source, payload["image"], image_override)
        if image.name in image_names:
            raise ValueError(f"Duplicate image filename in batch: {image.name}")
        image_names.add(image.name)
        with Image.open(image) as opened:
            if opened.size != (payload["width"], payload["height"]):
                raise ValueError(f"Image dimensions disagree with JSON: {image}")
        associated["image"] = image.name
        prepared.append((source, image, associated))

    image_dir = destination / "input/images"
    prediction_dir = destination / "input/predictions"
    image_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    for number, (_, image, associated) in enumerate(prepared, start=1):
        shutil.copy2(image, image_dir / image.name)
        (prediction_dir / f"{number:04d}_{image.stem}.json").write_text(
            json.dumps(associated, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    environment = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    completed = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "PPE_check_results.py")],
        cwd=destination, env=environment, capture_output=True,
        text=True, encoding="utf-8", errors="replace", timeout=120,
    )
    if completed.returncode:
        raise RuntimeError(f"Review tool failed:\n{completed.stdout}\n{completed.stderr}")
    report_path = destination / "output/report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    summary = {
        "min_score": 0.6,
        "source_count": len(prepared),
        "sources": [
            {
                "prediction": str(source), "image": str(image),
                "data_source": associated.get("data_source", "prediction_json"),
                "associations": associated["associations"],
            }
            for source, image, associated in prepared
        ],
        "report": str(report_path),
        "marked_images": str(destination / "output/images"),
        "counts": report["counts"],
        "review_count": len(report["reviews"]),
        "error_count": len(report["errors"]),
    }
    (destination / "integration_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", type=Path, nargs="+", required=True)
    parser.add_argument("--image", type=Path, help="Local original image, when the JSON contains a cloud path")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/association_check"))
    args = parser.parse_args()
    try:
        summary = run_association_check(args.prediction, args.output_dir, args.image)
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
