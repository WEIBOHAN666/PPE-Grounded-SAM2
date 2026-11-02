"""Run association and review on synthetic data without model weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.check_prediction_association import run_association_check


def run_demo(output_dir: Path) -> dict:
    """Create illustrative inputs, run the real postprocessor and check results."""
    destination = output_dir.expanduser().resolve()
    if destination.exists() and (
        not destination.is_dir() or any(destination.iterdir())
    ):
        raise ValueError(f"Use a new or empty output directory: {destination}")

    examples = Path(__file__).resolve().parent
    payload = json.loads((examples / "prediction.json").read_text(encoding="utf-8"))
    image = Image.new("RGB", (payload["width"], payload["height"]), "#f5f7fa")
    draw = ImageDraw.Draw(image)
    draw.text((80, 40), "SYNTHETIC DEMO - NO MODEL INFERENCE", fill="#26354a")
    for offset in (0, 1200, 2400):
        draw.rounded_rectangle((150 + offset, 550, 410 + offset, 850), radius=35, fill="#c2cddd")
        draw.ellipse((180 + offset, 350, 360 + offset, 550), fill="#e3c6aa")
    draw.rounded_rectangle((180, 260, 360, 440), radius=20, fill="#f0c33c")

    source = destination / "source"
    source.mkdir(parents=True, exist_ok=True)
    image_path = source / payload["image"]
    image.save(image_path)
    prediction_path = source / "prediction.json"
    prediction_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # Explicit image path avoids depending on the caller's current directory.
    summary = run_association_check([prediction_path], destination / "check", image_path)
    actual = json.loads(Path(summary["report"]).read_text(encoding="utf-8"))
    expected = json.loads((examples / "expected_report.json").read_text(encoding="utf-8"))
    if actual != expected:
        raise RuntimeError(f"Unexpected demo report: {actual!r}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/demo")
    args = parser.parse_args()
    try:
        summary = run_demo(args.output_dir)
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        parser.error(str(exc))
    print("Synthetic demo passed: 3 persons, 2 heads, 1 helmet; 2 pending reviews.")
    print(f"Report: {summary['report']}")
    print(f"Marked image directory: {summary['marked_images']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
