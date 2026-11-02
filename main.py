"""Command-line entry point for data preparation and model inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.datasets import inspect_dataset
from src.environment import environment_report
from src.visualization import visualize_split


def _print_report(report: dict) -> None:
    print(f"Dataset: {report['dataset_root']}")
    print(
        f"Total: {report['total_images']} images, "
        f"{report['total_annotations']} annotations"
    )
    for split in report["splits"]:
        print(f"\n[{split['split']}]")
        print(f"  images: {split['images']}")
        print(f"  annotations: {split['annotations']}")
        print(f"  classes: {split['annotations_by_category']}")
        print(f"  images without annotations: {split['images_without_annotations']}")
        print(f"  missing image files: {len(split['missing_image_files'])}")
        print(f"  invalid annotations: {split['invalid_annotations']}")
        print(f"  boxes outside image: {split['boxes_outside_image']}")


def inspect_command(args: argparse.Namespace) -> int:
    report = inspect_dataset(args.dataset_root, splits=args.splits)
    _print_report(report)

    report_path = Path(args.report).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nJSON report written to: {report_path}")

    has_critical_errors = any(
        split["missing_image_files"]
        or split["invalid_annotations"]
        or split["duplicate_image_ids"]
        or split["duplicate_annotation_ids"]
        for split in report["splits"]
    )
    return 1 if has_critical_errors else 0


def visualize_command(args: argparse.Namespace) -> int:
    split_dir = Path(args.dataset_root).expanduser() / args.split
    written = visualize_split(
        split_dir=split_dir,
        output_dir=args.output_dir,
        count=args.count,
        seed=args.seed,
    )
    print(f"Wrote {len(written)} visualizations to:")
    for path in written:
        print(f"  {path}")
    return 0


def doctor_command(args: argparse.Namespace) -> int:
    report = environment_report()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["model_runtime_ready"]:
        print("\nModel runtime dependencies are installed.")
    else:
        print(
            "\nLocal data tools are ready; model dependencies are intentionally "
            "deferred to the cloud GPU environment."
        )
    return 0


def infer_command(args: argparse.Namespace) -> int:
    from src.detection import GroundingDinoDetector
    from src.export import export_prediction
    from src.pipeline import GroundedSam2Pipeline
    from src.segmentation import Sam2Segmenter

    detector = GroundingDinoDetector(
        model_config_path=args.gdino_config,
        model_checkpoint_path=args.gdino_checkpoint,
        prompt=args.prompt,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        device=args.device,
    )
    segmenter = Sam2Segmenter(
        config_name=args.sam2_config,
        checkpoint_path=args.sam2_checkpoint,
        device=args.device,
    )
    pipeline = GroundedSam2Pipeline(detector, segmenter)
    result = pipeline.predict(args.image)
    paths = export_prediction(result, args.output_dir)
    print(f"Detected and segmented {len(result.instances)} instances.")
    print(f"Metadata: {paths['json']}")
    print(f"Overlay: {paths['overlay']}")
    print(f"Masks: {paths['masks']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PPE-Grounded-SAM2 data preparation and inference tools"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser(
        "inspect", help="Validate COCO annotations and image files"
    )
    inspect_parser.add_argument("--dataset-root", required=True, type=Path)
    inspect_parser.add_argument(
        "--splits", nargs="+", default=["train", "valid", "test"]
    )
    inspect_parser.add_argument(
        "--report", type=Path, default=Path("outputs/dataset_report.json")
    )
    inspect_parser.set_defaults(handler=inspect_command)

    visualize_parser = subparsers.add_parser(
        "visualize", help="Draw COCO ground-truth boxes on sample images"
    )
    visualize_parser.add_argument("--dataset-root", required=True, type=Path)
    visualize_parser.add_argument("--split", choices=["train", "valid", "test"], default="train")
    visualize_parser.add_argument("--count", type=int, default=8)
    visualize_parser.add_argument("--seed", type=int, default=42)
    visualize_parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/dataset_preview")
    )
    visualize_parser.set_defaults(handler=visualize_command)

    doctor_parser = subparsers.add_parser(
        "doctor", help="Report whether local and cloud model dependencies are ready"
    )
    doctor_parser.set_defaults(handler=doctor_command)

    infer_parser = subparsers.add_parser(
        "infer", help="Run Grounding DINO followed by SAM 2.1 on one image"
    )
    infer_parser.add_argument("--image", required=True, type=Path)
    infer_parser.add_argument("--gdino-config", required=True, type=Path)
    infer_parser.add_argument("--gdino-checkpoint", required=True, type=Path)
    infer_parser.add_argument(
        "--sam2-config",
        default="configs/sam2.1/sam2.1_hiera_s.yaml",
        help="SAM2 Hydra config name from the official repository",
    )
    infer_parser.add_argument("--sam2-checkpoint", required=True, type=Path)
    infer_parser.add_argument("--prompt", default="person. head. helmet.")
    infer_parser.add_argument("--box-threshold", type=float, default=0.35)
    infer_parser.add_argument("--text-threshold", type=float, default=0.25)
    infer_parser.add_argument("--device", default="cuda")
    infer_parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/inference")
    )
    infer_parser.set_defaults(handler=infer_command)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return args.handler(args)
    except (FileNotFoundError, ImportError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    sys.exit(main())
