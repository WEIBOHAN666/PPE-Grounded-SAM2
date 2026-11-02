"""Small, dependency-free COCO reader used before model training.

The first project milestone is deliberately independent of PyTorch.  It lets us
verify a downloaded dataset locally before uploading gigabytes of data to a GPU
server.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ANNOTATION_FILE = "_annotations.coco.json"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _duplicate_values(values: Iterable[Any]) -> list[Any]:
    counts = Counter(values)
    return sorted(value for value, count in counts.items() if count > 1)


@dataclass(frozen=True)
class CocoSplit:
    """Loaded COCO annotations together with their split directory."""

    name: str
    directory: Path
    annotation_path: Path
    images: list[dict[str, Any]]
    annotations: list[dict[str, Any]]
    categories: list[dict[str, Any]]

    @classmethod
    def load(cls, directory: str | Path) -> "CocoSplit":
        split_dir = Path(directory).expanduser().resolve()
        annotation_path = split_dir / ANNOTATION_FILE

        if not split_dir.is_dir():
            raise FileNotFoundError(f"Dataset split does not exist: {split_dir}")
        if not annotation_path.is_file():
            raise FileNotFoundError(f"COCO annotation file not found: {annotation_path}")

        try:
            payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {annotation_path}: {exc}") from exc

        required = ("images", "annotations", "categories")
        missing = [key for key in required if not isinstance(payload.get(key), list)]
        if missing:
            raise ValueError(
                f"Invalid COCO file {annotation_path}; missing list fields: {missing}"
            )

        return cls(
            name=split_dir.name,
            directory=split_dir,
            annotation_path=annotation_path,
            images=payload["images"],
            annotations=payload["annotations"],
            categories=payload["categories"],
        )

    @property
    def category_names(self) -> dict[int, str]:
        return {
            int(category["id"]): str(category["name"])
            for category in self.categories
            if "id" in category and "name" in category
        }

    def annotations_by_image(self) -> dict[int, list[dict[str, Any]]]:
        grouped: dict[int, list[dict[str, Any]]] = {
            int(image["id"]): [] for image in self.images if "id" in image
        }
        for annotation in self.annotations:
            image_id = annotation.get("image_id")
            if isinstance(image_id, int) and image_id in grouped:
                grouped[image_id].append(annotation)
        return grouped

    def inspect(self) -> dict[str, Any]:
        image_ids = [image.get("id") for image in self.images]
        annotation_ids = [annotation.get("id") for annotation in self.annotations]
        category_ids = [category.get("id") for category in self.categories]

        valid_image_ids = {value for value in image_ids if isinstance(value, int)}
        valid_category_ids = {value for value in category_ids if isinstance(value, int)}
        category_names = self.category_names
        category_counts: Counter[int] = Counter()

        missing_files: list[str] = []
        invalid_images = 0
        referenced_files: set[str] = set()
        image_sizes: dict[int, tuple[float, float]] = {}

        for image in self.images:
            image_id = image.get("id")
            file_name = image.get("file_name")
            width = image.get("width")
            height = image.get("height")

            if not isinstance(image_id, int) or not isinstance(file_name, str):
                invalid_images += 1
                continue

            referenced_files.add(Path(file_name).as_posix().lower())
            if not (self.directory / file_name).is_file():
                missing_files.append(file_name)

            if (
                isinstance(width, (int, float))
                and isinstance(height, (int, float))
                and width > 0
                and height > 0
            ):
                image_sizes[image_id] = (float(width), float(height))
            else:
                invalid_images += 1

        invalid_annotations = 0
        unknown_image_references = 0
        unknown_category_references = 0
        boxes_outside_image = 0

        for annotation in self.annotations:
            image_id = annotation.get("image_id")
            category_id = annotation.get("category_id")
            bbox = annotation.get("bbox")

            if image_id not in valid_image_ids:
                unknown_image_references += 1
                invalid_annotations += 1
            if category_id not in valid_category_ids:
                unknown_category_references += 1
                invalid_annotations += 1
            elif isinstance(category_id, int):
                category_counts[category_id] += 1

            if (
                not isinstance(bbox, list)
                or len(bbox) != 4
                or not all(isinstance(value, (int, float)) for value in bbox)
                or bbox[2] <= 0
                or bbox[3] <= 0
            ):
                invalid_annotations += 1
                continue

            if image_id in image_sizes:
                image_width, image_height = image_sizes[image_id]
                x, y, width, height = map(float, bbox)
                if x < 0 or y < 0 or x + width > image_width or y + height > image_height:
                    boxes_outside_image += 1

        disk_images = {
            path.relative_to(self.directory).as_posix().lower()
            for path in self.directory.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        }
        orphan_files = sorted(disk_images - referenced_files)

        counts_by_name = {
            category_names[category_id]: category_counts.get(category_id, 0)
            for category_id in sorted(category_names)
        }

        return {
            "split": self.name,
            "directory": str(self.directory),
            "annotation_file": str(self.annotation_path),
            "images": len(self.images),
            "annotations": len(self.annotations),
            "categories": category_names,
            "annotations_by_category": counts_by_name,
            "images_without_annotations": sum(
                not annotations
                for annotations in self.annotations_by_image().values()
            ),
            "missing_image_files": missing_files,
            "orphan_image_files": orphan_files,
            "invalid_image_records": invalid_images,
            "invalid_annotations": invalid_annotations,
            "unknown_image_references": unknown_image_references,
            "unknown_category_references": unknown_category_references,
            "boxes_outside_image": boxes_outside_image,
            "duplicate_image_ids": _duplicate_values(image_ids),
            "duplicate_annotation_ids": _duplicate_values(annotation_ids),
            "duplicate_category_ids": _duplicate_values(category_ids),
        }


def inspect_dataset(
    dataset_root: str | Path,
    splits: Iterable[str] = ("train", "valid", "test"),
) -> dict[str, Any]:
    """Inspect all split directories that are present under a dataset root."""

    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    reports: list[dict[str, Any]] = []
    for split_name in splits:
        split_dir = root / split_name
        if split_dir.is_dir():
            reports.append(CocoSplit.load(split_dir).inspect())

    if not reports:
        raise FileNotFoundError(
            f"No train/valid/test split containing {ANNOTATION_FILE} found under {root}"
        )

    return {
        "dataset_root": str(root),
        "total_images": sum(report["images"] for report in reports),
        "total_annotations": sum(report["annotations"] for report in reports),
        "splits": reports,
    }

