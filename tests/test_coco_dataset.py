from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.datasets.coco_dataset import CocoSplit, inspect_dataset


class CocoDatasetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.train = self.root / "train"
        self.train.mkdir()
        (self.train / "sample.jpg").write_bytes(b"test")
        payload = {
            "images": [
                {
                    "id": 1,
                    "file_name": "sample.jpg",
                    "width": 100,
                    "height": 80,
                }
            ],
            "categories": [{"id": 1, "name": "helmet"}],
            "annotations": [
                {
                    "id": 10,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [10, 12, 30, 20],
                }
            ],
        }
        (self.train / "_annotations.coco.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_load_and_inspect_valid_split(self) -> None:
        report = CocoSplit.load(self.train).inspect()
        self.assertEqual(report["images"], 1)
        self.assertEqual(report["annotations"], 1)
        self.assertEqual(report["annotations_by_category"], {"helmet": 1})
        self.assertEqual(report["missing_image_files"], [])
        self.assertEqual(report["invalid_annotations"], 0)

    def test_inspect_dataset_discovers_existing_splits(self) -> None:
        report = inspect_dataset(self.root)
        self.assertEqual(report["total_images"], 1)
        self.assertEqual([item["split"] for item in report["splits"]], ["train"])
if __name__ == "__main__":
    unittest.main()

