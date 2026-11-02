from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from scripts.check_prediction_association import run_association_check
from src.association import associate_prediction
from src.export import export_prediction
from src.schema import BoxXYXY, Detection, PredictionResult, SegmentedInstance


def instance(instance_id, label, box, score=0.9):
    return {"instance_id": instance_id, "label": label,
            "detection_score": score, "box_xyxy": box}


def prediction(instances):
    return {"image": "sample.png", "width": 640, "height": 480, "instances": instances}


class PredictionAssociationTest(unittest.TestCase):
    def test_original_ids_and_interleaved_classes(self):
        payload = prediction([
            instance(99, "helmet", [25, 10, 55, 30]),
            instance(7, "head", [20, 10, 60, 50]),
            instance(42, "person", [0, 0, 100, 200]),
            instance(8, "person", [120, 0, 220, 200]),
        ])
        self.assertEqual(associate_prediction(payload)["associations"], [
            {"person_id": 42, "head_id": 7, "helmet_id": 99},
            {"person_id": 8, "head_id": None, "helmet_id": None},
        ])

    def test_low_score_person_cannot_consume_a_head(self):
        payload = prediction([
            instance(5, "person", [0, 0, 100, 200], 0.3),
            instance(42, "person", [0, 0, 100, 200]),
            instance(7, "head", [20, 10, 60, 50]),
            instance(99, "helmet", [25, 10, 55, 30]),
        ])
        result = associate_prediction(payload)
        self.assertEqual(result["associations"], [{"person_id": 42, "head_id": 7, "helmet_id": 99}])
        self.assertEqual(len(result["instances"]), 4)

    def test_threshold_inclusive_and_zero_id(self):
        payload = prediction([
            instance(0, "person", [0, 0, 100, 200], 0.6),
            instance(10, "head", [20, 10, 60, 50], 0.6),
            instance(20, "helmet", [25, 10, 55, 30], 0.5999),
            instance(30, "helmet", [25, 10, 55, 30], 0.6),
        ])
        self.assertEqual(associate_prediction(payload)["associations"],
                         [{"person_id": 0, "head_id": 10, "helmet_id": 30}])

    def test_low_head_becomes_missing_and_orphans_are_ignored(self):
        payload = prediction([
            instance(5, "person", [0, 0, 100, 200]),
            instance(6, "head", [20, 10, 60, 50], 0.2),
            instance(7, "helmet", [25, 10, 55, 30]),
        ])
        self.assertEqual(associate_prediction(payload)["associations"],
                         [{"person_id": 5, "head_id": None, "helmet_id": None}])
        self.assertEqual(associate_prediction(prediction([]))["associations"], [])

    def test_preserve_metadata_without_mutating_input(self):
        payload = prediction([instance(9, "person", [0, 0, 100, 200])])
        payload["instances"][0].update(raw_phrase="a person", mask_file="masks/009_person.png")
        payload["associations"] = [{"person_id": -1}]
        before = deepcopy(payload)
        output = associate_prediction(payload)
        self.assertEqual(payload, before)
        self.assertEqual(output["instances"], payload["instances"])
        self.assertEqual(output["associations"], [{"person_id": 9, "head_id": None, "helmet_id": None}])
        output["instances"][0]["label"] = "head"
        self.assertEqual(payload["instances"][0]["label"], "person")

    def test_reject_duplicate_ids(self):
        with self.assertRaisesRegex(ValueError, "Duplicate instance_id"):
            associate_prediction(prediction([
                instance(1, "person", [0, 0, 100, 200]),
                instance(1, "head", [20, 10, 60, 50]),
            ]))

    def test_reject_invalid_exported_values(self):
        for field, value in [("detection_score", float("nan")),
                             ("detection_score", True),
                             ("detection_score", 1.1),
                             ("label", "unknown"),
                             ("box_xyxy", [0, 0, 0, 20]),
                             ("box_xyxy", [0, 0, 700, 20])]:
            with self.subTest(field=field, value=value):
                record = instance(1, "person", [0, 0, 100, 200])
                record[field] = value
                with self.assertRaises(ValueError):
                    associate_prediction(prediction([record]))

    def test_accept_actual_exporter_schema(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rgb = np.zeros((480, 640, 3), dtype=np.uint8)
            records = [
                Detection("helmet", 0.9, BoxXYXY(25, 10, 55, 30), "a helmet"),
                Detection("person", 0.9, BoxXYXY(0, 0, 100, 200), "person"),
                Detection("head", 0.9, BoxXYXY(20, 10, 60, 50), "head"),
            ]
            result = PredictionResult(root / "sample.png", rgb, "person. head. helmet.", tuple(
                SegmentedInstance(d, np.zeros((480, 640), dtype=bool), 0.8) for d in records
            ))
            paths = export_prediction(result, root / "export")
            payload = json.loads(paths["json"].read_text(encoding="utf-8"))
            output = associate_prediction(payload)
            self.assertEqual(output["associations"], [{"person_id": 2, "head_id": 3, "helmet_id": 1}])
            self.assertEqual(output["instances"][0]["mask_file"], payload["instances"][0]["mask_file"])

    def test_full_batch_uses_original_review_tool(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            predictions = []
            for name, records in [
                ("needs_review", [instance(42, "person", [20, 20, 250, 450])]),
                ("complete", [instance(77, "helmet", [25, 10, 55, 30]),
                              instance(99, "person", [0, 0, 100, 200], 0.2),
                              instance(0, "person", [0, 0, 100, 200]),
                              instance(9, "head", [20, 10, 60, 50])]),
            ]:
                image = root / f"{name}.png"
                Image.new("RGB", (640, 480), "white").save(image)
                payload = prediction(records)
                payload.update(image=str(image), data_source="synthetic_test")
                path = root / f"{name}.json"
                path.write_text(json.dumps(payload), encoding="utf-8")
                predictions.append(path)
            before = {p: p.read_bytes() for p in root.iterdir() if p.is_file()}
            summary = run_association_check(predictions, root / "check")
            report = json.loads(Path(summary["report"]).read_text(encoding="utf-8"))
            self.assertEqual(report["counts"], {"person": 2, "head": 1, "helmet": 1})
            self.assertEqual(report["errors"], [])
            self.assertEqual(report["reviews"], [{"person_id": 42, "reason": "head_unmatched",
                                                  "review_status": "pending", "image": "needs_review.png"}])
            with Image.open(root / "check/output/images/needs_review.png") as marked:
                self.assertEqual(marked.size, (640, 480))
                self.assertEqual(marked.getpixel((20, 200)), (255, 0, 0))
            self.assertEqual(len(list((root / "check/output/images").iterdir())), 2)
            for path, content in before.items():
                self.assertEqual(path.read_bytes(), content)

    def test_cloud_image_path_can_be_overridden(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "local.png"
            Image.new("RGB", (640, 480), "white").save(image)
            payload = prediction([])
            payload["image"] = "/cloud/server/photo.png"
            path = root / "prediction.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            summary = run_association_check([path], root / "check", image)
            self.assertEqual(summary["error_count"], 0)
            with Image.open(root / "check/output/images/local.png") as marked:
                self.assertEqual(marked.size, (640, 480))


if __name__ == "__main__":
    unittest.main()
