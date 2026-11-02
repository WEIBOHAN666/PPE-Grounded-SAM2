from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.detection.grounding_dino import GroundingDinoDetector, normalize_phrase
from src.export import export_prediction
from src.pipeline import GroundedSam2Pipeline
from src.schema import BoxXYXY, Detection, DetectionBatch
from src.segmentation import Sam2Segmenter


class FakeGroundingDinoApi:
    @staticmethod
    def load_model(config: str, checkpoint: str, device: str):
        return object()

    @staticmethod
    def load_image(path: str):
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        return image, "fake-tensor"

    @staticmethod
    def predict(**kwargs):
        boxes = np.asarray([[0.5, 0.5, 0.4, 0.2]], dtype=np.float32)
        scores = np.asarray([0.9], dtype=np.float32)
        return boxes, scores, ["helmet"]


class FakeSam2Predictor:
    def set_image(self, image: np.ndarray) -> None:
        self.image_shape = image.shape[:2]

    def predict(self, *, box, multimask_output, return_logits):
        height, width = self.image_shape
        mask = np.zeros((1, height, width), dtype=bool)
        x1, y1, x2, y2 = np.asarray(box, dtype=int)
        mask[0, y1:y2, x1:x2] = True
        return mask, np.asarray([0.88], dtype=np.float32), np.zeros((1, 256, 256))


class ModelAdapterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.image_path = self.root / "sample.jpg"
        Image.fromarray(np.zeros((100, 200, 3), dtype=np.uint8)).save(self.image_path)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_normalized_box_conversion(self) -> None:
        box = BoxXYXY.from_normalized_cxcywh([0.5, 0.5, 0.4, 0.2], 200, 100)
        self.assertEqual(box.as_list(), [60.0, 40.0, 140.0, 60.0])

    def test_phrase_normalization(self) -> None:
        self.assertEqual(normalize_phrase("a helmet"), "helmet")
        self.assertEqual(normalize_phrase("person"), "person")

    def test_grounding_dino_adapter(self) -> None:
        detector = GroundingDinoDetector(
            "unused-config.py",
            "unused-checkpoint.pth",
            api=FakeGroundingDinoApi,
        )
        batch = detector.detect(self.image_path)
        self.assertEqual(len(batch.detections), 1)
        self.assertEqual(batch.detections[0].label, "helmet")
        self.assertAlmostEqual(batch.detections[0].score, 0.9, places=5)
        for actual, expected in zip(
            batch.detections[0].box.as_list(), [60.0, 40.0, 140.0, 60.0]
        ):
            self.assertAlmostEqual(actual, expected, places=5)

    def test_sam2_adapter_and_end_to_end_export(self) -> None:
        detector = GroundingDinoDetector(
            "unused-config.py",
            "unused-checkpoint.pth",
            api=FakeGroundingDinoApi,
        )
        segmenter = Sam2Segmenter(predictor=FakeSam2Predictor())
        result = GroundedSam2Pipeline(detector, segmenter).predict(self.image_path)

        self.assertEqual(len(result.instances), 1)
        self.assertEqual(result.instances[0].mask.shape, (100, 200))
        self.assertEqual(int(result.instances[0].mask.sum()), 1600)

        outputs = export_prediction(result, self.root / "output")
        self.assertTrue(outputs["overlay"].is_file())
        self.assertTrue((outputs["masks"] / "001_helmet.png").is_file())
        metadata = json.loads(outputs["json"].read_text(encoding="utf-8"))
        self.assertEqual(metadata["instance_count"], 1)
        self.assertEqual(metadata["instances"][0]["label"], "helmet")


if __name__ == "__main__":
    unittest.main()
