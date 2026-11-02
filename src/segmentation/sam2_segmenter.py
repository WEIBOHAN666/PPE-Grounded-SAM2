"""Box-prompted image segmentation through the official SAM 2 API."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from src.schema import Detection, SegmentedInstance


class Sam2Segmenter:
    """Compute one SAM2 mask for every Grounding DINO detection."""

    def __init__(
        self,
        config_name: str = "configs/sam2.1/sam2.1_hiera_s.yaml",
        checkpoint_path: str | Path = "checkpoints/sam2.1_hiera_small.pt",
        *,
        device: str = "cuda",
        predictor: Any | None = None,
    ) -> None:
        self.config_name = config_name
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        self.device = device

        if predictor is None:
            if not self.checkpoint_path.is_file():
                raise FileNotFoundError(
                    f"SAM 2.1 checkpoint not found: {self.checkpoint_path}"
                )
            try:
                from sam2.build_sam import build_sam2
                from sam2.sam2_image_predictor import SAM2ImagePredictor
            except ImportError as exc:
                raise RuntimeError(
                    "SAM 2 is not installed. Install the official repository "
                    "with: pip install -e third_party/sam2"
                ) from exc

            model = build_sam2(
                self.config_name,
                str(self.checkpoint_path),
                device=self.device,
                mode="eval",
            )
            predictor = SAM2ImagePredictor(model)

        self._predictor = predictor

    def segment(
        self,
        image_rgb: NDArray[np.uint8],
        detections: Sequence[Detection],
    ) -> tuple[SegmentedInstance, ...]:
        image = np.asarray(image_rgb, dtype=np.uint8)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected an RGB HxWx3 image, got {image.shape}")
        if not detections:
            return ()

        self._predictor.set_image(image)
        instances: list[SegmentedInstance] = []
        for detection in detections:
            box = np.asarray(detection.box.as_list(), dtype=np.float32)
            masks, scores, _ = self._predictor.predict(
                box=box,
                multimask_output=False,
                return_logits=False,
            )

            masks_array = np.asarray(masks)
            scores_array = np.asarray(scores, dtype=np.float32).reshape(-1)
            if masks_array.ndim == 2:
                masks_array = masks_array[None, ...]
            if masks_array.ndim != 3 or masks_array.shape[0] == 0:
                raise RuntimeError(
                    f"SAM2 returned an invalid mask shape: {masks_array.shape}"
                )
            if scores_array.size != masks_array.shape[0]:
                raise RuntimeError("SAM2 returned mismatched masks and quality scores")

            best_index = int(np.argmax(scores_array))
            mask = np.asarray(masks_array[best_index] > 0, dtype=np.bool_)
            if mask.shape != image.shape[:2]:
                raise RuntimeError(
                    f"SAM2 mask shape {mask.shape} does not match image {image.shape[:2]}"
                )
            instances.append(
                SegmentedInstance(
                    detection=detection,
                    mask=mask,
                    mask_score=float(scores_array[best_index]),
                )
            )

        return tuple(instances)

