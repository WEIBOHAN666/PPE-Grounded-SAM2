"""Environment diagnostics that do not import heavy libraries eagerly."""

from __future__ import annotations

import importlib.util
import platform
import sys
from typing import Any


def environment_report() -> dict[str, Any]:
    packages = {
        name: importlib.util.find_spec(name) is not None
        for name in ("numpy", "PIL", "torch", "torchvision", "groundingdino", "sam2")
    }
    cuda_available: bool | None = None
    cuda_device: str | None = None
    if packages["torch"]:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            cuda_device = torch.cuda.get_device_name(0)

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": packages,
        "cuda_available": cuda_available,
        "cuda_device": cuda_device,
        "model_runtime_ready": all(
            packages[name] for name in ("torch", "torchvision", "groundingdino", "sam2")
        ),
    }

