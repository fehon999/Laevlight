import re
from pathlib import Path
from typing import Iterable, List

import cv2
import numpy as np
import torch


class Visualization:
    """Save model predictions and optional input images by sequence.

    Example:
        visualizer = Visualization("outputs/visualization", save_keys=("gt",))
        visualizer.save_batch(raw_batch, pred)
    """

    _INVALID_NAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
    _FILE_SUFFIX = {
        "low": "low",
        "gt": "gt",
        "img_refer": "refer",
    }

    def __init__(self, output_dir, save_keys: Iterable[str] = ()):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_keys = tuple(save_keys)
        self.count = 0

    def __call__(self, batch, pred: torch.Tensor) -> List[Path]:
        return self.save_batch(batch, pred)

    @classmethod
    def _safe_name(cls, value, fallback: str) -> str:
        name = cls._INVALID_NAME.sub("_", str(value)).strip().strip(".")
        return name if name else fallback

    @staticmethod
    def _value_at(values, index: int, default):
        if values is None:
            return default
        if isinstance(values, (list, tuple)):
            return values[index]
        if torch.is_tensor(values) and values.ndim > 0:
            value = values[index]
            return value.item() if value.numel() == 1 else value
        return values

    @staticmethod
    def _to_bgr_image(image: torch.Tensor) -> np.ndarray:
        if not torch.is_tensor(image):
            raise TypeError(f"Expected a torch.Tensor image, got {type(image).__name__}.")
        if image.ndim != 3:
            raise ValueError(f"Expected a [C, H, W] image, got shape {tuple(image.shape)}.")

        image = image.detach().float().clamp(0, 1).cpu()
        channels = image.shape[0]
        if channels == 1:
            image = image.repeat(3, 1, 1)
        elif channels != 3:
            raise ValueError(f"Only 1-channel or 3-channel images can be saved, got {channels}.")

        image = image.permute(1, 2, 0).numpy()
        image = (image * 255.0).round().astype(np.uint8)
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    @classmethod
    def _write_image(cls, image: torch.Tensor, path: Path) -> None:
        if not cv2.imwrite(str(path), cls._to_bgr_image(image)):
            raise OSError(f"Failed to save visualization image: {path}")

    def save_batch(self, batch, pred: torch.Tensor) -> List[Path]:
        """Save a [B, C, H, W] prediction batch and configured batch images."""
        if pred.ndim == 3:
            pred = pred.unsqueeze(0)
        if pred.ndim != 4:
            raise ValueError(f"Expected pred with shape [B, C, H, W], got {tuple(pred.shape)}.")

        seq_names = batch.get("seq_name")
        frame_ids = batch.get("frame_id")
        saved_paths = []

        for index in range(pred.shape[0]):
            fallback_frame = f"{self.count:06d}"
            seq_name = self._safe_name(
                self._value_at(seq_names, index, "sample"),
                "sample",
            )
            frame_name = self._safe_name(
                self._value_at(frame_ids, index, fallback_frame),
                fallback_frame,
            )
            sequence_dir = self.output_dir / seq_name
            sequence_dir.mkdir(parents=True, exist_ok=True)

            pred_path = sequence_dir / f"{frame_name}_pred.png"
            self._write_image(pred[index], pred_path)
            saved_paths.append(pred_path)

            for key in self.save_keys:
                images = batch.get(key)
                if images is None:
                    continue
                if images.ndim == 3:
                    images = images.unsqueeze(0)
                suffix = self._FILE_SUFFIX.get(key, self._safe_name(key, "image"))
                image_path = sequence_dir / f"{frame_name}_{suffix}.png"
                self._write_image(images[index], image_path)
                saved_paths.append(image_path)

            self.count += 1

        return saved_paths
