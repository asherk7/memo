"""Face preprocessing: MediaPipe detect → eye-line align → center crop → 112×112.

Input is an RGB image (the rest of the pipeline assumes RGB; any BGR→RGB
conversion belongs at the OpenCV load boundary, not here). Output is a
`(3, size, size)` float32 tensor in [0, 1]. Model-specific normalization
(ImageNet mean/std) is the encoder's job (Phase 3), not preprocessing's.

This module is stateless and deterministic. The only randomness in the face
path lives in `augment/image.py`.
"""

from __future__ import annotations

import math
import os
import urllib.request
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

__all__ = ["FaceNotFoundError", "preprocess_face"]


class FaceNotFoundError(Exception):
    """Raised when MediaPipe finds no face in a (non-None) input image.

    The pipeline catches this to silently degrade the face modality (§1);
    preprocessing itself just raises.
    """


# BlazeFace short-range keypoint indices (Tasks API order).
_RIGHT_EYE = 0
_LEFT_EYE = 1

_DEFAULT_SIZE = 112
_CROP_MARGIN = 1.4

# MediaPipe Tasks API needs an explicit model asset. Cache it on first use;
# override the location with MEMO_FACE_MODEL for offline/CI environments.
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)

# Lazily-created module-level detector — MediaPipe graph construction is the
# expensive part, so we build it once and reuse it across calls.
_detector: Any = None


def _model_path() -> str:
    override = os.environ.get("MEMO_FACE_MODEL")
    if override:
        return override
    cache_root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "memo"
    cache_root.mkdir(parents=True, exist_ok=True)
    dst = cache_root / "blaze_face_short_range.tflite"
    if not dst.exists():
        urllib.request.urlretrieve(_MODEL_URL, dst)  # noqa: S310
    return str(dst)


def _get_detector() -> Any:
    global _detector
    if _detector is None:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        options = vision.FaceDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=_model_path()),
            min_detection_confidence=0.5,
        )
        _detector = vision.FaceDetector.create_from_options(options)
    return _detector


def _to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected an RGB image of shape (H, W, 3), got {arr.shape}.")
    if arr.dtype == np.uint8:
        return np.ascontiguousarray(arr)
    # Float images follow the [0, 1] convention the rest of the pipeline uses.
    if np.issubdtype(arr.dtype, np.floating):
        arr = arr * 255.0
    return np.ascontiguousarray(np.clip(arr, 0, 255).astype(np.uint8))


def _align_crop_resize(
    image: np.ndarray,
    right_eye: tuple[float, float],
    left_eye: tuple[float, float],
    bbox: tuple[float, float, float, float],
    size: int = _DEFAULT_SIZE,
    margin: float = _CROP_MARGIN,
) -> torch.Tensor:
    """Rotate so the eye line is horizontal, crop a square around the face,
    and resize to `size`. Coordinates are in pixels; `bbox` is (x, y, w, h)."""
    h, w = image.shape[:2]
    rx, ry = right_eye
    lx, ly = left_eye

    angle = math.degrees(math.atan2(ly - ry, lx - rx))
    eye_center = ((rx + lx) / 2.0, (ry + ly) / 2.0)

    rot = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
    rotated = cv2.warpAffine(
        image, rot, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )

    bx, by, bw, bh = bbox
    cx, cy = bx + bw / 2.0, by + bh / 2.0
    tx = rot[0, 0] * cx + rot[0, 1] * cy + rot[0, 2]
    ty = rot[1, 0] * cx + rot[1, 1] * cy + rot[1, 2]

    half = margin * max(bw, bh) / 2.0
    x0 = max(0, int(round(tx - half)))
    y0 = max(0, int(round(ty - half)))
    x1 = min(w, int(round(tx + half)))
    y1 = min(h, int(round(ty + half)))

    crop = rotated[y0:y1, x0:x1]
    if crop.size == 0:
        crop = rotated

    resized = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    arr = resized.astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def preprocess_face(image: np.ndarray, size: int = _DEFAULT_SIZE) -> torch.Tensor:
    """Detect, align, crop, and resize a face to a `(3, size, size)` tensor.

    Raises `FaceNotFoundError` if no face is detected.
    """
    import mediapipe as mp

    img = _to_uint8_rgb(image)
    h, w = img.shape[:2]

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
    detections = _get_detector().detect(mp_image).detections
    if not detections:
        raise FaceNotFoundError("No face detected in the input image.")

    det = detections[0]
    kps = det.keypoints  # normalized [0, 1]
    right_eye = (kps[_RIGHT_EYE].x * w, kps[_RIGHT_EYE].y * h)
    left_eye = (kps[_LEFT_EYE].x * w, kps[_LEFT_EYE].y * h)

    bb = det.bounding_box  # pixels
    bbox = (float(bb.origin_x), float(bb.origin_y), float(bb.width), float(bb.height))

    return _align_crop_resize(img, right_eye, left_eye, bbox, size=size)
