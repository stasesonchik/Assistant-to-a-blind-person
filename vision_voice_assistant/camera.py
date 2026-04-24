from __future__ import annotations

import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class CameraError(RuntimeError):
    pass


@dataclass
class CameraSource:
    source: str | int
    width: int | None = None
    height: int | None = None

    def __post_init__(self) -> None:
        self._capture = None
        if isinstance(self.source, str) and self.source.isdigit():
            self.source = int(self.source)

    @property
    def is_snapshot_url(self) -> bool:
        if not isinstance(self.source, str):
            return False
        lowered = self.source.lower()
        return lowered.endswith((".jpg", ".jpeg", ".png")) or "shot" in lowered or "snapshot" in lowered

    def open(self) -> None:
        if self.is_snapshot_url:
            return
        cv2 = import_cv2()
        self._capture = cv2.VideoCapture(self.source)
        if self.width:
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height:
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if not self._capture.isOpened():
            raise CameraError(f"Cannot open camera source: {self.source}")

    def read(self):
        if self.is_snapshot_url:
            return self._read_snapshot_url()
        if self._capture is None:
            self.open()
        ok, frame = self._capture.read()
        if not ok or frame is None:
            raise CameraError("Cannot read frame from camera")
        return frame

    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def _read_snapshot_url(self):
        cv2 = import_cv2()
        with urllib.request.urlopen(str(self.source), timeout=10) as response:
            data = np.frombuffer(response.read(), dtype=np.uint8)
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if frame is None:
            raise CameraError(f"Snapshot URL did not return an image: {self.source}")
        return frame


def import_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise CameraError("Install OpenCV first: python -m pip install opencv-python") from exc
    return cv2


def save_frame(frame, captures_dir: str | Path, prefix: str = "capture") -> Path:
    cv2 = import_cv2()
    directory = Path(captures_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
    ok = cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise CameraError(f"Cannot save frame to {path}")
    return path

