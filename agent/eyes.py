"""
Camera in, JPEG bytes out.

The camera runs in a background thread and keeps only the newest frame.
When the model asks "what do you see?", it gets *now*, not a frame from
three seconds ago sitting in a buffer.
"""
from __future__ import annotations

import glob
import os
import threading
import time

import cv2
import numpy as np

import config


class Eyes:
    """Live camera, or a still image / folder of images if you don't have one."""

    def __init__(self, source: str | int | None = None):
        self.source = config.CAMERA if source is None else source
        self._latest: np.ndarray | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._cap = None
        self._files: list[str] = []
        self._file_i = 0

        if isinstance(self.source, str) and not self.source.strip().isdigit():
            self._load_files(os.path.expanduser(self.source))
        else:
            self._open_camera(int(self.source))

    # -- setup ------------------------------------------------------------------
    def _load_files(self, path: str):
        if os.path.isdir(path):
            exts = ("*.jpg", "*.jpeg", "*.png", "*.webp")
            self._files = sorted(f for e in exts for f in glob.glob(os.path.join(path, e)))
        else:
            self._files = [path]
        if not self._files:
            raise FileNotFoundError(f"No images found at {path}")

    def _open_camera(self, index: int):
        # DirectShow opens much faster on Windows
        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
        self._cap = cv2.VideoCapture(index, backend)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Couldn't open camera {index}. Try AGENT_CAMERA=1, check OS camera "
                "permissions for your terminal / Jupyter, or set CAMERA to a photo path"
            )
        threading.Thread(target=self._pump, daemon=True).start()
        # wait for the first real frame (cameras need a moment to adjust exposure)
        t0 = time.time()
        while self._latest is None and time.time() - t0 < 5:
            time.sleep(0.05)
        time.sleep(0.3)

    def _pump(self):
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if ok:
                with self._lock:
                    self._latest = frame
            else:
                time.sleep(0.01)

    # -- use --------------------------------------------------------------------
    @property
    def is_live(self) -> bool:
        return self._cap is not None

    def frame(self) -> np.ndarray:
        """The newest BGR frame (a copy, so you can draw on it)."""
        if self._files:
            f = cv2.imread(self._files[self._file_i % len(self._files)])
            if f is None:
                raise RuntimeError(f"Couldn't read {self._files[self._file_i]}")
            return f
        with self._lock:
            if self._latest is None:
                raise RuntimeError("Camera opened but no frames yet.")
            return self._latest.copy()

    def next_file(self):
        """With a folder of images: move on to the next one."""
        self._file_i += 1

    def close(self):
        self._stop.set()
        if self._cap is not None:
            time.sleep(0.05)
            self._cap.release()


# ---------------------------------------------------------------------------
# preprocessing: the part that decides how fast (and how smart) the model is
# ---------------------------------------------------------------------------

def preprocess(
    frame: np.ndarray,
    max_side: int = config.IMAGE_MAX_SIDE,
    jpeg_quality: int = config.JPEG_QUALITY,
    grayscale: bool = config.GRAYSCALE,
    center_crop: float = 1.0,
) -> bytes:
    """
    frame        BGR image from OpenCV
    max_side     shrink so the longest edge is at most this many pixels
    jpeg_quality 1-100
    grayscale    drop colour
    center_crop  1.0 = full frame, 0.5 = middle half (a cheap digital zoom)
    """
    img = frame
    if center_crop < 1.0:
        h, w = img.shape[:2]
        ch, cw = int(h * center_crop), int(w * center_crop)
        y, x = (h - ch) // 2, (w - cw) // 2
        img = img[y : y + ch, x : x + cw]

    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    if grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()


def how_different(a: np.ndarray, b: np.ndarray) -> float:
    """0.0 = identical frames, ~1.0 = totally different. Cheap motion detector."""
    small_a = cv2.cvtColor(cv2.resize(a, (64, 48)), cv2.COLOR_BGR2GRAY).astype(np.float32)
    small_b = cv2.cvtColor(cv2.resize(b, (64, 48)), cv2.COLOR_BGR2GRAY).astype(np.float32)
    return float(np.mean(np.abs(small_a - small_b)) / 255.0)


def save(jpeg_bytes: bytes, path: str = "last_seen.jpg") -> str:
    with open(path, "wb") as f:
        f.write(jpeg_bytes)
    return path
