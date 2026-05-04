"""Extract per-frame mean RGB signal from a video.

Saves a NumPy .npz file with arrays:
    t   - timestamps in seconds, shape (N,)
    rgb - mean RGB per frame, shape (N, 3)
    fps - nominal frames per second of the source video
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np


def extract(video_path: Path, out_path: Path) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {video_path.name}  fps={fps:.3f}  frames={n_total}")

    means = []
    timestamps = []
    t0 = time.time()
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # OpenCV gives BGR. Convert to RGB-ordered mean.
        # Use float64 mean for numerical stability over many frames.
        b = float(frame[:, :, 0].mean())
        g = float(frame[:, :, 1].mean())
        r = float(frame[:, :, 2].mean())
        means.append((r, g, b))
        # Per-frame timestamp from the container (ms -> s).
        ts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        timestamps.append(ts_ms / 1000.0)
        i += 1
        if i % 100 == 0:
            elapsed = time.time() - t0
            print(f"  frame {i}/{n_total}  ({i/elapsed:.1f} fps)")

    cap.release()
    rgb = np.asarray(means, dtype=np.float64)
    t = np.asarray(timestamps, dtype=np.float64)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, t=t, rgb=rgb, fps=fps)
    print(f"Saved {out_path}  shape={rgb.shape}")


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    video = Path(sys.argv[1]) if len(sys.argv) > 1 else here.parent / "data" / "IMG_1971.MOV"
    out = here / "working" / "signal.npz"
    extract(video, out)
