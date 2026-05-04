"""HSV experiment: re-extract per-frame mean H, S, V (computed per pixel
*then* averaged) inside the same forehead+cheeks ROI, alongside the RGB mean.

Saves working/signal_hsv.npz with arrays:
    t        - timestamps (s), shape (N,)
    hsv_roi  - mean (H, S, V) inside the ROI per frame, shape (N, 3)
                  H in degrees [0, 360), S in [0,1], V in [0,1]
    rgb_roi  - mean (R, G, B) inside the ROI per frame, shape (N, 3)
    fps      - nominal source FPS

Why per-pixel-then-average: HSV is a nonlinear function of RGB, so taking the
mean of the HSV values is *different* from converting the mean RGB to HSV.
The honest comparison for the question \"does HSV give a cleaner signal?\" is
the per-pixel mean.

Hue is circular: we do a circular mean using sin/cos to avoid wraparound bias.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh

FOREHEAD_IDS = [67, 109, 10, 338, 297, 332, 333, 334, 296, 336, 9, 107, 66, 105, 104, 103]
LEFT_CHEEK_IDS = [116, 117, 118, 119, 100, 142, 203, 206, 207, 187, 123]
RIGHT_CHEEK_IDS = [345, 346, 347, 348, 329, 371, 423, 426, 427, 411, 352]


def landmarks_to_xy(landmarks, ids: list[int], w: int, h: int) -> np.ndarray:
    return np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in ids], dtype=np.int32)


def build_mask(frame_shape: tuple[int, int], polygons: list[np.ndarray]) -> np.ndarray:
    h, w = frame_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for poly in polygons:
        cv2.fillPoly(mask, [poly], 255)
    return mask


def hsv_circular_mean(h_deg: np.ndarray, s: np.ndarray, v: np.ndarray) -> tuple[float, float, float]:
    # Circular mean for hue (in degrees). We weight by saturation*value so very
    # dark / unsaturated pixels (where hue is meaningless) don't pull the mean.
    weight = s * v + 1e-6
    rad = np.deg2rad(h_deg)
    cos_mean = np.average(np.cos(rad), weights=weight)
    sin_mean = np.average(np.sin(rad), weights=weight)
    h_mean = (np.rad2deg(np.arctan2(sin_mean, cos_mean)) + 360.0) % 360.0
    return float(h_mean), float(s.mean()), float(v.mean())


def extract(video_path: Path, out_path: Path) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {video_path.name}  fps={fps:.3f}  frames={n_total}")

    hsv_roi: list[tuple[float, float, float]] = []
    rgb_roi: list[tuple[float, float, float]] = []
    timestamps: list[float] = []
    n_lost = 0
    last_mask: np.ndarray | None = None

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as mesh:
        t0 = time.time()
        i = 0
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            h, w = frame_bgr.shape[:2]
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            results = mesh.process(frame_rgb)

            mask: np.ndarray | None = None
            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark
                forehead = landmarks_to_xy(lm, FOREHEAD_IDS, w, h)
                left = landmarks_to_xy(lm, LEFT_CHEEK_IDS, w, h)
                right = landmarks_to_xy(lm, RIGHT_CHEEK_IDS, w, h)
                mask = build_mask(frame_bgr.shape, [forehead, left, right])
                last_mask = mask
            else:
                mask = last_mask
                n_lost += 1

            ts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            timestamps.append(ts_ms / 1000.0)

            if mask is None:
                hsv_roi.append((np.nan, np.nan, np.nan))
                rgb_roi.append((np.nan, np.nan, np.nan))
                i += 1
                continue

            sel = mask > 0
            R = frame_rgb[:, :, 0][sel].astype(np.float32)
            G = frame_rgb[:, :, 1][sel].astype(np.float32)
            B = frame_rgb[:, :, 2][sel].astype(np.float32)

            # Per-pixel HSV via OpenCV (full hue range 0-360).
            # We need the masked pixels; convert just those by building a 1xN
            # image. cv2.cvtColor wants uint8 BGR or float32 BGR with H in [0,360].
            pixels_bgr = np.stack([B, G, R], axis=-1).astype(np.float32) / 255.0
            pixels_bgr = pixels_bgr.reshape(-1, 1, 3)
            pixels_hsv = cv2.cvtColor(pixels_bgr, cv2.COLOR_BGR2HSV).reshape(-1, 3)
            H_deg = pixels_hsv[:, 0]   # 0-360
            S = pixels_hsv[:, 1]       # 0-1
            V = pixels_hsv[:, 2]       # 0-1

            hsv_roi.append(hsv_circular_mean(H_deg, S, V))
            rgb_roi.append((float(R.mean()), float(G.mean()), float(B.mean())))

            i += 1
            if i % 200 == 0:
                elapsed = time.time() - t0
                print(f"  frame {i}/{n_total}  ({i/elapsed:.1f} fps)  lost={n_lost}")

    cap.release()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        t=np.asarray(timestamps, dtype=np.float64),
        hsv_roi=np.asarray(hsv_roi, dtype=np.float64),
        rgb_roi=np.asarray(rgb_roi, dtype=np.float64),
        fps=fps,
    )
    print(f"Saved {out_path}  N={len(timestamps)}  lost={n_lost}")


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    video = Path(sys.argv[1]) if len(sys.argv) > 1 else here.parent / "data" / "IMG_1971.MOV"
    out = here / "working" / "signal_hsv.npz"
    extract(video, out)
