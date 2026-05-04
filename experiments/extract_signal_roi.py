"""Extract per-frame mean RGB inside a face ROI (forehead + cheeks) using
MediaPipe FaceMesh.

Saves working/signal_roi.npz with arrays:
    t          - timestamps (seconds), shape (N,)
    rgb_roi    - mean RGB inside the ROI per frame, shape (N, 3)
    rgb_full   - mean RGB of the whole frame per frame, shape (N, 3) (for comparison)
    roi_area   - ROI area in pixels per frame (N,) (for diagnostics)
    fps        - nominal source FPS

Also saves working/roi_preview.jpg showing the ROI overlay on a sample frame.

FaceMesh landmark indices (canonical MediaPipe topology) for our patches:
    forehead polygon: a band above the eyebrows, below the hairline
    left cheek polygon:  centred under the left eye
    right cheek polygon: centred under the right eye
We avoid eyes, eyebrows, nostrils, lips, and hair.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh

# Landmark indices for the ROI polygons (chosen to lie on smooth skin).
# Reference: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
FOREHEAD_IDS = [67, 109, 10, 338, 297, 332, 333, 334, 296, 336, 9, 107, 66, 105, 104, 103]
LEFT_CHEEK_IDS = [116, 117, 118, 119, 100, 142, 203, 206, 207, 187, 123]
RIGHT_CHEEK_IDS = [345, 346, 347, 348, 329, 371, 423, 426, 427, 411, 352]


def landmarks_to_xy(landmarks, ids: list[int], w: int, h: int) -> np.ndarray:
    # Tasks API: landmarks is a list of NormalizedLandmark with .x .y .z
    return np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in ids], dtype=np.int32)


def build_mask(frame_shape: tuple[int, int], polygons: list[np.ndarray]) -> np.ndarray:
    h, w = frame_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for poly in polygons:
        cv2.fillPoly(mask, [poly], 255)
    return mask


def extract(video_path: Path, out_path: Path, preview_path: Path) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {video_path.name}  fps={fps:.3f}  frames={n_total}")

    rgb_roi: list[tuple[float, float, float]] = []
    rgb_full: list[tuple[float, float, float]] = []
    timestamps: list[float] = []
    roi_areas: list[int] = []
    n_lost = 0
    last_mask: np.ndarray | None = None
    preview_saved = False

    options = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    with options as mesh:
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
            forehead = left = right = None
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

            rgb_full.append((
                float(frame_rgb[:, :, 0].mean()),
                float(frame_rgb[:, :, 1].mean()),
                float(frame_rgb[:, :, 2].mean()),
            ))

            if mask is not None:
                area = int(mask.sum() // 255)
                roi_areas.append(area)
                if area > 0:
                    sel = mask > 0
                    rgb_roi.append((
                        float(frame_rgb[:, :, 0][sel].mean()),
                        float(frame_rgb[:, :, 1][sel].mean()),
                        float(frame_rgb[:, :, 2][sel].mean()),
                    ))
                else:
                    rgb_roi.append((np.nan, np.nan, np.nan))
            else:
                roi_areas.append(0)
                rgb_roi.append((np.nan, np.nan, np.nan))

            if (
                not preview_saved
                and ts_ms / 1000.0 > 20.0
                and mask is not None
                and forehead is not None
            ):
                preview = frame_bgr.copy()
                overlay = frame_bgr.copy()
                overlay[mask > 0] = (0, 255, 255)
                preview = cv2.addWeighted(overlay, 0.35, preview, 0.65, 0)
                cv2.polylines(preview, [forehead, left, right], True, (0, 255, 0), 2)
                cv2.imwrite(str(preview_path), preview)
                preview_saved = True
                print(f"  preview -> {preview_path}")

            i += 1
            if i % 200 == 0:
                elapsed = time.time() - t0
                print(f"  frame {i}/{n_total}  ({i/elapsed:.1f} fps)  lost={n_lost}")

    cap.release()
    rgb_roi_arr = np.asarray(rgb_roi, dtype=np.float64)
    rgb_full_arr = np.asarray(rgb_full, dtype=np.float64)
    t = np.asarray(timestamps, dtype=np.float64)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        t=t,
        rgb_roi=rgb_roi_arr,
        rgb_full=rgb_full_arr,
        roi_area=np.asarray(roi_areas, dtype=np.int64),
        fps=fps,
    )
    pct_lost = 100.0 * n_lost / max(1, len(t))
    print(f"Saved {out_path}  N={len(t)}  detection-lost frames: {n_lost} ({pct_lost:.1f}%)")


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    video = Path(sys.argv[1]) if len(sys.argv) > 1 else here.parent / "data" / "IMG_1971.MOV"
    out = here / "working" / "signal_roi.npz"
    preview = here / "working" / "roi_preview.jpg"
    extract(video, out, preview)
