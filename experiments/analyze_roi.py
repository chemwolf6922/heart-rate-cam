"""Compare ROI vs whole-frame heart-rate estimation on the test video.

Reads working/signal_roi.npz produced by extract_signal_roi.py and runs the
same chrominance pipeline (G/<G> - R/<R>, bandpass, FFT/Welch peak in
0.7-3 Hz) on:
    - whole-frame RGB
    - ROI RGB (forehead + cheeks via MediaPipe)

Ground truth: 69-74 BPM.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, detrend, filtfilt, welch

HR_LOW_HZ = 0.7
HR_HIGH_HZ = 3.0


def resample_uniform(t: np.ndarray, x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    t0, t1 = t[0], t[-1]
    n = int(np.floor((t1 - t0) * fs)) + 1
    tu = t0 + np.arange(n) / fs
    if x.ndim == 1:
        return tu, np.interp(tu, t, x)
    out = np.empty((n, x.shape[1]))
    for c in range(x.shape[1]):
        out[:, c] = np.interp(tu, t, x[:, c])
    return tu, out


def bandpass(x: np.ndarray, fs: float, lo: float, hi: float, order: int = 4) -> np.ndarray:
    nyq = fs / 2
    b, a = butter(order, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, x, axis=0)


def fft_peak_bpm(x: np.ndarray, fs: float) -> tuple[float, np.ndarray, np.ndarray]:
    n = len(x)
    nfft = max(1 << 14, 1 << (int(np.ceil(np.log2(n))) + 2))
    spec = np.fft.rfft(x * np.hanning(n), n=nfft)
    freqs = np.fft.rfftfreq(nfft, d=1 / fs)
    mag = np.abs(spec)
    band = (freqs >= HR_LOW_HZ) & (freqs <= HR_HIGH_HZ)
    return float(freqs[np.argmax(mag * band)] * 60), freqs, mag


def welch_peak_bpm(x: np.ndarray, fs: float) -> tuple[float, np.ndarray, np.ndarray]:
    nperseg = min(len(x), int(fs * 12))
    f, p = welch(x, fs=fs, nperseg=nperseg)
    band = (f >= HR_LOW_HZ) & (f <= HR_HIGH_HZ)
    return float(f[band][np.argmax(p[band])] * 60), f, p


def chrominance_signals(rgb: np.ndarray) -> dict[str, np.ndarray]:
    R, G, B = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    Rn = R / R.mean() - 1
    Gn = G / G.mean() - 1
    Bn = B / B.mean() - 1
    return {
        "G":   Gn,
        "G-R": Gn - Rn,
        "POS-X (3R-2G)": 3 * Rn - 2 * Gn,
    }


def evaluate(label: str, t: np.ndarray, rgb: np.ndarray,
             trim_start_s: float, trim_end_s: float):
    # Sort/dedup, trim, resample.
    order = np.argsort(t)
    t = t[order]
    rgb = rgb[order]
    keep = np.concatenate([[True], np.diff(t) > 0])
    t = t[keep]
    rgb = rgb[keep]

    # Drop NaNs (frames where no face was found).
    nan_mask = np.isnan(rgb).any(axis=1)
    if nan_mask.any():
        print(f"  {label}: dropping {nan_mask.sum()} NaN frames")
    t = t[~nan_mask]
    rgb = rgb[~nan_mask]

    fs = int(round((len(t) - 1) / (t[-1] - t[0])))
    tu, rgbu = resample_uniform(t, rgb, fs)
    mask = (tu >= tu[0] + trim_start_s) & (tu <= tu[-1] - trim_end_s)
    tu = tu[mask]
    rgbu = rgbu[mask]

    sigs = chrominance_signals(rgbu)
    print(f"\n{label}  (fs={fs} Hz, {len(tu)/fs:.1f}s after trim)")
    print(f"      {'signal':<22}{'FFT BPM':>10}{'Welch BPM':>12}")
    print(f"      {'-'*44}")
    out = {}
    for name, sig in sigs.items():
        sig_d = detrend(sig, type="linear")
        sig_bp = bandpass(sig_d, fs, HR_LOW_HZ, HR_HIGH_HZ)
        bpm_fft, freqs, mag = fft_peak_bpm(sig_bp, fs)
        bpm_w, fw, pw = welch_peak_bpm(sig_bp, fs)
        flag = "  <-- 69-74" if 69 <= bpm_w <= 74 else ""
        print(f"      {name:<22}{bpm_fft:>10.2f}{bpm_w:>12.2f}{flag}")
        out[name] = {"sig_bp": sig_bp, "tu": tu, "fw": fw, "pw": pw,
                     "bpm_fft": bpm_fft, "bpm_w": bpm_w}
    return out


def main(trim_start_s: float = 15.0, trim_end_s: float = 0.0) -> None:
    here = Path(__file__).resolve().parent
    work = here / "working"
    data = np.load(work / "signal_roi.npz")
    t = data["t"]
    rgb_roi = data["rgb_roi"]
    rgb_full = data["rgb_full"]
    fps = float(data["fps"])
    area = data["roi_area"]
    print(f"Loaded N={len(t)}  fps={fps:.2f}  "
          f"mean ROI area={area.mean():.0f} px  "
          f"min ROI area={area.min()}")

    full = evaluate("WHOLE FRAME", t, rgb_full, trim_start_s, trim_end_s)
    roi  = evaluate("FACE ROI",    t, rgb_roi,  trim_start_s, trim_end_s)

    # Plot side-by-side comparison.
    sig_names = ["G", "G-R", "POS-X (3R-2G)"]
    fig, axes = plt.subplots(len(sig_names), 2, figsize=(14, 7))
    for i, name in enumerate(sig_names):
        for j, (variant, label) in enumerate(((full, "whole frame"), (roi, "face ROI"))):
            res = variant[name]
            ax = axes[i, j]
            m = (res["fw"] >= 0.5) & (res["fw"] <= 4.0)
            ax.plot(res["fw"][m] * 60, res["pw"][m], color="#0ea5b7")
            ax.axvspan(69, 74, color="g", alpha=0.2, label="GT 69-74")
            ax.axvline(res["bpm_w"], color="r", ls="--",
                       label=f"Welch {res['bpm_w']:.1f}")
            ax.set_xlim(40, 200)
            ax.set_xlabel("BPM")
            ax.set_title(f"{label} - {name}")
            ax.grid(alpha=0.3)
            ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    out = work / "analysis_roi.png"
    plt.savefig(out, dpi=110)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
