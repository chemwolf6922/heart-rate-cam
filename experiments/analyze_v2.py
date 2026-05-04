"""Iteration 2: handle the motion-artifact transient and try a chrominance signal.

Findings from analyze.py: a big motion artifact at t~8-10s pollutes the FFT and
biases the peak toward ~50 BPM, even though a real peak near 70 BPM is visible.

This script:
  - trims a configurable warm-up window (default 15 s),
  - compares per-channel signals to a simple chrominance signal X = 3R - 2G,
    which is one of the building blocks of POS/CHROM rPPG and largely cancels
    luma-motion artifacts (motion shifts all channels together; pulse modulates
    them differently).
  - reports both FFT-peak and Welch-peak BPM.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, detrend, filtfilt, welch

HR_LOW_HZ = 0.7   # 42 BPM
HR_HIGH_HZ = 3.0  # 180 BPM


def resample_uniform(t: np.ndarray, x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    t0, t1 = t[0], t[-1]
    n = int(np.floor((t1 - t0) * fs)) + 1
    t_uniform = t0 + np.arange(n) / fs
    if x.ndim == 1:
        return t_uniform, np.interp(t_uniform, t, x)
    out = np.empty((n, x.shape[1]))
    for c in range(x.shape[1]):
        out[:, c] = np.interp(t_uniform, t, x[:, c])
    return t_uniform, out


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


def normalize(x: np.ndarray) -> np.ndarray:
    """Per-channel zero-mean unit-std (over time)."""
    return (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-12)


def main(trim_start_s: float = 15.0, trim_end_s: float = 0.0) -> None:
    here = Path(__file__).resolve().parent
    work = here / "working"
    work.mkdir(parents=True, exist_ok=True)
    data = np.load(work / "signal.npz")
    t = data["t"]
    rgb = data["rgb"]
    nominal_fps = float(data["fps"])

    # Sort/dedup by time.
    order = np.argsort(t)
    t = t[order]
    rgb = rgb[order]
    keep = np.concatenate([[True], np.diff(t) > 0])
    t = t[keep]
    rgb = rgb[keep]

    actual_fps = (len(t) - 1) / (t[-1] - t[0])
    print(f"Loaded {len(t)} frames, duration {t[-1]-t[0]:.2f}s, "
          f"nominal fps {nominal_fps:.2f}, actual fps {actual_fps:.2f}")

    fs = int(round(actual_fps))
    t_u, x = resample_uniform(t, rgb, fs)

    # Trim warmup / cool-down.
    mask = (t_u >= t_u[0] + trim_start_s) & (t_u <= t_u[-1] - trim_end_s)
    t_u = t_u[mask]
    x = x[mask]
    print(f"Analyzing {len(t_u)/fs:.2f}s (trimmed {trim_start_s}s start, "
          f"{trim_end_s}s end), fs={fs} Hz")

    # Build candidate signals.
    R, G, B = x[:, 0], x[:, 1], x[:, 2]

    # Standard rPPG approach: normalize each channel (divide by mean) so that
    # multiplicative illumination changes cancel.
    Rn = R / R.mean() - 1
    Gn = G / G.mean() - 1
    Bn = B / B.mean() - 1

    # POS-style chrominance signals.
    # X = 3 Rn - 2 Gn   (pulse-emphasizing)
    # Y = 1.5 Rn + Gn - 1.5 Bn
    Xc = 3 * Rn - 2 * Gn
    Yc = 1.5 * Rn + Gn - 1.5 * Bn

    candidates = {
        "R":   Rn,
        "G":   Gn,
        "B":   Bn,
        "G-R": Gn - Rn,
        "POS-X (3R-2G)": Xc,
        "POS-Y (1.5R+G-1.5B)": Yc,
    }

    print("\n      signal             FFT BPM   Welch BPM")
    print("      " + "-" * 46)
    results = {}
    for name, sig in candidates.items():
        sig_d = detrend(sig, type="linear")
        sig_bp = bandpass(sig_d, fs, HR_LOW_HZ, HR_HIGH_HZ)
        bpm_fft, freqs, mag = fft_peak_bpm(sig_bp, fs)
        bpm_w, fw, pw = welch_peak_bpm(sig_bp, fs)
        results[name] = {
            "sig_bp": sig_bp, "freqs": freqs, "mag": mag,
            "fw": fw, "pw": pw, "bpm_fft": bpm_fft, "bpm_w": bpm_w,
        }
        flag = "  <-- in 69-74" if 69 <= bpm_w <= 74 else ""
        print(f"      {name:<22}{bpm_fft:7.2f}    {bpm_w:7.2f}{flag}")

    # Plot.
    n = len(candidates)
    fig, axes = plt.subplots(n, 2, figsize=(14, 2.2 * n))
    for i, (name, res) in enumerate(results.items()):
        ax = axes[i, 0]
        ax.plot(t_u, res["sig_bp"], lw=0.8)
        ax.set_title(f"{name}  (bandpassed)")
        ax.set_xlabel("time (s)")
        ax.grid(alpha=0.3)

        ax = axes[i, 1]
        m = (res["fw"] >= 0.5) & (res["fw"] <= 4.0)
        ax.plot(res["fw"][m] * 60, res["pw"][m])
        ax.axvspan(69, 74, color="g", alpha=0.2, label="GT 69-74")
        ax.axvline(res["bpm_w"], color="r", ls="--",
                   label=f"Welch peak {res['bpm_w']:.1f}")
        ax.set_xlim(40, 200)
        ax.set_xlabel("BPM")
        ax.set_title(f"{name}  Welch PSD")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    out = work / "analysis_v2.png"
    plt.savefig(out, dpi=110)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
