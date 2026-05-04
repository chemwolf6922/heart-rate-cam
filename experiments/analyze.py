"""Analyze the extracted RGB signal and estimate heart rate.

Method (simple baseline):
    1. Resample the per-frame mean RGB to a uniform sample rate.
    2. Detrend each channel.
    3. Bandpass filter to the plausible heart-rate band (0.7 - 3.0 Hz, ~42-180 BPM).
    4. FFT and pick the peak in that band for each channel.
    5. Report BPM and save a diagnostic figure.

Ground truth for the test clip: ~69-74 BPM.
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
    """Linearly resample columns of x at uniform sample rate fs."""
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
    """Return (bpm, freqs_hz, magnitude) for the dominant peak in the HR band."""
    n = len(x)
    # Zero-pad for finer frequency resolution.
    nfft = max(1 << 14, 1 << int(np.ceil(np.log2(n))) + 2)
    spec = np.fft.rfft(x * np.hanning(n), n=nfft)
    freqs = np.fft.rfftfreq(nfft, d=1 / fs)
    mag = np.abs(spec)
    band = (freqs >= HR_LOW_HZ) & (freqs <= HR_HIGH_HZ)
    if not np.any(band):
        return float("nan"), freqs, mag
    peak_idx = np.argmax(mag * band)
    bpm = freqs[peak_idx] * 60.0
    return float(bpm), freqs, mag


def main() -> None:
    here = Path(__file__).resolve().parent
    work = here / "working"
    work.mkdir(parents=True, exist_ok=True)
    data = np.load(work / "signal.npz")
    t = data["t"]
    rgb = data["rgb"]  # (N, 3) -> R, G, B
    nominal_fps = float(data["fps"])

    # Drop possible duplicate or non-monotonic timestamps.
    keep = np.concatenate([[True], np.diff(t) > 0])
    t = t[keep]
    rgb = rgb[keep]

    actual_fps = (len(t) - 1) / (t[-1] - t[0])
    print(f"Frames: {len(t)}  duration: {t[-1] - t[0]:.2f}s  "
          f"nominal fps: {nominal_fps:.2f}  actual fps: {actual_fps:.2f}")

    fs = round(actual_fps)
    if fs < 10:
        fs = max(int(round(actual_fps)), 1)
    print(f"Resampling to fs={fs} Hz")

    t_u, x = resample_uniform(t, rgb, fs)

    # Detrend (remove slow drift) per channel.
    x_d = detrend(x, axis=0, type="linear")

    # Bandpass to heart-rate band.
    x_bp = bandpass(x_d, fs, HR_LOW_HZ, HR_HIGH_HZ)

    channels = ["R", "G", "B"]
    print("\nFFT peak BPM per channel (band-limited):")
    bpms = {}
    spectra = {}
    for c, name in enumerate(channels):
        bpm, freqs, mag = fft_peak_bpm(x_bp[:, c], fs)
        bpms[name] = bpm
        spectra[name] = (freqs, mag)
        print(f"  {name}: {bpm:6.2f} BPM")

    # Welch as a sanity check (more stable for noisy data).
    print("\nWelch peak BPM per channel:")
    welch_bpms = {}
    for c, name in enumerate(channels):
        nperseg = min(len(x_bp), fs * 10)  # 10-second windows
        f_w, p_w = welch(x_bp[:, c], fs=fs, nperseg=nperseg)
        band = (f_w >= HR_LOW_HZ) & (f_w <= HR_HIGH_HZ)
        peak = f_w[band][np.argmax(p_w[band])]
        welch_bpms[name] = peak * 60.0
        print(f"  {name}: {peak * 60.0:6.2f} BPM")

    # Plot diagnostics.
    fig, axes = plt.subplots(3, 1, figsize=(10, 9))

    # Raw mean intensities.
    ax = axes[0]
    for c, name, color in zip(range(3), channels, ["r", "g", "b"]):
        ax.plot(t_u, x[:, c], color=color, label=f"{name} mean")
    ax.set_title("Whole-frame mean intensity (raw)")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("intensity")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    # Bandpassed signals.
    ax = axes[1]
    for c, name, color in zip(range(3), channels, ["r", "g", "b"]):
        ax.plot(t_u, x_bp[:, c], color=color, label=f"{name} bp")
    ax.set_title(f"Bandpassed {HR_LOW_HZ}-{HR_HIGH_HZ} Hz (heart-rate band)")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("a.u.")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    # Spectra.
    ax = axes[2]
    for name, color in zip(channels, ["r", "g", "b"]):
        freqs, mag = spectra[name]
        m = (freqs >= 0.5) & (freqs <= 4.0)
        ax.plot(freqs[m] * 60, mag[m], color=color,
                label=f"{name}  FFT={bpms[name]:.1f}  Welch={welch_bpms[name]:.1f}")
    ax.axvspan(HR_LOW_HZ * 60, HR_HIGH_HZ * 60, color="k", alpha=0.05)
    ax.axvspan(69, 74, color="g", alpha=0.15, label="ground truth 69-74")
    ax.set_title("Magnitude spectrum (BPM)")
    ax.set_xlabel("BPM")
    ax.set_ylabel("|FFT|")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out = work / "analysis.png"
    plt.savefig(out, dpi=120)
    print(f"\nSaved {out}")

    # Verdict.
    best_channel = max(welch_bpms, key=lambda k: 0 if np.isnan(welch_bpms[k]) else 1)
    print(f"\nGround truth: 69-74 BPM")
    print(f"Best estimate (Welch, {best_channel}): {welch_bpms[best_channel]:.2f} BPM")


if __name__ == "__main__":
    main()
