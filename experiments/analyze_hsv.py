"""Compare HSV vs RGB-chrominance heart-rate estimation on the ROI signal.

Loads working/signal_hsv.npz produced by extract_signal_hsv.py.

Candidate signals tested:
  - H        : circular hue (degrees), unwrapped
  - S        : saturation
  - V        : value (~ luma)
  - G        : green channel only (sanity)
  - G-R      : chrominance (the previous winner)
  - POS-X    : 3 R/<R> - 2 G/<G>

Each is detrended, bandpassed 0.7-3 Hz, and FFT/Welch-peaked in 42-180 BPM.
Ground truth: 69-74 BPM.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, detrend, filtfilt, welch

HR_LO = 0.7
HR_HI = 3.0


def resample_uniform(t, x, fs):
    t0, t1 = t[0], t[-1]
    n = int(np.floor((t1 - t0) * fs)) + 1
    tu = t0 + np.arange(n) / fs
    if x.ndim == 1:
        return tu, np.interp(tu, t, x)
    out = np.empty((n, x.shape[1]))
    for c in range(x.shape[1]):
        out[:, c] = np.interp(tu, t, x[:, c])
    return tu, out


def bandpass(x, fs, lo, hi, order=4):
    nyq = fs / 2
    b, a = butter(order, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, x, axis=0)


def fft_peak_bpm(x, fs):
    n = len(x)
    nfft = max(1 << 14, 1 << (int(np.ceil(np.log2(n))) + 2))
    spec = np.fft.rfft(x * np.hanning(n), n=nfft)
    freqs = np.fft.rfftfreq(nfft, d=1 / fs)
    mag = np.abs(spec)
    band = (freqs >= HR_LO) & (freqs <= HR_HI)
    peak_idx = int(np.argmax(mag * band))
    bpm = float(freqs[peak_idx] * 60)
    # Heuristic SNR: peak / median of in-band magnitudes.
    in_band = mag[band]
    snr = float(mag[peak_idx] / (np.median(in_band) + 1e-12))
    return bpm, snr, freqs, mag


def welch_peak_bpm(x, fs):
    nperseg = min(len(x), int(fs * 12))
    f, p = welch(x, fs=fs, nperseg=nperseg)
    band = (f >= HR_LO) & (f <= HR_HI)
    peak_idx_b = int(np.argmax(p[band]))
    f_band = f[band]
    p_band = p[band]
    bpm = float(f_band[peak_idx_b] * 60)
    snr = float(p_band[peak_idx_b] / (np.median(p_band) + 1e-12))
    return bpm, snr, f, p


def unwrap_hue_deg(h_deg: np.ndarray) -> np.ndarray:
    """Unwrap a circular angle in degrees so it can be linearly filtered."""
    return np.rad2deg(np.unwrap(np.deg2rad(h_deg)))


def main(trim_start_s: float = 15.0) -> None:
    here = Path(__file__).resolve().parent
    work = here / "working"
    data = np.load(work / "signal_hsv.npz")
    t = data["t"]
    hsv = data["hsv_roi"]   # N x 3, H in deg, S in [0,1], V in [0,1]
    rgb = data["rgb_roi"]   # N x 3, 0-255

    # Sort, dedup, drop NaNs.
    order = np.argsort(t)
    t, hsv, rgb = t[order], hsv[order], rgb[order]
    keep = np.concatenate([[True], np.diff(t) > 0])
    t, hsv, rgb = t[keep], hsv[keep], rgb[keep]
    nan_mask = np.isnan(hsv).any(axis=1) | np.isnan(rgb).any(axis=1)
    if nan_mask.any():
        print(f"Dropping {nan_mask.sum()} NaN rows")
    t = t[~nan_mask]
    hsv = hsv[~nan_mask]
    rgb = rgb[~nan_mask]

    fs = int(round((len(t) - 1) / (t[-1] - t[0])))
    tu, hsvu = resample_uniform(t, hsv, fs)
    _, rgbu = resample_uniform(t, rgb, fs)

    mask = tu >= tu[0] + trim_start_s
    tu = tu[mask]
    hsvu = hsvu[mask]
    rgbu = rgbu[mask]
    print(f"fs={fs} Hz, analysing {len(tu)/fs:.1f}s after {trim_start_s}s trim")

    # Build candidate signals.
    H = unwrap_hue_deg(hsvu[:, 0])  # degrees, monotonic-ish
    S = hsvu[:, 1]
    V = hsvu[:, 2]
    R, G, B = rgbu[:, 0], rgbu[:, 1], rgbu[:, 2]
    Rn = R / R.mean() - 1
    Gn = G / G.mean() - 1
    Bn = B / B.mean() - 1
    POSX = 3 * Rn - 2 * Gn

    candidates = {
        "H (hue, deg)": H,
        "S (sat)":      S,
        "V (val)":      V,
        "G":            Gn,
        "G - R":        Gn - Rn,
        "POS-X":        POSX,
    }

    print(f"\n      {'signal':<14}{'FFT BPM':>10}{'FFT SNR':>10}{'Welch BPM':>12}{'Welch SNR':>12}")
    print(f"      {'-'*60}")
    results = {}
    for name, sig in candidates.items():
        sig_d = detrend(sig, type="linear")
        sig_bp = bandpass(sig_d, fs, HR_LO, HR_HI)
        bpm_f, snr_f, _, _ = fft_peak_bpm(sig_bp, fs)
        bpm_w, snr_w, fw, pw = welch_peak_bpm(sig_bp, fs)
        flag = "  <-- 69-74" if 69 <= bpm_w <= 74 else ""
        print(f"      {name:<14}{bpm_f:>10.2f}{snr_f:>10.2f}{bpm_w:>12.2f}{snr_w:>12.2f}{flag}")
        results[name] = {
            "sig_bp": sig_bp, "tu": tu, "fw": fw, "pw": pw,
            "bpm_w": bpm_w, "snr_w": snr_w,
        }

    # Plot Welch PSDs.
    fig, axes = plt.subplots(len(candidates), 1, figsize=(10, 2.0 * len(candidates)))
    for ax, (name, res) in zip(axes, results.items()):
        m = (res["fw"] >= 0.5) & (res["fw"] <= 4.0)
        ax.plot(res["fw"][m] * 60, res["pw"][m], color="#0ea5b7")
        ax.axvspan(69, 74, color="g", alpha=0.2, label="GT 69-74")
        ax.axvline(res["bpm_w"], color="r", ls="--",
                   label=f"peak {res['bpm_w']:.1f}  SNR {res['snr_w']:.1f}")
        ax.set_xlim(40, 200)
        ax.set_title(f"{name}")
        ax.set_xlabel("BPM")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    out = work / "analysis_hsv.png"
    plt.savefig(out, dpi=110)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
