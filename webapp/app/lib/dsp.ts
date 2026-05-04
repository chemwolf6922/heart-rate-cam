// Tiny DSP utilities used by the heart-rate monitor.
//
// Everything runs client-side. We avoid pulling in fft.js etc. because all we
// need is one radix-2 FFT on ~256-1024 samples per update.

/** In-place radix-2 Cooley-Tukey FFT. `re` and `im` must have length = power of two. */
export function fft(re: Float32Array, im: Float32Array): void {
  const n = re.length;
  if ((n & (n - 1)) !== 0) throw new Error(`FFT length must be power of two, got ${n}`);

  // Bit-reversal permutation.
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }

  // Butterflies.
  for (let size = 2; size <= n; size <<= 1) {
    const half = size >> 1;
    const theta = (-2 * Math.PI) / size;
    const wRe = Math.cos(theta);
    const wIm = Math.sin(theta);
    for (let i = 0; i < n; i += size) {
      let curRe = 1;
      let curIm = 0;
      for (let k = 0; k < half; k++) {
        const aRe = re[i + k];
        const aIm = im[i + k];
        const bRe = re[i + k + half] * curRe - im[i + k + half] * curIm;
        const bIm = re[i + k + half] * curIm + im[i + k + half] * curRe;
        re[i + k] = aRe + bRe;
        im[i + k] = aIm + bIm;
        re[i + k + half] = aRe - bRe;
        im[i + k + half] = aIm - bIm;
        const nextRe = curRe * wRe - curIm * wIm;
        const nextIm = curRe * wIm + curIm * wRe;
        curRe = nextRe;
        curIm = nextIm;
      }
    }
  }
}

/** Linearly resample a non-uniform time series onto a uniform grid at `fs` Hz. */
export function resampleUniform(
  t: Float32Array,
  x: Float32Array,
  fs: number,
): { tu: Float32Array; xu: Float32Array } {
  const t0 = t[0];
  const t1 = t[t.length - 1];
  const n = Math.max(2, Math.floor((t1 - t0) * fs) + 1);
  const tu = new Float32Array(n);
  const xu = new Float32Array(n);
  let j = 0;
  for (let i = 0; i < n; i++) {
    const tt = t0 + i / fs;
    tu[i] = tt;
    while (j + 1 < t.length && t[j + 1] < tt) j++;
    if (j + 1 >= t.length) {
      xu[i] = x[t.length - 1];
    } else {
      const span = t[j + 1] - t[j];
      const a = span > 0 ? (tt - t[j]) / span : 0;
      xu[i] = x[j] * (1 - a) + x[j + 1] * a;
    }
  }
  return { tu, xu };
}

/** Subtract linear least-squares trend in place. */
export function detrend(x: Float32Array): void {
  const n = x.length;
  if (n < 2) return;
  let sumX = 0;
  let sumY = 0;
  let sumXX = 0;
  let sumXY = 0;
  for (let i = 0; i < n; i++) {
    sumX += i;
    sumY += x[i];
    sumXX += i * i;
    sumXY += i * x[i];
  }
  const denom = n * sumXX - sumX * sumX;
  if (denom === 0) {
    const mean = sumY / n;
    for (let i = 0; i < n; i++) x[i] -= mean;
    return;
  }
  const slope = (n * sumXY - sumX * sumY) / denom;
  const intercept = (sumY - slope * sumX) / n;
  for (let i = 0; i < n; i++) x[i] -= slope * i + intercept;
}

/** Apply a Hann window in place. */
export function hann(x: Float32Array): void {
  const n = x.length;
  for (let i = 0; i < n; i++) {
    x[i] *= 0.5 * (1 - Math.cos((2 * Math.PI * i) / (n - 1)));
  }
}

/** Round up to the next power of two (>=2). */
export function nextPow2(n: number): number {
  let p = 2;
  while (p < n) p <<= 1;
  return p;
}

export interface PeakResult {
  /** Peak frequency in Hz. */
  freqHz: number;
  /** Peak BPM. */
  bpm: number;
  /** Heuristic SNR: peak power / median in-band power. */
  snr: number;
  /** Spectrum frequencies (Hz). */
  freqs: Float32Array;
  /** Spectrum magnitudes. */
  mags: Float32Array;
}

/**
 * Compute magnitude spectrum and pick the peak BPM in [loHz, hiHz].
 * Returns null if input is too short.
 */
export function spectralPeak(
  x: Float32Array,
  fs: number,
  loHz = 0.7,
  hiHz = 3.0,
): PeakResult | null {
  const n = x.length;
  if (n < 16) return null;
  const nfft = nextPow2(Math.max(n * 2, 1024));
  const re = new Float32Array(nfft);
  const im = new Float32Array(nfft);

  // Copy detrended + Hann-windowed input into the zero-padded FFT buffer.
  const buf = new Float32Array(n);
  buf.set(x);
  detrend(buf);
  hann(buf);
  re.set(buf);

  fft(re, im);

  const halfN = nfft / 2;
  const freqs = new Float32Array(halfN);
  const mags = new Float32Array(halfN);
  for (let i = 0; i < halfN; i++) {
    freqs[i] = (i * fs) / nfft;
    mags[i] = Math.hypot(re[i], im[i]);
  }

  let peakIdx = -1;
  let peakMag = 0;
  let bandStart = -1;
  let bandEnd = -1;
  for (let i = 0; i < halfN; i++) {
    if (freqs[i] >= loHz && freqs[i] <= hiHz) {
      if (bandStart < 0) bandStart = i;
      bandEnd = i;
      if (mags[i] > peakMag) {
        peakMag = mags[i];
        peakIdx = i;
      }
    }
  }
  if (peakIdx < 0) return null;

  // Quadratic interpolation around the peak for sub-bin precision.
  let freqHz = freqs[peakIdx];
  if (peakIdx > 0 && peakIdx < halfN - 1) {
    const a = mags[peakIdx - 1];
    const b = mags[peakIdx];
    const c = mags[peakIdx + 1];
    const denom = a - 2 * b + c;
    if (denom !== 0) {
      const offset = (0.5 * (a - c)) / denom;
      freqHz = freqs[peakIdx] + offset * (fs / nfft);
    }
  }

  // Heuristic SNR: peak / median of in-band magnitudes.
  const inBand: number[] = [];
  for (let i = bandStart; i <= bandEnd; i++) inBand.push(mags[i]);
  inBand.sort((a, b) => a - b);
  const median = inBand[Math.floor(inBand.length / 2)] || 1e-9;
  const snr = peakMag / median;

  return { freqHz, bpm: freqHz * 60, snr, freqs, mags };
}
