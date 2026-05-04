"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { FaceLandmarker } from "@mediapipe/tasks-vision";
import { resampleUniform, spectralPeak, type PeakResult } from "@/app/lib/dsp";
import {
  buildRoiPolygons,
  loadFaceLandmarker,
  meanRgbInRoi,
  type RoiPolygons,
} from "@/app/lib/face";

// --- Tunables ---------------------------------------------------------------

const TARGET_FS = 30;          // Resampling rate (Hz). Webcam ≈ 30 fps.
const WINDOW_SECONDS = 10;     // Length of analysis window.
const WARMUP_SECONDS = 4;      // Skip the first few seconds (sensor settling).
const MAX_SECONDS = 20;        // Ring-buffer capacity.
const HR_LO_HZ = 0.7;          // 42 BPM
const HR_HI_HZ = 3.0;          // 180 BPM
const SAMPLE_WIDTH = 320;      // Width of the offscreen sampling canvas.
const UPDATE_INTERVAL_MS = 250;

// --- Types ------------------------------------------------------------------

type Status = "idle" | "starting" | "running" | "error";

interface Sample {
  t: number; // seconds since start
  r: number;
  g: number;
  b: number;
}

// --- Helpers ----------------------------------------------------------------

function meanRgbWholeFrame(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  stride: number = 4,
): { r: number; g: number; b: number } {
  const img = ctx.getImageData(0, 0, width, height).data;
  let r = 0;
  let g = 0;
  let b = 0;
  let n = 0;
  for (let y = 0; y < height; y += stride) {
    const row = y * width * 4;
    for (let x = 0; x < width; x += stride) {
      const i = row + x * 4;
      r += img[i];
      g += img[i + 1];
      b += img[i + 2];
      n++;
    }
  }
  return { r: r / n, g: g / n, b: b / n };
}

function chrominanceGR(samples: Sample[]): { t: Float32Array; x: Float32Array } {
  // G/<G> − R/<R>: cancels common-mode lighting drift.
  const n = samples.length;
  let mr = 0;
  let mg = 0;
  for (let i = 0; i < n; i++) {
    mr += samples[i].r;
    mg += samples[i].g;
  }
  mr /= n;
  mg /= n;
  const t = new Float32Array(n);
  const x = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    t[i] = samples[i].t;
    x[i] = samples[i].g / mg - samples[i].r / mr;
  }
  return { t, x };
}

// --- Component --------------------------------------------------------------

export default function HeartRateMonitor() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const sampleCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const maskCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const waveCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const specCanvasRef = useRef<HTMLCanvasElement | null>(null);

  const samplesRef = useRef<Sample[]>([]);
  const startTimeRef = useRef<number>(0);
  const rafRef = useRef<number | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const lastUpdateRef = useRef<number>(0);
  const landmarkerRef = useRef<FaceLandmarker | null>(null);
  const lastRoiRef = useRef<RoiPolygons | null>(null);
  const lastRoiTimeRef = useRef<number>(0);

  const [status, setStatus] = useState<Status>("idle");
  const [errorMsg, setErrorMsg] = useState<string>("");
  const [bpm, setBpm] = useState<number | null>(null);
  const [snr, setSnr] = useState<number>(0);
  const [bufferFill, setBufferFill] = useState<number>(0);
  const [fps, setFps] = useState<number>(0);
  const [roiActive, setRoiActive] = useState<boolean>(false);
  const [modelLoading, setModelLoading] = useState<boolean>(false);

  const stop = useCallback(() => {
    if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    rafRef.current = null;
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (videoRef.current) videoRef.current.srcObject = null;
    samplesRef.current = [];
    lastRoiRef.current = null;
    setStatus("idle");
    setBpm(null);
    setSnr(0);
    setBufferFill(0);
    setRoiActive(false);
  }, []);

  // Draw the polygon overlay on top of the live video preview.
  const drawOverlay = useCallback(
    (roi: RoiPolygons | null, sampleW: number, sampleH: number) => {
      const canvas = overlayCanvasRef.current;
      const video = videoRef.current;
      if (!canvas || !video) return;
      // Match overlay to the displayed video size.
      const displayW = video.clientWidth;
      const displayH = video.clientHeight;
      if (displayW === 0 || displayH === 0) return;
      if (canvas.width !== displayW || canvas.height !== displayH) {
        canvas.width = displayW;
        canvas.height = displayH;
      }
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.clearRect(0, 0, displayW, displayH);
      if (!roi) return;

      // Sample canvas was a uniform scaling of the displayed video. Map
      // sample-pixel polygons onto the displayed area, then mirror in X to
      // match the CSS scale-x-[-1] preview.
      const sx = displayW / sampleW;
      const sy = displayH / sampleH;

      ctx.save();
      // Mirror to match the mirrored preview.
      ctx.translate(displayW, 0);
      ctx.scale(-1, 1);
      ctx.lineWidth = 2;
      ctx.strokeStyle = "rgba(34, 211, 238, 0.9)";
      ctx.fillStyle = "rgba(34, 211, 238, 0.18)";
      for (const poly of [roi.forehead, roi.leftCheek, roi.rightCheek]) {
        if (poly.length < 3) continue;
        ctx.beginPath();
        ctx.moveTo(poly[0][0] * sx, poly[0][1] * sy);
        for (let i = 1; i < poly.length; i++) {
          ctx.lineTo(poly[i][0] * sx, poly[i][1] * sy);
        }
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
      }
      ctx.restore();
    },
    [],
  );

  // Process one frame: draw video, run face detector, sample R/G in ROI.
  const processFrame = useCallback(() => {
    const video = videoRef.current;
    const sampleCanvas = sampleCanvasRef.current;
    const maskCanvas = maskCanvasRef.current;
    if (!video || !sampleCanvas || !maskCanvas) return;
    if (video.readyState < 2) return;

    const w = sampleCanvas.width;
    const h = sampleCanvas.height;
    const sampleCtx = sampleCanvas.getContext("2d", { willReadFrequently: true });
    const maskCtx = maskCanvas.getContext("2d", { willReadFrequently: true });
    if (!sampleCtx || !maskCtx) return;
    sampleCtx.drawImage(video, 0, 0, w, h);

    const tsMs = performance.now();
    let used: { r: number; g: number; b: number };
    let roi: RoiPolygons | null = null;
    const landmarker = landmarkerRef.current;

    if (landmarker) {
      const res = landmarker.detectForVideo(sampleCanvas, tsMs);
      if (res.faceLandmarks.length > 0) {
        roi = buildRoiPolygons(res.faceLandmarks[0], w, h);
        lastRoiRef.current = roi;
        lastRoiTimeRef.current = tsMs;
      } else if (lastRoiRef.current && tsMs - lastRoiTimeRef.current < 500) {
        // Reuse a recent ROI for a brief detection gap.
        roi = lastRoiRef.current;
      }
    }

    if (roi) {
      const m = meanRgbInRoi(sampleCtx, maskCtx, roi, w, h, 2);
      if (m) {
        used = m;
        if (!roiActive) setRoiActive(true);
      } else {
        used = meanRgbWholeFrame(sampleCtx, w, h);
        if (roiActive) setRoiActive(false);
      }
    } else {
      used = meanRgbWholeFrame(sampleCtx, w, h);
      if (roiActive) setRoiActive(false);
    }

    drawOverlay(roi, w, h);

    const t = tsMs / 1000 - startTimeRef.current;
    samplesRef.current.push({ t, r: used.r, g: used.g, b: used.b });

    const cutoff = t - MAX_SECONDS;
    while (samplesRef.current.length > 0 && samplesRef.current[0].t < cutoff) {
      samplesRef.current.shift();
    }
  }, [drawOverlay, roiActive]);

  const computeBpm = useCallback((): PeakResult | null => {
    const samples = samplesRef.current;
    if (samples.length < 2) return null;
    const elapsed = samples[samples.length - 1].t - samples[0].t;
    if (elapsed < WARMUP_SECONDS) return null;

    const tEnd = samples[samples.length - 1].t;
    const tStart = Math.max(samples[0].t, tEnd - WINDOW_SECONDS);
    const windowed = samples.filter((s) => s.t >= tStart);
    if (windowed.length < TARGET_FS * 2) return null;

    const { t, x } = chrominanceGR(windowed);
    const { xu } = resampleUniform(t, x, TARGET_FS);
    return spectralPeak(xu, TARGET_FS, HR_LO_HZ, HR_HI_HZ);
  }, []);

  const drawWaveform = useCallback(() => {
    const canvas = waveCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const w = canvas.width;
    const h = canvas.height;
    ctx.fillStyle = "#0a0a0a";
    ctx.fillRect(0, 0, w, h);

    const samples = samplesRef.current;
    if (samples.length < 2) return;
    const tEnd = samples[samples.length - 1].t;
    const tStart = Math.max(samples[0].t, tEnd - WINDOW_SECONDS);
    const windowed = samples.filter((s) => s.t >= tStart);
    if (windowed.length < 2) return;

    const { t, x } = chrominanceGR(windowed);
    const { tu, xu } = resampleUniform(t, x, TARGET_FS);

    let mn = Infinity;
    let mx = -Infinity;
    for (let i = 0; i < xu.length; i++) {
      if (xu[i] < mn) mn = xu[i];
      if (xu[i] > mx) mx = xu[i];
    }
    const span = Math.max(1e-6, mx - mn);

    ctx.strokeStyle = "#22d3ee";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i < xu.length; i++) {
      const px = (i / (xu.length - 1)) * w;
      const py = h - ((xu[i] - mn) / span) * (h - 8) - 4;
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.stroke();

    ctx.fillStyle = "#737373";
    ctx.font = "11px ui-sans-serif, system-ui";
    ctx.fillText(
      `G−R chrominance, last ${(tu[tu.length - 1] - tu[0]).toFixed(1)}s`,
      6,
      14,
    );
  }, []);

  const drawSpectrum = useCallback((peak: PeakResult | null) => {
    const canvas = specCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const w = canvas.width;
    const h = canvas.height;
    ctx.fillStyle = "#0a0a0a";
    ctx.fillRect(0, 0, w, h);

    if (!peak) {
      ctx.fillStyle = "#737373";
      ctx.font = "11px ui-sans-serif, system-ui";
      ctx.fillText("Spectrum (collecting…)", 6, 14);
      return;
    }

    const { freqs, mags } = peak;
    const lo = 0.5;
    const hi = 4.0;
    let mx = 0;
    for (let i = 0; i < freqs.length; i++) {
      if (freqs[i] >= lo && freqs[i] <= hi && mags[i] > mx) mx = mags[i];
    }
    if (mx <= 0) return;

    const bpmToX = (bpm: number) => ((bpm / 60 - lo) / (hi - lo)) * w;

    // Vertical gridlines at common BPM ticks.
    const ticks = [40, 60, 80, 100, 120, 150, 180, 220];
    ctx.strokeStyle = "rgba(115,115,115,0.18)";
    ctx.lineWidth = 1;
    for (const bpm of ticks) {
      const x = bpmToX(bpm);
      if (x < 0 || x > w) continue;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h - 14);
      ctx.stroke();
    }

    // Heart-rate band shading (under the curve).
    const bandX0 = ((HR_LO_HZ - lo) / (hi - lo)) * w;
    const bandX1 = ((HR_HI_HZ - lo) / (hi - lo)) * w;
    ctx.fillStyle = "rgba(34, 211, 238, 0.06)";
    ctx.fillRect(bandX0, 0, bandX1 - bandX0, h - 14);

    // Spectrum curve.
    ctx.strokeStyle = "#22d3ee";
    ctx.lineWidth = 1;
    ctx.beginPath();
    let started = false;
    for (let i = 0; i < freqs.length; i++) {
      const f = freqs[i];
      if (f < lo || f > hi) continue;
      const px = ((f - lo) / (hi - lo)) * w;
      const py = h - 14 - (mags[i] / mx) * (h - 22) - 4;
      if (!started) {
        ctx.moveTo(px, py);
        started = true;
      } else {
        ctx.lineTo(px, py);
      }
    }
    ctx.stroke();

    // Peak marker + numeric label.
    const peakX = bpmToX(peak.bpm);
    ctx.strokeStyle = "#f43f5e";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(peakX, 0);
    ctx.lineTo(peakX, h - 14);
    ctx.stroke();
    ctx.fillStyle = "#fda4af";
    ctx.font = "11px ui-sans-serif, system-ui";
    const peakLabel = `${peak.bpm.toFixed(1)} BPM`;
    const labelW = ctx.measureText(peakLabel).width;
    const labelX = Math.min(Math.max(peakX + 4, 4), w - labelW - 4);
    ctx.fillText(peakLabel, labelX, 12);

    // Tick labels on the BPM axis.
    ctx.fillStyle = "#a3a3a3";
    ctx.font = "10px ui-sans-serif, system-ui";
    ctx.textBaseline = "alphabetic";
    for (const bpm of ticks) {
      const x = bpmToX(bpm);
      if (x < 0 || x > w) continue;
      const text = String(bpm);
      const tw = ctx.measureText(text).width;
      ctx.fillText(text, Math.min(Math.max(x - tw / 2, 2), w - tw - 2), h - 3);
    }
    ctx.fillText("BPM", w - 26, h - 3);
  }, []);

  const tick = useCallback(() => {
    rafRef.current = requestAnimationFrame(tick);
    processFrame();

    const now = performance.now();
    if (now - lastUpdateRef.current < UPDATE_INTERVAL_MS) return;
    lastUpdateRef.current = now;

    const samples = samplesRef.current;
    if (samples.length === 0) return;
    const elapsed = samples[samples.length - 1].t - samples[0].t;
    setBufferFill(Math.min(1, elapsed / WINDOW_SECONDS));
    setFps(samples.length / Math.max(0.1, elapsed));

    const peak = computeBpm();
    drawWaveform();
    drawSpectrum(peak);
    if (peak && elapsed >= WARMUP_SECONDS) {
      setBpm(peak.bpm);
      setSnr(peak.snr);
    }
  }, [processFrame, computeBpm, drawWaveform, drawSpectrum]);

  const start = useCallback(async () => {
    setErrorMsg("");
    setStatus("starting");
    try {
      // Kick off the model load (in parallel with camera permission prompt).
      setModelLoading(true);
      const landmarkerPromise = loadFaceLandmarker().catch((e) => {
        // Non-fatal: fall back to whole-frame.
        console.warn("FaceLandmarker failed to load, falling back:", e);
        return null;
      });

      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "user",
          width: { ideal: 640 },
          height: { ideal: 480 },
          frameRate: { ideal: 30 },
        },
        audio: false,
      });
      streamRef.current = stream;
      const video = videoRef.current!;
      video.srcObject = stream;
      await video.play();

      const sampleCanvas = sampleCanvasRef.current!;
      const maskCanvas = maskCanvasRef.current!;
      const aspect = video.videoWidth / video.videoHeight || 4 / 3;
      const w = Math.min(SAMPLE_WIDTH, video.videoWidth || SAMPLE_WIDTH);
      const h = Math.round(w / aspect);
      sampleCanvas.width = w;
      sampleCanvas.height = h;
      maskCanvas.width = w;
      maskCanvas.height = h;

      // Wait for landmarker (up to camera startup time anyway).
      const lm = await landmarkerPromise;
      landmarkerRef.current = lm;
      setModelLoading(false);

      samplesRef.current = [];
      startTimeRef.current = performance.now() / 1000;
      lastUpdateRef.current = 0;
      lastRoiRef.current = null;
      setStatus("running");
      rafRef.current = requestAnimationFrame(tick);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      setErrorMsg(msg);
      setStatus("error");
      setModelLoading(false);
    }
  }, [tick]);

  useEffect(() => {
    return () => stop();
  }, [stop]);

  const ready = bufferFill >= 1 && bpm !== null;
  const sourceLabel = roiActive ? "face ROI" : "whole frame";

  return (
    <div className="w-full max-w-4xl mx-auto p-6 flex flex-col gap-6">
      <header className="flex items-baseline justify-between">
        <h1 className="text-2xl font-semibold tracking-tight">Heart-rate camera</h1>
        <span className="text-xs text-neutral-500 font-mono">
          {fps > 0 ? `${fps.toFixed(1)} fps` : ""}
        </span>
      </header>

      <p className="text-sm text-neutral-400 -mt-2">
        Hold still in even lighting and look at the camera. The estimate
        stabilizes after about {WARMUP_SECONDS + WINDOW_SECONDS / 2}s.
      </p>

      <div className="grid md:grid-cols-2 gap-6">
        <div className="flex flex-col gap-4">
          <div className="relative aspect-[4/3] rounded-xl overflow-hidden bg-black border border-neutral-800">
            <video
              ref={videoRef}
              playsInline
              muted
              className="w-full h-full object-cover scale-x-[-1]"
            />
            <canvas
              ref={overlayCanvasRef}
              className="absolute inset-0 w-full h-full pointer-events-none"
            />
            <canvas ref={sampleCanvasRef} className="hidden" />
            <canvas ref={maskCanvasRef} className="hidden" />
            {status === "idle" && (
              <div className="absolute inset-0 flex items-center justify-center bg-black/50">
                <button
                  onClick={start}
                  className="rounded-lg bg-cyan-500 hover:bg-cyan-400 text-black font-medium px-5 py-2 transition-colors"
                >
                  Start camera
                </button>
              </div>
            )}
            {status === "starting" && (
              <div className="absolute inset-0 flex items-center justify-center bg-black/50 text-neutral-200 text-sm">
                {modelLoading ? "Loading face model…" : "Requesting camera…"}
              </div>
            )}
            {status === "error" && (
              <div className="absolute inset-0 flex items-center justify-center bg-black/70 text-rose-300 text-sm p-4 text-center">
                {errorMsg || "Could not access camera."}
              </div>
            )}
            {status === "running" && (
              <div className="absolute top-2 right-2 text-[10px] uppercase tracking-wider px-2 py-0.5 rounded-full font-mono backdrop-blur-sm"
                   style={{
                     background: roiActive ? "rgba(34,211,238,0.18)" : "rgba(245,158,11,0.18)",
                     color: roiActive ? "#67e8f9" : "#fbbf24",
                   }}>
                {sourceLabel}
              </div>
            )}
          </div>

          {status === "running" && (
            <button
              onClick={stop}
              className="self-start rounded-lg border border-neutral-700 hover:border-neutral-500 text-neutral-200 px-3 py-1.5 text-sm transition-colors"
            >
              Stop
            </button>
          )}
        </div>

        <div className="flex flex-col gap-4">
          <div className="rounded-xl border border-neutral-800 p-5 bg-neutral-950">
            <div className="text-xs uppercase tracking-wider text-neutral-500">
              Heart rate
            </div>
            <div className="mt-1 flex items-baseline gap-3">
              <span
                className={`text-6xl font-mono tabular-nums ${
                  ready ? "text-cyan-300" : "text-neutral-600"
                }`}
              >
                {bpm !== null ? bpm.toFixed(1) : "--"}
              </span>
              <span className="text-neutral-500 text-lg">BPM</span>
            </div>
            <div className="mt-3 text-xs text-neutral-500 flex gap-4">
              <span>
                buffer{" "}
                <span className="text-neutral-300 font-mono">
                  {(bufferFill * 100).toFixed(0)}%
                </span>
              </span>
              <span>
                snr{" "}
                <span className="text-neutral-300 font-mono">
                  {snr ? snr.toFixed(1) : "--"}
                </span>
              </span>
              <span>
                source{" "}
                <span className="text-neutral-300 font-mono">{sourceLabel}</span>
              </span>
            </div>
            <div className="mt-2 h-1 rounded-full bg-neutral-800 overflow-hidden">
              <div
                className="h-full bg-cyan-500 transition-[width] duration-200"
                style={{ width: `${bufferFill * 100}%` }}
              />
            </div>
          </div>

          <div className="rounded-xl border border-neutral-800 bg-neutral-950 overflow-hidden">
            <canvas
              ref={waveCanvasRef}
              width={520}
              height={120}
              className="w-full block"
            />
          </div>
          <div className="rounded-xl border border-neutral-800 bg-neutral-950 overflow-hidden">
            <canvas
              ref={specCanvasRef}
              width={520}
              height={120}
              className="w-full block"
            />
          </div>
        </div>
      </div>

      <details className="text-sm text-neutral-400">
        <summary className="cursor-pointer text-neutral-300">How it works</summary>
        <div className="mt-2 space-y-2">
          <p>
            MediaPipe Face Landmarker picks out forehead and cheek polygons each
            frame. We average the red and green channels of pixels{" "}
            <em>inside</em> those polygons only — that throws away the
            background, hair, and clothing, and tracks the face when it moves.
          </p>
          <p>
            The chrominance signal{" "}
            <code className="font-mono text-neutral-200">G/⟨G⟩ − R/⟨R⟩</code>{" "}
            cancels common-mode lighting changes and highlights the small
            green-vs-red modulation caused by the pulse. It is resampled to{" "}
            {TARGET_FS} Hz, detrended, Hann-windowed and FFT-analysed. The peak
            in {HR_LO_HZ}–{HR_HI_HZ} Hz ({HR_LO_HZ * 60}–{HR_HI_HZ * 60} BPM) is
            reported as your heart rate.
          </p>
          <p>
            If the face detector fails (no permission, unsupported, etc.) the
            page falls back to averaging the whole frame and reports{" "}
            <span className="font-mono text-amber-300">whole frame</span>.
          </p>
        </div>
      </details>
    </div>
  );
}
