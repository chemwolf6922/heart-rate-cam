// MediaPipe Face Landmarker integration for the heart-rate webapp.
//
// Loads the WASM and model bundled in /public, exposes a small wrapper, and
// provides the same FOREHEAD/CHEEK landmark indices used in the Python
// experiment so the webapp processes the same skin patches.

import {
  FaceLandmarker,
  FilesetResolver,
  type NormalizedLandmark,
} from "@mediapipe/tasks-vision";

// When deployed to GitHub Pages as a project site, public assets live under
// /<repo-name>/.  Next.js rewrites <Image>/<Link> URLs but not raw fetches, so
// we have to prepend the base path manually for the WASM and model files.
const BASE_PATH = process.env.NEXT_PUBLIC_BASE_PATH ?? "";

// Same indices we used (and verified) in experiments/extract_signal_roi.py.
export const FOREHEAD_IDS = [
  67, 109, 10, 338, 297, 332, 333, 334, 296, 336, 9, 107, 66, 105, 104, 103,
];
export const LEFT_CHEEK_IDS = [
  116, 117, 118, 119, 100, 142, 203, 206, 207, 187, 123,
];
export const RIGHT_CHEEK_IDS = [
  345, 346, 347, 348, 329, 371, 423, 426, 427, 411, 352,
];

/** Polygons (in pixel coords) for the three skin patches we sample. */
export interface RoiPolygons {
  forehead: Array<[number, number]>;
  leftCheek: Array<[number, number]>;
  rightCheek: Array<[number, number]>;
}

/** Convert a list of normalized landmarks to integer pixel coords. */
export function landmarksToPolygon(
  lms: NormalizedLandmark[],
  ids: number[],
  width: number,
  height: number,
): Array<[number, number]> {
  return ids.map((i) => {
    const lm = lms[i];
    return [Math.round(lm.x * width), Math.round(lm.y * height)] as [number, number];
  });
}

/** Build all three ROI polygons from a landmark list and an image size. */
export function buildRoiPolygons(
  lms: NormalizedLandmark[],
  width: number,
  height: number,
): RoiPolygons {
  return {
    forehead: landmarksToPolygon(lms, FOREHEAD_IDS, width, height),
    leftCheek: landmarksToPolygon(lms, LEFT_CHEEK_IDS, width, height),
    rightCheek: landmarksToPolygon(lms, RIGHT_CHEEK_IDS, width, height),
  };
}

let landmarkerSingleton: FaceLandmarker | null = null;
let loadingPromise: Promise<FaceLandmarker> | null = null;

/** Lazy-load the FaceLandmarker. Self-hosted WASM and model from /public. */
export async function loadFaceLandmarker(): Promise<FaceLandmarker> {
  if (landmarkerSingleton) return landmarkerSingleton;
  if (loadingPromise) return loadingPromise;

  loadingPromise = (async () => {
    const fileset = await FilesetResolver.forVisionTasks(`${BASE_PATH}/mediapipe-wasm`);
    const lm = await FaceLandmarker.createFromOptions(fileset, {
      baseOptions: {
        modelAssetPath: `${BASE_PATH}/face_landmarker.task`,
        delegate: "GPU",
      },
      runningMode: "VIDEO",
      numFaces: 1,
      outputFaceBlendshapes: false,
      outputFacialTransformationMatrixes: false,
    });
    landmarkerSingleton = lm;
    return lm;
  })();
  return loadingPromise;
}

/**
 * Compute mean R, G, B inside the union of the three ROI polygons.
 *
 * Strategy: rasterize the polygons onto a small mask canvas (same size as the
 * sample canvas), then iterate (subsampled) over the sample-canvas pixels and
 * sum where the mask alpha is non-zero. This is much faster than per-pixel
 * point-in-polygon tests.
 */
export function meanRgbInRoi(
  sampleCtx: CanvasRenderingContext2D,
  maskCtx: CanvasRenderingContext2D,
  roi: RoiPolygons,
  width: number,
  height: number,
  stride: number = 2,
): { r: number; g: number; b: number; count: number } | null {
  // Draw mask.
  maskCtx.clearRect(0, 0, width, height);
  maskCtx.fillStyle = "#fff";
  for (const poly of [roi.forehead, roi.leftCheek, roi.rightCheek]) {
    if (poly.length < 3) continue;
    maskCtx.beginPath();
    maskCtx.moveTo(poly[0][0], poly[0][1]);
    for (let i = 1; i < poly.length; i++) maskCtx.lineTo(poly[i][0], poly[i][1]);
    maskCtx.closePath();
    maskCtx.fill();
  }

  const sample = sampleCtx.getImageData(0, 0, width, height).data;
  const mask = maskCtx.getImageData(0, 0, width, height).data;
  let r = 0;
  let g = 0;
  let b = 0;
  let n = 0;
  for (let y = 0; y < height; y += stride) {
    const row = y * width * 4;
    for (let x = 0; x < width; x += stride) {
      const i = row + x * 4;
      // Mask was drawn as opaque white; alpha channel encodes presence.
      if (mask[i + 3] > 128) {
        r += sample[i];
        g += sample[i + 1];
        b += sample[i + 2];
        n++;
      }
    }
  }
  if (n === 0) return null;
  return { r: r / n, g: g / n, b: b / n, count: n };
}
