"""Visual A/B helper: pull fixed ROIs from each experiment's denoised image
and stack them into one comparison grid per ROI, plus a noise-floor metric."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

EXP_ROOT = Path("experiments")
OUT_DIR = EXP_ROOT / "_eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Pre-chosen ROIs on the 1088x1280 ISP-style render (column, row, w, h).
# Each highlights a different failure mode the model could exhibit.
# Image dimensions are ~ width=1088, height=1280 (column,row format here).
ROIS = {
    "bench_wood":     (40,  580, 320, 220),  # diagonal wooden bench planks - texture
    "ceiling":        (480, 90,  280, 200),  # underside of canopy - flat/blob area
    "sky_corner":     (820, 30,  220, 300),  # bright sky / building edge
    "floor_planks":   (340, 920, 520, 240),  # wooden floor texture far-right
    "window_grid":    (40,  240, 360, 320),  # building reflections - edges
    "shrub":          (440, 470, 220, 180),  # foliage detail
    # Corner regions - lens shading is strong here, blob artifacts often
    # show up first. These are the user's main concern.
    "corner_TL":      (0,    0, 220, 220),   # top-left
    "corner_TR":      (868,  0, 220, 220),   # top-right
    "corner_BL":      (0, 1060, 220, 220),   # bottom-left
    "corner_BR":      (868,1060, 220, 220),  # bottom-right
}


def annotate(img: np.ndarray, text: str) -> np.ndarray:
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), thickness=-1)
    cv2.putText(img, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    return img


def noise_floor(img: np.ndarray, x: int, y: int, w: int, h: int) -> float:
    """High-frequency std (residual pixel-scale chroma/luma noise) on a flat
    region. Lower = smoother."""
    crop = img[y:y + h, x:x + w].astype(np.float32)
    blurred = cv2.boxFilter(crop, ddepth=-1, ksize=(5, 5))
    return float((crop - blurred).std(axis=(0, 1)).mean())


def blob_residual(img: np.ndarray, x: int, y: int, w: int, h: int) -> float:
    """Low-frequency *blob/banding* energy on a flat region.

    We isolate variation at the ~16-pixel scale by computing the std of
    ``boxFilter_15(crop) - boxFilter_31(crop)``: this band-pass keeps the
    typical "blob" frequency while removing both pixel noise and scene-scale
    gradient. Lower = fewer blob/banding artefacts.
    """
    crop = img[y:y + h, x:x + w].astype(np.float32)
    low = cv2.boxFilter(crop, ddepth=-1, ksize=(15, 15))
    very_low = cv2.boxFilter(crop, ddepth=-1, ksize=(31, 31))
    return float((low - very_low).std(axis=(0, 1)).mean())


def edge_sharpness(img: np.ndarray, x: int, y: int, w: int, h: int) -> float:
    """Mean Sobel gradient magnitude on a region containing real edges.
    Higher = sharper edges preserved."""
    crop = cv2.cvtColor(img[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(crop, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(crop, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.sqrt(gx * gx + gy * gy).mean())


def main():
    exp_dirs = sorted(p for p in EXP_ROOT.iterdir() if p.is_dir() and not p.name.startswith("_"))
    if not exp_dirs:
        print("No experiments/ dirs found")
        return
    rows: dict[str, list[tuple[str, np.ndarray]]] = {k: [] for k in ROIS}

    # Also include the noisy reference (same across all experiments).
    noisy = cv2.imread(str(exp_dirs[0] / "sensor2_isp_without_denoise.png"))
    if noisy is not None:
        for roi_name, (x, y, w, h) in ROIS.items():
            crop = noisy[y:y + h, x:x + w]
            rows[roi_name].append(("noisy", annotate(crop, f"noisy")))

    # Multiple flat-region samples (avoid lucky/unlucky single ROI).
    flat_rois = [
        ("ceiling_a",     520, 130, 80, 80),
        ("ceiling_b",     680, 200, 80, 80),
        ("floor_far",     820, 970, 80, 80),
        ("sky_top",       870,  60, 80, 80),
        # Lens-shading corner samples on relatively uniform sky / shadow
        ("corner_TL_dark",  20,   20, 100, 100),
        ("corner_TR_sky",  960,   20, 100, 100),
        ("corner_BL_dark",  20, 1140, 100, 100),
        ("corner_BR_dark", 960, 1140, 100, 100),
    ]
    edge_rois = [
        ("window_mullion", 100, 350, 200, 120),  # vertical lines on building
        ("bench_seam",     120, 720, 200, 120),  # bench plank seams
    ]

    metrics = []
    for d in exp_dirs:
        img = cv2.imread(str(d / "sensor2_isp_with_denoise.png"))
        if img is None:
            continue
        nfs = [noise_floor(img, x, y, w, h) for _, x, y, w, h in flat_rois]
        blobs = [blob_residual(img, x, y, w, h) for _, x, y, w, h in flat_rois]
        sharps = [edge_sharpness(img, x, y, w, h) for _, x, y, w, h in edge_rois]
        metrics.append({
            "name": d.name,
            "noise_floor_avg": float(np.mean(nfs)),
            "blob_residual_avg": float(np.mean(blobs)),
            "blob_residual_per": {n: round(v, 3) for (n, *_), v in zip(flat_rois, blobs)},
            "edge_sharpness_avg": float(np.mean(sharps)),
        })
        for roi_name, (x, y, w, h) in ROIS.items():
            crop = img[y:y + h, x:x + w]
            rows[roi_name].append((d.name, annotate(crop, d.name)))

    # Stack per ROI: [noisy | exp0 | exp1 | ...] horizontally, 2x NN-zoomed.
    for roi_name, items in rows.items():
        if not items:
            continue
        zoomed = []
        for tag, img in items:
            big = cv2.resize(img, (img.shape[1] * 2, img.shape[0] * 2),
                             interpolation=cv2.INTER_NEAREST)
            zoomed.append(big)
        stacked = np.concatenate(zoomed, axis=1)
        cv2.imwrite(str(OUT_DIR / f"roi_{roi_name}.png"), stacked)
        # Also write each strategy's ROI as a single 2x zoomed file for
        # closer inspection by name.
        roi_dir = OUT_DIR / f"roi_{roi_name}_separate"
        roi_dir.mkdir(parents=True, exist_ok=True)
        for (tag, img), big in zip(items, zoomed):
            cv2.imwrite(str(roi_dir / f"{tag}.png"), big)

    # Composite score:
    # - blob_residual is the user's main complaint -> 50% weight (lower=better)
    # - high-freq noise_floor                       -> 25% weight (lower=better)
    # - edge_sharpness                               -> 25% weight (higher=better)
    nf_arr = np.array([m["noise_floor_avg"] for m in metrics])
    bl_arr = np.array([m["blob_residual_avg"] for m in metrics])
    sh_arr = np.array([m["edge_sharpness_avg"] for m in metrics])

    def normalise_lower_better(arr):
        return (arr.max() - arr) / (arr.max() - arr.min() + 1e-9)

    def normalise_higher_better(arr):
        return (arr - arr.min()) / (arr.max() - arr.min() + 1e-9)

    nf_score = normalise_lower_better(nf_arr)
    bl_score = normalise_lower_better(bl_arr)
    sh_score = normalise_higher_better(sh_arr)
    composite = 0.5 * bl_score + 0.25 * nf_score + 0.25 * sh_score

    for m, nf_s, bl_s, sh_s, c in zip(metrics, nf_score, bl_score, sh_score, composite):
        m["smoothness_norm"] = float(nf_s)
        m["blobless_norm"] = float(bl_s)
        m["sharpness_norm"] = float(sh_s)
        m["composite_score"] = float(c)

    metrics_sorted = sorted(metrics, key=lambda m: m["composite_score"], reverse=True)
    import json
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics_sorted, indent=2))
    with (OUT_DIR / "metrics.txt").open("w", encoding="utf-8") as f:
        f.write(
            f"{'name':24s}  {'blob':>8s}  {'hf_noise':>8s}  {'edge_sh':>8s}  {'composite':>10s}\n"
        )
        f.write("-" * 76 + "\n")
        for m in metrics_sorted:
            line = (
                f"{m['name']:24s}  "
                f"{m['blob_residual_avg']:8.4f}  "
                f"{m['noise_floor_avg']:8.4f}  "
                f"{m['edge_sharpness_avg']:8.3f}  "
                f"{m['composite_score']:10.4f}\n"
            )
            f.write(line)
            print(line.rstrip())
    print(f"Wrote {len(rows)} ROI comparison panels to {OUT_DIR}/")


if __name__ == "__main__":
    main()
