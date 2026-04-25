"""Generate 20 ROI comparison strips from the 7-scene x 4-model inference set.

Each strip is: NOISY | baseline_5min | lam01_10min | lam02_10min | lam05_10min,
2x nearest-neighbour zoomed for clear pixel-level inspection.

ROIs span:
  - lens shading corners (3)
  - smooth flats (concrete / sky / wall) (4)
  - building grids / blinds / windows (4)
  - foliage / hedges (3)
  - high-freq railings / spokes (3)
  - wood grain / brick patterns (2)
  - glass / signage (1)
"""
from __future__ import annotations
from pathlib import Path

import cv2
import numpy as np

SCENE_DIR = Path("experiments/_eval/multi_scene")
OUT_DIR = Path("experiments/_eval/rois_20")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    ("baseline_5min", "baseline 5min"),
    ("lam01_10min",   "lam=0.01  10min"),
    ("lam02_10min",   "lam=0.02  10min"),
    ("lam05_10min",   "lam=0.05  10min"),
]

# (scene, roi_name, x, y, w, h, category, comment)
# Image dimensions: 1088 wide x 1280 high.
ROIS = [
    # ---- changqiao_qiaodi (bridge bottom + bench + windows) ----
    ("changqiao_qiaodi",   "01_corner_TL_lens",     0,    0,  220, 220, "shading",  "extreme dark + lens shading"),
    ("changqiao_qiaodi",   "02_ceiling_concrete",   480,  50, 220, 160, "smooth",   "smooth concrete underside"),
    ("changqiao_qiaodi",   "03_window_grid",        50,  220, 280, 280, "grid",     "building window verticals"),
    ("changqiao_qiaodi",   "04_bench_wood_grain",   180, 480, 280, 220, "texture",  "wood grain mid-freq"),

    # ---- langan_louti (stairs + railing + sky + glass tower) ----
    ("langan_louti",       "05_railing_high_freq",  100, 550, 380, 280, "edges",    "diagonal stair railings - hardest test"),
    ("langan_louti",       "06_sky_smooth_flat",    300, 30,  280, 160, "smooth",   "sky flat region near horizon"),
    ("langan_louti",       "07_glass_tower_grid",   400, 80,  220, 280, "grid",     "glass tower window grid"),

    # ---- louti_shumu_caoguan (tree + shrubs + brick) ----
    ("louti_shumu_caoguan","08_tree_branches",      280, 100, 320, 280, "foliage",  "dense tree branches"),
    ("louti_shumu_caoguan","09_brick_pavement",     320, 800, 380, 280, "pattern",  "brick pavement regular grid"),
    ("louti_shumu_caoguan","10_dense_shrubs",       620, 280, 280, 220, "foliage",  "dense shrub leaves"),

    # ---- qiang_qiaodi (bridge curve + wall + dark corner) ----
    ("qiang_qiaodi",       "11_corner_BR_dark",     820, 250, 240, 280, "shading",  "deep dark corner"),
    ("qiang_qiaodi",       "12_wall_white_smooth",  340, 460, 320, 200, "smooth",   "flat white wall"),
    ("qiang_qiaodi",       "13_bridge_curve",       180, 100, 320, 220, "smooth",   "curved bridge underside"),

    # ---- shumu_guanmu_che (bikes + tree + tower blinds) ----
    ("shumu_guanmu_che",   "14_bicycle_cluster",    50,  580, 380, 240, "complex",  "dense parked bicycles"),
    ("shumu_guanmu_che",   "15_tower_blinds_vert",  680, 80,  360, 280, "grid",     "office tower vertical blinds"),
    ("shumu_guanmu_che",   "16_tree_trunk_dark",    300, 420, 220, 340, "texture",  "tree trunk + bark texture"),

    # ---- shumu_langan (glass facade + hedge) ----
    ("shumu_langan",       "17_glass_facade",       40,  50,  280, 320, "specular", "specular glass facade"),
    ("shumu_langan",       "18_hedge_texture",      300, 600, 380, 220, "foliage",  "hedge dense low-mid freq"),

    # ---- zixingche_shumu_jiedao (bicycles + signs) ----
    ("zixingche_shumu_jiedao","19_bicycle_spokes",  20,  830, 320, 320, "edges",    "bicycle spokes very thin"),
    ("zixingche_shumu_jiedao","20_shop_signs",      540, 380, 400, 260, "complex",  "shop windows + signage text"),
]

ZOOM = 2  # nearest-neighbour upscale factor


def label_panel(img: np.ndarray, text: str, height: int = 28) -> np.ndarray:
    header = np.zeros((height, img.shape[1], 3), dtype=np.uint8)
    cv2.putText(header, text, (8, height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    return np.concatenate([header, img], axis=0)


def crop(scene: str, model_key: str, x: int, y: int, w: int, h: int) -> np.ndarray:
    if model_key == "__noisy__":
        path = SCENE_DIR / scene / "noisy_isp.png"
    else:
        path = SCENE_DIR / scene / f"{model_key}_with_denoise.png"
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    H, W = img.shape[:2]
    x = max(0, min(x, W - w)); y = max(0, min(y, H - h))
    patch = img[y:y+h, x:x+w]
    big = cv2.resize(patch, (patch.shape[1] * ZOOM, patch.shape[0] * ZOOM),
                     interpolation=cv2.INTER_NEAREST)
    return big


def main():
    rows = []
    for scene, name, x, y, w, h, cat, comment in ROIS:
        cells = [label_panel(crop(scene, "__noisy__", x, y, w, h), "NOISY")]
        for model_key, label in MODELS:
            cells.append(label_panel(crop(scene, model_key, x, y, w, h), label))
        strip = np.concatenate(cells, axis=1)
        out_path = OUT_DIR / f"{name}.png"
        cv2.imwrite(str(out_path), strip)
        print(f"{name:40s} -> {out_path.name}  shape={strip.shape}  scene={scene}")
        rows.append((scene, name, cat, comment, out_path.name))
    print(f"\nWrote {len(rows)} ROI strips to {OUT_DIR}/")


if __name__ == "__main__":
    main()
