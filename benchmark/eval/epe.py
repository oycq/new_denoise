"""Aggregate per-frame stereo disparity into a single benchmark score.

Input
-----
A directory of disparity PNGs named ``<scenario>_<gain>_<denoise>.png`` where
``denoise ∈ {0, 1}`` and ``gain ∈ {1, 2, 3, 4, 6, 8, 10, 12, 14, 16}``
(ISO = gain · 100). PNG values are disparity · 100 in uint16.

Metric
------
Per scene, the **single baseline** is the clean ``(iso=100, denoise=0)``
disparity image. Every other frame in that scene — including the denoised
ISO=100 image — is compared against that one baseline:

    EPE(scene, iso, k) = mean | disp(scene, iso, k) - disp(scene, 100, 0) | / 100

The benchmark **score** is the mean of ``EPE(scene, iso, denoise=1)`` across
all scenes and all ISOs in ``[100..MAX_ISO]``. One number, lower is better.

The no-denoise EPE curve is still drawn on the summary plot for visual
context (it's the "what the depth net would do without us"), but it does
not enter the score.
"""
from __future__ import annotations

import glob
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np

MAX_ISO = 1600
Y_LIMIT = 1.5
BASELINE_ISO = 100


def _pick_cjk_fonts() -> list[str]:
    """Return matplotlib font-family names that actually resolve on this box.

    Hard-coding ``Noto Sans CJK SC`` is fragile: it lives inside a ``.ttc``
    container and matplotlib only indexes the first subface (typically ``JP``).
    Probe a handful of likely CJK candidates and keep whichever ones
    ``findfont`` resolves without falling back to the default.
    """
    import matplotlib.font_manager as fm

    candidates = [
        "WenQuanYi Micro Hei", "WenQuanYi Zen Hei",
        "Noto Sans CJK SC", "Noto Sans CJK JP",
        "Noto Sans CJK TC", "Noto Sans CJK HK",
        "Source Han Sans SC", "Source Han Sans CN",
        "AR PL UMing CN", "AR PL UKai CN",
        "Microsoft YaHei", "PingFang SC", "SimHei",
    ]
    found = []
    for name in candidates:
        try:
            fm.findfont(name, fallback_to_default=False)
            found.append(name)
        except Exception:
            pass
    return found + ["sans-serif"]


plt.rcParams["font.sans-serif"] = _pick_cjk_fonts()
plt.rcParams["axes.unicode_minus"] = False


@dataclass
class EvalResult:
    score: Optional[float]                          # mean denoised EPE
    per_scene: list = field(default_factory=list)   # list of dict per scene
    avg: dict = field(default_factory=dict)         # global mean curves
    plot_path: Optional[str] = None

    def to_dict(self):
        return asdict(self)


def _parse_filename(path: str):
    parts = Path(path).stem.split("_")
    if len(parts) < 3:
        return None
    try:
        gain = int(parts[-2])
        return {
            "scenario": "_".join(parts[:-2]),
            "iso": gain * 100,
            "denoise": int(parts[-1]),
            "path": path,
        }
    except ValueError:
        return None


def _load_disp(path: str) -> Optional[np.ndarray]:
    img = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
    return None if img is None else img.astype(np.float32)


def _epe(img: np.ndarray, gt: np.ndarray) -> float:
    return float(np.mean(np.abs((img - gt) / 100.0)))


def _process_scene(records: list[dict]) -> dict:
    """Compute EPE vs the single ``(iso=BASELINE_ISO, denoise=0)`` reference.

    Returns a dict with two parallel curves (orig and denoised) suitable
    for the per-scene plot, plus convenience scene-level means.
    """
    ref_rec = next(
        (r for r in records
         if r["iso"] == BASELINE_ISO and r["denoise"] == 0),
        None,
    )
    if ref_rec is None:
        return {}
    ref = _load_disp(ref_rec["path"])
    if ref is None:
        return {}

    table: dict[int, dict[int, float]] = {}
    for r in records:
        if r["iso"] > MAX_ISO:
            continue
        cur = _load_disp(r["path"])
        if cur is None:
            continue
        table.setdefault(r["iso"], {})[r["denoise"]] = _epe(cur, ref)

    isos = sorted(table)
    out = {"x_orig": [], "y_orig": [], "x_dn": [], "y_dn": []}
    for i in isos:
        if 0 in table[i]:
            out["x_orig"].append(i); out["y_orig"].append(table[i][0])
        if 1 in table[i]:
            out["x_dn"].append(i); out["y_dn"].append(table[i][1])

    out["mean_orig"] = (float(np.mean(out["y_orig"]))
                        if out["y_orig"] else None)
    out["mean_dn"]   = (float(np.mean(out["y_dn"]))
                        if out["y_dn"] else None)
    return out


def _global_average(per_scene: list[dict]) -> dict:
    """Average EPE per ISO across all scenes (for the 'global' subplot)."""
    bucket: dict[int, dict[int, list[float]]] = {}
    for d in per_scene:
        for x, y in zip(d["x_orig"], d["y_orig"]):
            bucket.setdefault(x, {0: [], 1: []})[0].append(y)
        for x, y in zip(d["x_dn"], d["y_dn"]):
            bucket.setdefault(x, {0: [], 1: []})[1].append(y)
    avg = {"x_orig": [], "y_orig": [], "x_dn": [], "y_dn": []}
    for k in sorted(bucket):
        if bucket[k][0]:
            avg["x_orig"].append(k); avg["y_orig"].append(float(np.mean(bucket[k][0])))
        if bucket[k][1]:
            avg["x_dn"].append(k); avg["y_dn"].append(float(np.mean(bucket[k][1])))
    avg["mean_orig"] = (float(np.mean(avg["y_orig"]))
                        if avg["y_orig"] else None)
    avg["mean_dn"]   = (float(np.mean(avg["y_dn"]))
                        if avg["y_dn"] else None)
    return avg


def _score_from_per_scene(per_scene: list[dict]) -> Optional[float]:
    """Pool every denoised EPE across all scenes & ISOs into one mean.

    This is *not* the same as ``mean(scene_means)`` if scenes have different
    ISO coverage; flattening gives every (scene, iso) sample equal weight,
    which matches the user-facing definition: "average diff across denoised
    ISO 100..1600".
    """
    samples = []
    for s in per_scene:
        samples.extend(s["y_dn"])
    return float(np.mean(samples)) if samples else None


def _plot_subplot(ax, data, title, is_average=False):
    if data["x_orig"]:
        ax.plot(data["x_orig"], data["y_orig"], label="无降噪",
                marker="o", color="blue", linewidth=1.5, markersize=5)
    if data["x_dn"]:
        ax.plot(data["x_dn"], data["y_dn"], label="降噪",
                marker="s", color="red", linewidth=1.5, markersize=5)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylim(0, Y_LIMIT)
    ax.set_xlabel("ISO", fontsize=9)
    ax.set_ylabel("EPE vs ISO=100 无降噪 (Pixel)", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.6)

    mean_dn = data.get("mean_dn")
    if mean_dn is not None:
        ax.axhline(mean_dn, color="red", linestyle=":", linewidth=1, alpha=0.6)
        ax.text(0.98, 0.97, f"降噪平均\n{mean_dn:.4f}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                          edgecolor="#888", alpha=0.9))

    if is_average:
        for sp in ax.spines.values():
            sp.set_linewidth(1.5); sp.set_color("#333")
    ax.legend(fontsize=8, loc="upper left")


def _save_summary_plot(per_scene, avg, score, out_path: Path):
    fig, axes = plt.subplots(3, 3, figsize=(18, 14), constrained_layout=True)
    flat = axes.flatten()
    for i in range(8):
        if i < len(per_scene):
            _plot_subplot(flat[i], per_scene[i], f"场景 {i + 1}")
        else:
            flat[i].axis("off")
    _plot_subplot(flat[8], avg, "全局平均", is_average=True)

    title = (f"评估指标: {score:.4f}"
             if score is not None else "评估指标: 无数据")
    fig.suptitle(title, fontsize=18, fontweight="bold", y=1.02)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def _print_summary(per_scene: list[dict], avg: dict, score: Optional[float]):
    print(f"\n=== EPE vs ISO=100 无降噪 基准 (ISO 100..{MAX_ISO} 平均) ===")
    header = f"{'场景':<10}{'无降噪':>10}{'降噪':>10}{'Δ(无降噪-降噪)':>18}"
    print(header); print("-" * len(header))
    for i, d in enumerate(per_scene, 1):
        mo = d.get("mean_orig"); md = d.get("mean_dn")
        o = f"{mo:.4f}" if mo is not None else "  -  "
        x = f"{md:.4f}" if md is not None else "  -  "
        diff = f"{mo - md:+.4f}" if (mo is not None and md is not None) else "   -   "
        print(f"场景 {i:<6}{o:>10}{x:>10}{diff:>18}")
    print("-" * len(header))
    mo = avg.get("mean_orig"); md = avg.get("mean_dn")
    o = f"{mo:.4f}" if mo is not None else "  -  "
    x = f"{md:.4f}" if md is not None else "  -  "
    diff = f"{mo - md:+.4f}" if (mo is not None and md is not None) else "   -   "
    print(f"{'全局平均':<10}{o:>10}{x:>10}{diff:>18}")


def evaluate(
    depth_dir: str | Path,
    plot_dir: str | Path,
    *,
    quiet: bool = False,
) -> EvalResult:
    """Aggregate disparity PNGs in ``depth_dir`` into an :class:`EvalResult`
    and save ``plot_dir/Summary_Depth_EPE_Final.png``.
    """
    depth_dir = Path(depth_dir)
    plot_dir  = Path(plot_dir)
    if not depth_dir.exists():
        raise FileNotFoundError(f"depth dir missing: {depth_dir}")

    records = []
    for f in glob.glob(str(depth_dir / "*.png")):
        info = _parse_filename(f)
        if info: records.append(info)
    if not records:
        raise RuntimeError(f"no parseable PNGs under {depth_dir}")

    scenarios = sorted({r["scenario"] for r in records})
    per_scene = []
    for s in scenarios:
        d = _process_scene([r for r in records if r["scenario"] == s])
        if d:
            per_scene.append(d)
    avg = _global_average(per_scene)
    score = _score_from_per_scene(per_scene)

    plot_path = plot_dir / "Summary_Depth_EPE_Final.png"
    _save_summary_plot(per_scene, avg, score, plot_path)

    if not quiet:
        print(f"处理场景数: {len(per_scene)} | 最大ISO: {MAX_ISO}")
        print(f"基准: 每个场景的 (ISO={BASELINE_ISO}, denoise=0)")
        _print_summary(per_scene, avg, score)
        print(f"\n[完成] 图表已保存: {plot_path}")
        print("\n" + "=" * 50); print("最终结论"); print("=" * 50)
        if score is not None:
            print(f"评估指标 = {score:.4f}")
        else:
            print("评估指标: 无数据")

    return EvalResult(
        score=score, per_scene=per_scene, avg=avg, plot_path=str(plot_path),
    )
