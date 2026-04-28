"""Aggregate per-frame stereo disparity into a single benchmark score.

Input:
    A directory of disparity PNGs named ``<scenario>_<gain>_<denoise>.png``
    where ``denoise ∈ {0, 1}`` and ``gain ∈ {1, 2, 3, 4, 6, 8, 10, 12, 14, 16}``
    (ISO = gain · 100). The PNG values are disparity · 100 in uint16.

What it computes:
    - For each scenario / ISO, EPE (mean abs diff vs that scenario's
      ``ISO=100, denoise=<same>`` reference image), in pixels.
    - "Score" = average EPE at ISO = HIGHLIGHT_ISO across all scenarios,
      taken on the *denoised* branch.
    - "QC" = average over scenarios of the cross-EPE between the
      ``ISO=100 / denoise=0`` and ``ISO=100 / denoise=1`` images. If the
      denoiser is well-behaved at clean ISO=100, this stays small (< 0.6).
      Failing QC means the score is suppressed.

A 3x3 ``Summary_Depth_EPE_Final.png`` plot (8 scenes + global average) is
saved alongside the structured result for human inspection.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np

HIGHLIGHT_ISO = 1000
MAX_ISO = 1600
QC_THRESHOLD = 0.6
Y_LIMIT = 1.5

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
    qc_pass: bool
    score: Optional[float]            # mean denoise EPE at HIGHLIGHT_ISO
    iso100_delta: Optional[float]     # mean cross-EPE at ISO=100
    qc_threshold: float = QC_THRESHOLD
    per_scene: list = field(default_factory=list)  # list of dict per scene
    avg: dict = field(default_factory=dict)
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
    gt0 = next((d for d in records if d["iso"] == 100 and d["denoise"] == 0), None)
    gt1 = next((d for d in records if d["iso"] == 100 and d["denoise"] == 1), None)
    img_gt0 = _load_disp(gt0["path"]) if gt0 else None
    img_gt1 = _load_disp(gt1["path"]) if gt1 else None
    if img_gt0 is None and img_gt1 is None:
        return {}

    table: dict[int, dict[int, float]] = {}
    for r in records:
        if r["iso"] > MAX_ISO:
            continue
        ref = img_gt1 if r["denoise"] == 1 else img_gt0
        if ref is None:
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

    if img_gt0 is not None and img_gt1 is not None:
        out["iso100_cross_epe"] = float(np.mean(np.abs(img_gt0 - img_gt1)) / 100.0)
    return out


def _global_average(per_scene: list[dict]) -> dict:
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
    cross = [d["iso100_cross_epe"] for d in per_scene if "iso100_cross_epe" in d]
    if cross:
        avg["iso100_cross_epe"] = float(np.mean(cross))
    return avg


def _epe_at(data: dict, iso: int) -> dict:
    out = {"orig": None, "dn": None}
    for x, y in zip(data.get("x_orig", []), data.get("y_orig", [])):
        if x == iso:
            out["orig"] = float(y); break
    for x, y in zip(data.get("x_dn", []), data.get("y_dn", [])):
        if x == iso:
            out["dn"] = float(y); break
    return out


def _plot_subplot(ax, data, title, is_average=False):
    if data["x_orig"]:
        ax.plot(data["x_orig"], data["y_orig"], label="原始",
                marker="o", color="blue", linewidth=1.5, markersize=5)
    if data["x_dn"]:
        ax.plot(data["x_dn"], data["y_dn"], label="降噪",
                marker="s", color="red", linewidth=1.5, markersize=5)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylim(0, Y_LIMIT)
    ax.set_xlabel("ISO", fontsize=9)
    ax.set_ylabel("EPE (Pixel)", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.6)

    e = _epe_at(data, HIGHLIGHT_ISO)
    if e["orig"] is not None or e["dn"] is not None:
        ax.axvline(HIGHLIGHT_ISO, color="gray", linestyle=":", alpha=0.7)
        for key, color in (("orig", "blue"), ("dn", "red")):
            if e[key] is not None:
                ax.scatter([HIGHLIGHT_ISO], [e[key]], color=color, s=60,
                           edgecolor="white", linewidth=1.2, zorder=5)

    cross = data.get("iso100_cross_epe")
    if cross is not None:
        ax.scatter([100], [cross], color="purple", marker="D", s=42,
                   edgecolor="white", linewidth=1.0, zorder=5,
                   label=f"ISO=100 降噪Δ={cross:.3f}")

    if is_average:
        for sp in ax.spines.values():
            sp.set_linewidth(1.5); sp.set_color("#333")
    ax.legend(fontsize=8, loc="upper left")


def _save_summary_plot(per_scene, avg, score, qc_pass, out_path: Path):
    fig, axes = plt.subplots(3, 3, figsize=(18, 14), constrained_layout=True)
    flat = axes.flatten()
    for i in range(8):
        if i < len(per_scene):
            _plot_subplot(flat[i], per_scene[i], f"场景 {i + 1}")
        else:
            flat[i].axis("off")
    _plot_subplot(flat[8], avg, "全局平均", is_average=True)

    if qc_pass and score is not None:
        title, color = f"评估指标: {score:.4f}", "black"
    else:
        title, color = (f"评估失败: ISO=100 降噪Δ ≥ {QC_THRESHOLD}", "#c00")
    fig.suptitle(title, fontsize=18, fontweight="bold", y=1.02, color=color)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def _print_iso_table(per_scene, avg, iso=HIGHLIGHT_ISO):
    print(f"\n=== ISO={iso} EPE (单位: Pixel) ===")
    header = f"{'场景':<10}{'原始':>10}{'降噪':>10}{'Δ(原-降)':>14}{'ISO=100 降噪Δ':>16}"
    print(header); print("-" * len(header))
    for i, d in enumerate(per_scene, 1):
        e = _epe_at(d, iso)
        o = f"{e['orig']:.4f}" if e['orig'] is not None else "  -  "
        x = f"{e['dn']:.4f}" if e['dn']   is not None else "  -  "
        diff = (f"{e['orig'] - e['dn']:+.4f}"
                if (e['orig'] is not None and e['dn'] is not None) else "   -   ")
        c = d.get("iso100_cross_epe")
        cs = f"{c:.4f}" if c is not None else "  -  "
        print(f"场景 {i:<6}{o:>10}{x:>10}{diff:>14}{cs:>16}")
    print("-" * len(header))
    e = _epe_at(avg, iso)
    o = f"{e['orig']:.4f}" if e['orig'] is not None else "  -  "
    x = f"{e['dn']:.4f}" if e['dn']   is not None else "  -  "
    diff = (f"{e['orig'] - e['dn']:+.4f}"
            if (e['orig'] is not None and e['dn'] is not None) else "   -   ")
    c = avg.get("iso100_cross_epe")
    cs = f"{c:.4f}" if c is not None else "  -  "
    print(f"{'全局平均':<10}{o:>10}{x:>10}{diff:>14}{cs:>16}")


def evaluate(
    depth_dir: str | Path,
    plot_dir: str | Path,
    *,
    quiet: bool = False,
) -> EvalResult:
    """Aggregate disparity PNGs in ``depth_dir`` into an EvalResult and save
    ``plot_dir/Summary_Depth_EPE_Final.png``."""
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

    if not quiet:
        print(f"处理场景数: {len(per_scene)} | 最大ISO限制: {MAX_ISO}")
        print("指标: EPE (Mean Absolute Error)")
        _print_iso_table(per_scene, avg, HIGHLIGHT_ISO)

    cross = avg.get("iso100_cross_epe")
    qc_pass = cross is not None and cross < QC_THRESHOLD
    score = _epe_at(avg, HIGHLIGHT_ISO)["dn"] if qc_pass else None

    plot_path = plot_dir / "Summary_Depth_EPE_Final.png"
    _save_summary_plot(per_scene, avg, score, qc_pass, plot_path)

    if not quiet:
        print(f"\n[完成] 图表已保存: {plot_path}")
        print("\n" + "=" * 50); print("最终结论"); print("=" * 50)
        if qc_pass:
            print(f"通过 QC (ISO=100 降噪Δ 平均 = {cross:.4f} < {QC_THRESHOLD})")
            if score is not None:
                print(f"评估指标 = {score:.4f}")
        else:
            cs = f"{cross:.4f}" if cross is not None else "N/A"
            print(f"未通过 QC (ISO=100 降噪Δ 平均 = {cs} ≥ {QC_THRESHOLD})")
            print("评估指标: 不输出")

    return EvalResult(
        qc_pass=qc_pass, score=score, iso100_delta=cross,
        per_scene=per_scene, avg=avg, plot_path=str(plot_path),
    )
