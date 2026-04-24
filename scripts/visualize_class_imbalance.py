"""
CamVid Dataset Class Imbalance Visualization
=============================================
Scans train/val/test indexed mask directories, counts per-class pixel frequencies,
and saves five separate publication-ready figures.

Expected project layout (run from scripts/):
    CS5242-Image-Semantic-Segmentation/
    ├── data/CamVid/
    ├── scripts/
    │   └── visualize_class_imbalance.py   ← this file
    └── ...

Usage (from the scripts/ directory):
    cd CS5242-Image-Semantic-Segmentation/scripts
    python visualize_class_imbalance.py

Output (created automatically inside scripts/outputs/):
    fig1_pixel_frequency.png     — horizontal bar: per-class pixel frequency
    fig2_log_scale.png           — vertical bar: log-scale pixel counts
    fig3_pie_topN_vs_rest.png    — pie: top-8 classes vs. the rest
    fig4_split_distribution.png  — stacked bar: class share per train/val/test split
    fig5_cumulative_coverage.png — line: cumulative pixel coverage curve
    class_stats.csv              — raw per-class statistics table
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from tqdm import tqdm

# ──────────────────────────────────────────────
# Path configuration
# ──────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DATA_ROOT      = os.path.join(PROJECT_ROOT, "data", "CamVid")
CLASS_DICT_CSV = os.path.join(DATA_ROOT, "class_dict.csv")
SPLITS = {
    "Train": os.path.join(DATA_ROOT, "train_labels_indexed"),
    "Val":   os.path.join(DATA_ROOT, "val_labels_indexed"),
    "Test":  os.path.join(DATA_ROOT, "test_labels_indexed"),
}
NUM_CLASSES = 32
VOID_INDEX  = 30      # "Void" class (RGB 0,0,0) — excluded from all plots
MASK_EXT    = ("*.png", "*.bmp")
OUTPUT_DIR  = os.path.join(SCRIPT_DIR, "outputs")

# ──────────────────────────────────────────────
# Shared style
# ──────────────────────────────────────────────
BG       = "white"
SPINE_C  = "#cccccc"
GRID_C   = "#eeeeee"
TEXT_C   = "#222222"
ACCENT   = "#d95f3b"   # warm red-orange  — highlights dominant classes
ACCENT2  = "#2a9d8f"   # teal             — highlights rare classes
ACCENT3  = "#e9c46a"   # amber

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor":   BG,
    "axes.edgecolor":   SPINE_C,
    "axes.labelcolor":  TEXT_C,
    "axes.titlecolor":  TEXT_C,
    "xtick.color":      TEXT_C,
    "ytick.color":      TEXT_C,
    "text.color":       TEXT_C,
    "grid.color":       GRID_C,
    "grid.linewidth":   0.8,
    "font.family":      "DejaVu Sans",
    "font.size":        10,
    "savefig.facecolor": BG,
    "savefig.dpi":      180,
})

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def load_class_dict(path: str) -> pd.DataFrame:
    """Load class_dict.csv; strip whitespace from column names and values."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df["name"] = df["name"].str.strip()
    df["r"] = df["r"].astype(int)
    df["g"] = df["g"].astype(int)
    df["b"] = df["b"].astype(int)
    return df


def count_pixels(mask_dir: str, num_classes: int) -> np.ndarray:
    """Count pixels per class index across all mask images in mask_dir."""
    counts = np.zeros(num_classes, dtype=np.int64)
    files = []
    for ext in MASK_EXT:
        files.extend(glob.glob(os.path.join(mask_dir, ext)))
    if not files:
        print(f"  [!] No mask files found in {mask_dir}")
        return counts
    for fpath in tqdm(files, desc=f"  {os.path.basename(mask_dir)}", leave=False):
        arr = np.array(Image.open(fpath), dtype=np.int32)
        for cls in range(num_classes):
            counts[cls] += int((arr == cls).sum())
    return counts


def pixel_frequency(counts: np.ndarray, exclude: int = VOID_INDEX) -> np.ndarray:
    """Return per-class pixel share (0–1) after zeroing out the excluded index."""
    c = counts.astype(np.float64).copy()
    c[exclude] = 0.0
    total = c.sum()
    return c / total if total > 0 else c


def imbalance_ratio(counts: np.ndarray, exclude: int = VOID_INDEX) -> float:
    """max / min pixel count among non-zero, non-void classes."""
    c = counts.astype(np.float64).copy()
    c[exclude] = 0.0
    valid = c[c > 0]
    return float(valid.max() / valid.min()) if len(valid) >= 2 else 1.0


def save(fig: plt.Figure, name: str) -> None:
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ──────────────────────────────────────────────
# Figure 1 — Per-class pixel frequency (horizontal bar)
# ──────────────────────────────────────────────

def plot_frequency_bar(vis_df: pd.DataFrame, vis_colors: list) -> None:
    n     = len(vis_df)
    freqs = vis_df["frequency"].values * 100
    y_pos = np.arange(n)

    fig, ax = plt.subplots(figsize=(12, 10))
    bars = ax.barh(y_pos, freqs, color=vis_colors, edgecolor="none", height=0.72)

    # Outline top-3 (dominant) and bottom-5 (rare) for emphasis
    order = np.argsort(freqs)[::-1]
    for i in order[:3]:
        bars[i].set_edgecolor(ACCENT)
        bars[i].set_linewidth(1.6)
    for i in order[-5:]:
        bars[i].set_edgecolor(ACCENT2)
        bars[i].set_linewidth(1.2)
        bars[i].set_alpha(0.65)

    # Inline value labels
    for bar, freq in zip(bars, freqs):
        label = f"{freq:.2f}%" if freq >= 0.05 else f"{freq:.3f}%"
        ax.text(
            bar.get_width() + freqs.max() * 0.005,
            bar.get_y() + bar.get_height() / 2,
            label, va="center", ha="left", fontsize=7.5, color="#555555",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(vis_df["class_name"].tolist(), fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Pixel share (%)", fontsize=10)
    ax.set_title(
        "CamVid — Per-Class Pixel Frequency\n(sorted descending, Void excluded)",
        fontsize=13, fontweight="bold", pad=14,
    )
    ax.set_xlim(0, freqs.max() * 1.16)
    ax.grid(axis="x", linestyle="--", alpha=0.6)
    ax.spines[["top", "right"]].set_visible(False)

    # Legend for border colours
    legend_handles = [
        mpatches.Patch(facecolor="white", edgecolor=ACCENT,  linewidth=1.5, label="Top-3 dominant classes"),
        mpatches.Patch(facecolor="white", edgecolor=ACCENT2, linewidth=1.2, label="Bottom-5 rare classes", alpha=0.65),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8.5,
              framealpha=0.9, edgecolor=SPINE_C)

    fig.tight_layout()
    save(fig, "fig1_pixel_frequency.png")


# ──────────────────────────────────────────────
# Figure 2 — Log-scale pixel count (vertical bar)
# ──────────────────────────────────────────────

def plot_log_scale(vis_df: pd.DataFrame, vis_colors: list) -> None:
    n        = len(vis_df)
    log_vals = np.log10(vis_df["total_pixels"].values + 1)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(np.arange(n), log_vals, color=vis_colors, edgecolor="none", width=0.8)

    # Reference horizontal lines
    for exp, label in [(3, "1 K"), (4, "10 K"), (5, "100 K"), (6, "1 M"), (7, "10 M")]:
        ax.axhline(exp, color=SPINE_C, linewidth=0.9, linestyle="--", zorder=0)
        ax.text(n - 0.3, exp + 0.04, label, fontsize=7, color="#888888", ha="right", va="bottom")

    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(vis_df["class_name"].tolist(), rotation=50, ha="right", fontsize=8)
    ax.set_ylabel("log₁₀ (pixel count)", fontsize=10)
    ax.set_title(
        "CamVid — Pixel Count per Class  (log scale)\n(Void excluded)",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(-0.6, n - 0.4)

    fig.tight_layout()
    save(fig, "fig2_log_scale.png")


# ──────────────────────────────────────────────
# Figure 3 — Pie: top-8 vs. rest
# ──────────────────────────────────────────────

def plot_pie(vis_df: pd.DataFrame, vis_colors: list) -> None:
    TOP_N    = 8
    n        = len(vis_df)
    top_df   = vis_df.iloc[:TOP_N]
    rest_pct = vis_df.iloc[TOP_N:]["frequency"].sum() * 100
    top_pcts = top_df["frequency"].values * 100

    pie_vals   = list(top_pcts) + [rest_pct]
    pie_colors = list(vis_colors[:TOP_N]) + ["#dddddd"]
    pie_labels = list(top_df["class_name"]) + [f"Other  ({n - TOP_N} classes)"]

    fig, ax = plt.subplots(figsize=(9, 7))
    wedges, _, autotexts = ax.pie(
        pie_vals,
        colors=pie_colors,
        autopct=lambda p: f"{p:.1f}%" if p > 2.5 else "",
        startangle=140,
        wedgeprops=dict(edgecolor="white", linewidth=1.8),
        pctdistance=0.80,
    )
    for at in autotexts:
        at.set_fontsize(8.5)
        at.set_color("#333333")

    legend_patches = [
        mpatches.Patch(color=pie_colors[i], label=pie_labels[i])
        for i in range(len(pie_labels))
    ]
    ax.legend(
        handles=legend_patches,
        loc="lower center", bbox_to_anchor=(0.5, -0.22),
        fontsize=8.5, framealpha=0.0,
        ncol=3, handlelength=1.1, handleheight=0.9,
    )
    ax.set_title(
        f"CamVid — Top-{TOP_N} Classes vs. Rest\n(share of total pixels, Void excluded)",
        fontsize=13, fontweight="bold", pad=16,
    )

    fig.tight_layout()
    save(fig, "fig3_pie_topN_vs_rest.png")


# ──────────────────────────────────────────────
# Figure 4 — Per-split stacked bar
# ──────────────────────────────────────────────

def plot_split_distribution(
    vis_df: pd.DataFrame,
    split_counts: dict,
    class_colors_norm: np.ndarray,
) -> None:
    split_names = list(split_counts.keys())
    TOP_N       = 12
    top_df      = vis_df.head(TOP_N)
    x           = np.arange(len(split_names))
    bottom      = np.zeros(len(split_names))

    fig, ax = plt.subplots(figsize=(7, 6))

    for _, row in top_df.iterrows():
        idx   = int(row["class_index"])
        color = tuple(class_colors_norm[idx])
        vals  = np.array([
            split_counts[s][idx] / max(split_counts[s].sum(), 1) * 100
            for s in split_names
        ])
        ax.bar(x, vals, bottom=bottom, color=color, edgecolor="none",
               label=row["class_name"], width=0.55)
        bottom += vals

    # Remaining classes grouped as "Other"
    other_vals = np.maximum(100 - bottom, 0)
    ax.bar(x, other_vals, bottom=bottom, color="#e0e0e0",
           edgecolor="none", label="Other", width=0.55)

    ax.set_xticks(x)
    ax.set_xticklabels(split_names, fontsize=11)
    ax.set_ylabel("Pixel share (%)", fontsize=10)
    ax.set_ylim(0, 108)
    ax.set_title(
        "CamVid — Class Distribution per Split\n(top-12 classes shown individually)",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.12),
        fontsize=7.5, framealpha=0.9, edgecolor=SPINE_C,
        ncol=7,  # 13 items (12 classes + Other) across 2 rows → ceil(13/2) = 7 per row
    )

    fig.tight_layout()
    # Extra bottom margin so the legend is not clipped
    fig.subplots_adjust(bottom=0.22)
    save(fig, "fig4_split_distribution.png")


# ──────────────────────────────────────────────
# Figure 5 — Cumulative pixel coverage
# ──────────────────────────────────────────────

def plot_cumulative(vis_df: pd.DataFrame) -> None:
    n          = len(vis_df)
    cumulative = np.cumsum(vis_df["frequency"].values) * 100
    x          = np.arange(1, n + 1)

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.fill_between(x, cumulative, color=ACCENT, alpha=0.12)
    ax.plot(x, cumulative, color=ACCENT, linewidth=2.2,
            marker="o", markersize=4, markeredgewidth=0)

    # Annotate 50 / 80 / 95 % thresholds
    for thresh, color, label in [
        (50, ACCENT3,  "50%"),
        (80, ACCENT2,  "80%"),
        (95, "#9b59b6", "95%"),
    ]:
        idx_reach = int(np.searchsorted(cumulative, thresh))
        if idx_reach < n:
            ax.axhline(thresh, linestyle="--", linewidth=1.0, color=color, alpha=0.8)
            ax.axvline(idx_reach + 1, linestyle=":", linewidth=0.9, color=color, alpha=0.6)
            ax.annotate(
                f"{label} coverage\n→ top-{idx_reach + 1} classes",
                xy=(idx_reach + 1, thresh),
                xytext=(idx_reach + 2.2, thresh - 8),
                fontsize=8, color=color,
                arrowprops=dict(arrowstyle="-", color=color, lw=0.8),
            )

    ax.set_xlabel("Number of classes  (sorted by pixel frequency, descending)", fontsize=10)
    ax.set_ylabel("Cumulative pixel coverage (%)", fontsize=10)
    ax.set_title(
        "CamVid — Cumulative Pixel Coverage\n(Void excluded)",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.set_xlim(1, n)
    ax.set_ylim(0, 104)
    ax.grid(linestyle="--", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    save(fig, "fig5_cumulative_coverage.png")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 1. Load class dictionary ──
    print("[*] Loading class dictionary …")
    class_df = load_class_dict(CLASS_DICT_CSV)
    class_names       = class_df["name"].tolist()
    class_rgb         = class_df[["r", "g", "b"]].values   # shape (32, 3)
    class_colors_norm = class_rgb / 255.0                   # shape (32, 3), range [0, 1]

    # ── 2. Count pixels per split ──
    print("[*] Counting pixels per class …")
    split_counts = {}
    for split_name, mask_dir in SPLITS.items():
        if os.path.isdir(mask_dir):
            print(f"  Split: {split_name}")
            split_counts[split_name] = count_pixels(mask_dir, NUM_CLASSES)
        else:
            print(f"  [!] Skipping missing directory: {mask_dir}")

    if not split_counts:
        raise RuntimeError("No valid split directories found. Check DATA_ROOT.")

    total_counts = sum(split_counts.values())

    # ── 3. Build statistics DataFrame and save CSV ──
    freq_total = pixel_frequency(total_counts)
    rows = []
    for idx in range(NUM_CLASSES):
        row = {
            "class_index":  idx,
            "class_name":   class_names[idx] if idx < len(class_names) else f"Class_{idx:02d}",
            "r": int(class_rgb[idx, 0]),
            "g": int(class_rgb[idx, 1]),
            "b": int(class_rgb[idx, 2]),
            "total_pixels": int(total_counts[idx]),
            "frequency":    float(freq_total[idx]),
        }
        for s, cnts in split_counts.items():
            row[f"pixels_{s.lower()}"] = int(cnts[idx])
        rows.append(row)

    stats_df = pd.DataFrame(rows)
    csv_path = os.path.join(OUTPUT_DIR, "class_stats.csv")
    stats_df.to_csv(csv_path, index=False)
    print(f"[*] Statistics saved → {csv_path}")

    # Exclude Void and sort by frequency for all plots
    vis_df = (
        stats_df[stats_df["class_index"] != VOID_INDEX]
        .copy()
        .sort_values("total_pixels", ascending=False)
        .reset_index(drop=True)
    )
    vis_colors = [tuple(class_colors_norm[idx]) for idx in vis_df["class_index"]]

    # ── 4. Render each figure separately ──
    print("[*] Rendering figures …")
    plot_frequency_bar(vis_df, vis_colors)
    plot_log_scale(vis_df, vis_colors)
    plot_pie(vis_df, vis_colors)
    plot_split_distribution(vis_df, split_counts, class_colors_norm)
    plot_cumulative(vis_df)

    print("\n[*] All done.  Outputs are in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()