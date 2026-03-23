"""
vis_result.py  ——  Visualize and compare training metrics from multiple DAB-DETR log files.

Usage:
    python tools/vis_result.py --logs output/autodl/fine_tune/log.txt
    python tools/vis_result.py --logs Baseline=output/baseline/log.txt FocalLoss=output/focal/log.txt --save plots/

Each PLOT_GROUP defines one figure with one or more sub-panels.
To add / remove metrics, only edit PLOT_GROUPS at the bottom of this file.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns

# 设置 seaborn 主题，确保符合学术论文美观、干净的风格，不紧凑
sns.set_theme(style="whitegrid", palette="deep", context="notebook", font_scale=1.1)

# ── COCO metric names (order matches test_coco_eval_bbox list) ──────────────
COCO_NAMES = [
    "AP@.5:.95",
    "AP@.50",
    "AP@.75",
    "AP_s",
    "AP_m",
    "AP_l",
    "AR@1",
    "AR@10",
    "AR@100",
    "AR_s",
    "AR_m",
    "AR_l",
]


# ───────────────────────────────────────────────────────────────────────────
# Data loading
# ───────────────────────────────────────────────────────────────────────────


def load_logs(log_args: list[str]) -> list[dict]:
    """
    Load multiple log files.
    Accepts paths or named paths like 'ExpName=path/to/log.txt'.
    Returns a list of dicts: [{'name': 'Exp1', 'records': [...]}, ...]
    """
    experiments = []
    for i, arg in enumerate(log_args):
        if "=" in arg:
            name, path = arg.split("=", 1)
        else:
            name = Path(arg).parent.name if Path(arg).parent.name else f"Exp_{i + 1}"
            path = arg

        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        records.sort(key=lambda r: r["epoch"])
        experiments.append({"name": name, "records": records, "path": path})
    return experiments


def extract(records: list[dict], key: str) -> tuple[list, list]:
    """Return (epochs, values) for a scalar metric key."""
    epochs, values = [], []
    for r in records:
        if key in r:
            epochs.append(r["epoch"])
            values.append(r[key])
    return epochs, values


def extract_coco(records: list[dict], idx: int) -> tuple[list, list]:
    """Return (epochs, values) for the i-th COCO bbox metric."""
    epochs, values = [], []
    for r in records:
        v = r.get("test_coco_eval_bbox")
        if v is not None and idx < len(v):
            epochs.append(r["epoch"])
            values.append(v[idx])
    return epochs, values


# ───────────────────────────────────────────────────────────────────────────
# Plot helpers
# ───────────────────────────────────────────────────────────────────────────


def _style_axes(ax, title: str, ylabel: str, grid: bool = True):
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Epoch", fontsize=11, fontweight="medium")
    ax.set_ylabel(ylabel, fontsize=11, fontweight="medium")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    # Grid styling is natively handled beautifully by seaborn 'whitegrid' theme


def plot_lines(ax, experiments: list[dict], series: list[dict]):
    """
    Draw multiple lines on *ax* using Seaborn.

    Each entry in *series* is a dict:
        label   : legend label
        key     : scalar key in log record  (mutually exclusive with coco_idx)
        coco_idx: integer index into test_coco_eval_bbox
        style   : optional dict passed to sns.lineplot()
    """
    is_multi_exp = len(experiments) > 1

    line_styles = ["-", "--", "-.", ":"]
    markers = ["o", "s", "^", "D", "v", "p", "*", "X", "<", ">"]

    # Use seaborn colour palette
    colors = sns.color_palette("deep")

    for exp_idx, exp in enumerate(experiments):
        records = exp["records"]
        exp_name = exp["name"]

        for s_idx, s in enumerate(series):
            if is_multi_exp:
                # Group by experiment colour:
                colour = colors[exp_idx % len(colors)]
                ls = line_styles[s_idx % len(line_styles)]
                mk = markers[s_idx % len(markers)]
                label = f"{exp_name} - {s['label']}"
            else:
                colour = s.get("colour", colors[s_idx % len(colors)])
                ls = s.get("style", {}).get("linestyle", "-")
                mk = s.get("style", {}).get("marker", markers[s_idx % len(markers)])
                label = s["label"]

            if "coco_idx" in s:
                x, y = extract_coco(records, s["coco_idx"])
            else:
                x, y = extract(records, s["key"])

            if x:
                sns.lineplot(
                    x=x,
                    y=y,
                    label=label,
                    color=colour,
                    linestyle=ls,
                    marker=mk,
                    markersize=3,
                    linewidth=1.5,
                    alpha=0.85,
                    ax=ax,
                )

    if ax.get_legend():
        ax.legend(fontsize=9, framealpha=0.95, loc="best", borderaxespad=0.5)


# ───────────────────────────────────────────────────────────────────────────
# Figure builders
# ───────────────────────────────────────────────────────────────────────────


def build_figure(experiments: list[dict], group: dict) -> plt.Figure:
    """
    Build one matplotlib Figure from a *group* descriptor.

    group keys:
        title    : figure-level suptitle
        panels   : list of panel descriptors
            title   : subplot title
            ylabel  : y-axis label
            series  : list of series dicts (see plot_lines)
        nrows/ncols: optional layout override (default: auto)
    """
    panels = group["panels"]
    n = len(panels)
    ncols = group.get("ncols", min(n, 3))
    nrows = group.get("nrows", (n + ncols - 1) // ncols)

    # Make figures slightly wider and taller so they aren't overly compact
    fig_w = max(5 * ncols, 9)
    fig_h = 4 * nrows + 1.0
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)

    exp_suffix = (
        " vs ".join(e["name"] for e in experiments)
        if len(experiments) > 1
        else experiments[0]["name"]
    )
    fig.suptitle(f"{group['title']} ({exp_suffix})", fontsize=15, fontweight="bold", y=1.02)

    for idx, panel in enumerate(panels):
        ax = axes[idx // ncols][idx % ncols]
        plot_lines(ax, experiments, panel["series"])
        _style_axes(ax, panel["title"], panel.get("ylabel", ""))

    # hide unused axes
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    # Add generous padding to avoid cramped layouts
    fig.tight_layout(pad=2.0, w_pad=2.0, h_pad=2.5)
    return fig


# ───────────────────────────────────────────────────────────────────────────
# ★  PLOT_GROUPS  ——  edit this section to customise / extend plots  ★
# ───────────────────────────────────────────────────────────────────────────

PLOT_GROUPS = [
    # ── Figure 1: overall loss ────────────────────────────────────────────
    {
        "title": "Overall Loss",
        "ncols": 2,
        "panels": [
            {
                "title": "Total Loss",
                "ylabel": "Loss",
                "series": [
                    {"label": "Train", "key": "train_loss"},
                    {"label": "Val", "key": "test_loss"},
                ],
            },
            {
                "title": "Learning Rate",
                "ylabel": "LR",
                "series": [
                    {"label": "LR", "key": "train_lr"},
                ],
            },
        ],
    },
    # ── Figure 2: component losses ────────────────────────────────────────
    {
        "title": "Component Losses",
        "ncols": 3,
        "panels": [
            {
                "title": "Classification Loss",
                "ylabel": "Loss",
                "series": [
                    {"label": "Train CE", "key": "train_loss_ce"},
                    {"label": "Val CE", "key": "test_loss_ce"},
                ],
            },
            {
                "title": "Bounding-Box L1 Loss",
                "ylabel": "Loss",
                "series": [
                    {"label": "Train BBox", "key": "train_loss_bbox"},
                    {"label": "Val BBox", "key": "test_loss_bbox"},
                ],
            },
            {
                "title": "GIoU Loss",
                "ylabel": "Loss",
                "series": [
                    {"label": "Train GIoU", "key": "train_loss_giou"},
                    {"label": "Val GIoU", "key": "test_loss_giou"},
                ],
            },
        ],
    },
    # ── Figure 3: COCO AP metrics ─────────────────────────────────────────
    {
        "title": "COCO Evaluation — Average Precision",
        "ncols": 3,
        "panels": [
            {
                "title": "AP (primary metrics)",
                "ylabel": "AP",
                "series": [
                    {"label": COCO_NAMES[0], "coco_idx": 0},
                    {"label": COCO_NAMES[1], "coco_idx": 1},
                    {"label": COCO_NAMES[2], "coco_idx": 2},
                ],
            },
            {
                "title": "AP by Object Size",
                "ylabel": "AP",
                "series": [
                    {"label": COCO_NAMES[3], "coco_idx": 3},
                    {"label": COCO_NAMES[4], "coco_idx": 4},
                    {"label": COCO_NAMES[5], "coco_idx": 5},
                ],
            },
            {
                "title": "AR (Average Recall)",
                "ylabel": "AR",
                "series": [
                    {"label": COCO_NAMES[6], "coco_idx": 6},
                    {"label": COCO_NAMES[7], "coco_idx": 7},
                    {"label": COCO_NAMES[8], "coco_idx": 8},
                    {"label": COCO_NAMES[9], "coco_idx": 9},
                    {"label": COCO_NAMES[10], "coco_idx": 10},
                    {"label": COCO_NAMES[11], "coco_idx": 11},
                ],
            },
        ],
    },
]


# ───────────────────────────────────────────────────────────────────────────
# Entry point
# ───────────────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(description="Visualise DAB-DETR training log")
    parser.add_argument(
        "--logs",
        nargs="+",
        help="Path to one or multiple log.txt files. Format: path/to/log.txt or ExpName:path/to/log.txt",
        required=True,
    )
    parser.add_argument(
        "--save",
        default="tools/plots",
        help="Directory to save PNG figures. Leave empty to display interactively.",
        required=False,
    )
    parser.add_argument("--dpi", type=int, default=300, help="DPI for saved figures (default: 300)")
    return parser.parse_args()


def main():
    args = parse_args()
    experiments = load_logs(args.logs)
    for exp in experiments:
        print(f"Loaded {len(exp['records'])} epoch(s) for '{exp['name']}' from '{exp['path']}'")

    save_dir = Path(args.save) if args.save else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    exp_names = "_vs_".join(e["name"].replace(" ", "-").replace("/", "-") for e in experiments)

    for i, group in enumerate(PLOT_GROUPS):
        fig = build_figure(experiments, group)
        if save_dir:
            fname = (
                save_dir
                / f"fig{i + 1}_{group['title'].replace(' ', '_').replace('/', '-')}_{exp_names}.png"
            )
            fig.savefig(fname, dpi=args.dpi, bbox_inches="tight")
            print(f"  Saved → {fname}")
        else:
            plt.show()
        plt.close(fig)


if __name__ == "__main__":
    main()
