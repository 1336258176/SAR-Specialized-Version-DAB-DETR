"""
Extract academic CV metrics from DAB-DETR style log.txt files and generate
clean, publication-oriented plots.

Usage:
    python tools/plots/vis_result.py \
        -i output/exp1/log.txt output/exp2/log.txt \
        -o tools/plots/paper_figures

    python tools/plots/vis_result.py \
        -i Baseline=output/a/log.txt Ours=output/b/log.txt \
        -o tools/plots/paper_figures
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    raise SystemExit(
        "matplotlib is required for plotting. Please install it first, "
        "for example: pip install matplotlib"
    ) from exc


COCO_METRIC_NAMES = [
    "AP",
    "AP50",
    "AP75",
    "APs",
    "APm",
    "APl",
    "AR1",
    "AR10",
    "AR100",
    "ARs",
    "ARm",
    "ARl",
]


COCO_METRIC_DISPLAY_NAMES = [
    "mAP@[.50:.95]",
    r"AP$_{50}$",
    r"AP$_{75}$",
    r"AP$_S$",
    r"AP$_M$",
    r"AP$_L$",
    r"AR$_1$",
    r"AR$_{10}$",
    r"AR$_{100}$",
    r"AR$_S$",
    r"AR$_M$",
    r"AR$_L$",
]


KEY_METRICS = {
    "AP": 0,
    "AP50": 1,
    "AP75": 2,
    "APs": 3,
    "APm": 4,
    "APl": 5,
    "AR1": 6,
    "AR10": 7,
    "AR100": 8,
}


def set_paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.linewidth": 0.8,
            "lines.linewidth": 2.0,
            "lines.markersize": 4.5,
            "grid.linewidth": 0.5,
            "grid.alpha": 0.25,
            "savefig.dpi": 300,
            "figure.dpi": 150,
        }
    )


def parse_log_arg(arg: str, idx: int) -> Tuple[str, Path]:
    if "=" in arg:
        name, path = arg.split("=", 1)
    else:
        path = arg
        parent = Path(path).parent.name
        name = parent if parent else f"exp_{idx:02d}"
    return name, Path(path)


def load_records(log_path: Path) -> List[dict]:
    records = []
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    records.sort(key=lambda item: item.get("epoch", -1))
    return records


def build_experiments(log_args: List[str]) -> List[Dict]:
    experiments = []
    for idx, arg in enumerate(log_args):
        name, path = parse_log_arg(arg, idx)
        records = load_records(path)
        experiments.append(
            {
                "name": name,
                "path": str(path),
                "records": records,
            }
        )
    return experiments


def extract_scalar(records: List[dict], key: str) -> Tuple[List[int], List[float]]:
    xs, ys = [], []
    for item in records:
        if key in item:
            xs.append(item["epoch"])
            ys.append(item[key])
    return xs, ys


def extract_coco_metric(records: List[dict], metric_idx: int) -> Tuple[List[int], List[float]]:
    xs, ys = [], []
    for item in records:
        metrics = item.get("test_coco_eval_bbox")
        if metrics is not None and metric_idx < len(metrics):
            xs.append(item["epoch"])
            ys.append(metrics[metric_idx])
    return xs, ys


def summarize_experiment(exp: Dict) -> Dict:
    records = exp["records"]
    if not records:
        return {"name": exp["name"], "path": exp["path"]}

    ap_epochs, ap_values = extract_coco_metric(records, KEY_METRICS["AP"])
    if ap_values:
        best_idx = int(np.argmax(ap_values))
        best_epoch = ap_epochs[best_idx]
        best_ap = ap_values[best_idx]
        best_record = next(item for item in records if item["epoch"] == best_epoch)
    else:
        best_epoch = None
        best_ap = None
        best_record = records[-1]

    final_record = records[-1]
    summary = {
        "name": exp["name"],
        "path": exp["path"],
        "num_epochs": len(records),
        "best_epoch_by_AP": best_epoch,
        "best_AP": best_ap,
        # mAP in COCO-style logs is AP@[.5:.95], keep an explicit alias for clarity.
        "best_mAP": best_ap,
        "final_epoch": final_record.get("epoch"),
        "final_train_loss": final_record.get("train_loss"),
        "final_val_loss": final_record.get("test_loss"),
        "final_lr": final_record.get("train_lr"),
        "epoch_time": final_record.get("epoch_time"),
        "n_parameters": final_record.get("n_parameters"),
    }

    final_metrics = final_record.get("test_coco_eval_bbox")
    if final_metrics is not None and len(final_metrics) > KEY_METRICS["AP"]:
        summary["final_mAP"] = final_metrics[KEY_METRICS["AP"]]
    else:
        summary["final_mAP"] = None

    for metric_name, metric_idx in KEY_METRICS.items():
        metrics = best_record.get("test_coco_eval_bbox")
        if metrics is not None and metric_idx < len(metrics):
            summary[f"best_{metric_name}"] = metrics[metric_idx]
        else:
            summary[f"best_{metric_name}"] = None

    return summary


def save_summary(experiments: List[Dict], output_dir: Path) -> None:
    summaries = [summarize_experiment(exp) for exp in experiments]
    json_path = output_dir / "metrics_summary.json"
    csv_path = output_dir / "metrics_summary.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)

    fieldnames = list(summaries[0].keys()) if summaries else []
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)


def save_map_history(experiments: List[Dict], output_dir: Path) -> None:
    """
    Save per-epoch mAP/AP50/AP75 history for each experiment.
    mAP here follows COCO AP@[.5:.95].
    """
    csv_path = output_dir / "map_history.csv"
    fieldnames = ["name", "epoch", "mAP", "AP50", "AP75"]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for exp in experiments:
            for item in exp["records"]:
                metrics = item.get("test_coco_eval_bbox")
                if metrics is None:
                    continue
                row = {
                    "name": exp["name"],
                    "epoch": item.get("epoch"),
                    "mAP": metrics[0] if len(metrics) > 0 else None,
                    "AP50": metrics[1] if len(metrics) > 1 else None,
                    "AP75": metrics[2] if len(metrics) > 2 else None,
                }
                writer.writerow(row)


def format_axes(ax, xlabel: str = "Epoch", ylabel: str = "") -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="major", axis="both")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(x=0.03, y=0.08)


def save_figure(fig, output_dir: Path, stem: str, rect=None) -> None:
    fig.tight_layout(rect=rect)
    fig.savefig(output_dir / f"{stem}.png", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def add_shared_legend(fig, axes, n_items: int) -> None:
    handles, labels = axes[0].get_legend_handles_labels()
    if not handles:
        return
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=min(4, max(1, n_items)),
        frameon=False,
        handlelength=2.4,
        columnspacing=1.2,
    )


def plot_learning_curves(experiments: List[Dict], output_dir: Path) -> None:
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b", "#e377c2"]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.6))
    axes = axes.ravel()

    panels = [
        ("train_loss", "Train Loss"),
        ("test_loss", "Val Loss"),
        ("train_loss_bbox", "Train Box Loss"),
        ("train_loss_giou", "Train GIoU Loss"),
    ]

    for exp_idx, exp in enumerate(experiments):
        color = colors[exp_idx % len(colors)]
        for ax, (key, title) in zip(axes, panels):
            xs, ys = extract_scalar(exp["records"], key)
            if xs:
                ax.plot(xs, ys, color=color, label=exp["name"], marker="o", markevery=max(1, len(xs) // 10))
            ax.set_title(title)
            format_axes(ax, ylabel=title)

    add_shared_legend(fig, axes, len(experiments))
    save_figure(fig, output_dir, "learning_curves", rect=(0.0, 0.0, 1.0, 0.94))


def plot_learning_rate_panels(experiments: List[Dict], output_dir: Path) -> None:
    """
    Plot LR schedules in a single figure with separate panels per experiment.
    Each subplot contains only one experiment's LR curve.
    """
    if not experiments:
        return

    n = len(experiments)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 3.6 * nrows))
    axes = np.atleast_1d(axes).ravel()
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b", "#e377c2"]

    for idx, exp in enumerate(experiments):
        ax = axes[idx]
        color = colors[idx % len(colors)]
        xs, ys = extract_scalar(exp["records"], "train_lr")
        if xs:
            ax.plot(xs, ys, color=color, marker="o", markevery=max(1, len(xs) // 10))
        ax.set_title(exp["name"])
        format_axes(ax, ylabel="Learning Rate")

    # Hide unused axes when experiments do not fill the full grid.
    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    save_figure(fig, output_dir, "learning_rate_panels", rect=(0.0, 0.0, 1.0, 0.98))


def plot_detection_metrics(experiments: List[Dict], output_dir: Path) -> None:
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b", "#e377c2"]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.6))
    axes = axes.ravel()
    panels = [
        (COCO_METRIC_DISPLAY_NAMES[0], 0),
        (COCO_METRIC_DISPLAY_NAMES[1], 1),
        (COCO_METRIC_DISPLAY_NAMES[2], 2),
        (COCO_METRIC_DISPLAY_NAMES[8], 8),
    ]

    for exp_idx, exp in enumerate(experiments):
        color = colors[exp_idx % len(colors)]
        for ax, (title, metric_idx) in zip(axes, panels):
            xs, ys = extract_coco_metric(exp["records"], metric_idx)
            if xs:
                ax.plot(xs, ys, color=color, label=exp["name"], marker="o", markevery=max(1, len(xs) // 10))
            ax.set_title(title)
            format_axes(ax, ylabel=title)

    add_shared_legend(fig, axes, len(experiments))
    save_figure(fig, output_dir, "detection_metrics", rect=(0.0, 0.0, 1.0, 0.94))


def plot_recall_metrics(experiments: List[Dict], output_dir: Path) -> None:
    """
    Plot recall-centric curves while preserving the existing visual style.
    """
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b", "#e377c2"]
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 7.6))
    axes = axes.ravel()
    panels = [
        (COCO_METRIC_DISPLAY_NAMES[6], 6),
        (COCO_METRIC_DISPLAY_NAMES[7], 7),
        (COCO_METRIC_DISPLAY_NAMES[8], 8),
        (COCO_METRIC_DISPLAY_NAMES[9], 9),
        (COCO_METRIC_DISPLAY_NAMES[10], 10),
        (COCO_METRIC_DISPLAY_NAMES[11], 11),
    ]

    for exp_idx, exp in enumerate(experiments):
        color = colors[exp_idx % len(colors)]
        for ax, (title, metric_idx) in zip(axes, panels):
            xs, ys = extract_coco_metric(exp["records"], metric_idx)
            if xs:
                ax.plot(xs, ys, color=color, label=exp["name"], marker="o", markevery=max(1, len(xs) // 10))
            ax.set_title(title)
            format_axes(ax, ylabel=title)

    add_shared_legend(fig, axes, len(experiments))
    save_figure(fig, output_dir, "recall_metrics", rect=(0.0, 0.0, 1.0, 0.94))


def plot_coco_full_metrics(experiments: List[Dict], output_dir: Path) -> None:
    """
    Plot all COCO bbox metrics in a single 12-panel figure.
    """
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b", "#e377c2"]
    fig, axes = plt.subplots(3, 4, figsize=(15.2, 9.4))
    axes = axes.ravel()

    for exp_idx, exp in enumerate(experiments):
        color = colors[exp_idx % len(colors)]
        for metric_idx, metric_name in enumerate(COCO_METRIC_DISPLAY_NAMES):
            ax = axes[metric_idx]
            xs, ys = extract_coco_metric(exp["records"], metric_idx)
            if xs:
                ax.plot(xs, ys, color=color, label=exp["name"], marker="o", markevery=max(1, len(xs) // 10))
            ax.set_title(metric_name)
            format_axes(ax, ylabel=metric_name)

    add_shared_legend(fig, axes, len(experiments))
    save_figure(fig, output_dir, "coco_full_metrics", rect=(0.0, 0.0, 1.0, 0.95))


def plot_best_metric_bars(experiments: List[Dict], output_dir: Path) -> None:
    summaries = [summarize_experiment(exp) for exp in experiments]
    metric_names = ["best_mAP", "best_AP50", "best_AP75", "best_AR100"]
    display_names = [
        COCO_METRIC_DISPLAY_NAMES[0],
        COCO_METRIC_DISPLAY_NAMES[1],
        COCO_METRIC_DISPLAY_NAMES[2],
        COCO_METRIC_DISPLAY_NAMES[8],
    ]
    exp_names = [item["name"] for item in summaries]
    x = np.arange(len(exp_names))
    width = 0.18
    colors = ["#17becf", "#ff7f0e", "#2ca02c", "#9467bd"]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    max_value = 0.0
    for idx, (metric_name, display_name) in enumerate(zip(metric_names, display_names)):
        values = [item.get(metric_name) or 0.0 for item in summaries]
        if values:
            max_value = max(max_value, max(values))
        offset = (idx - (len(metric_names) - 1) / 2) * width
        bars = ax.bar(x + offset, values, width=width, label=display_name, color=colors[idx], alpha=0.9)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(0.005, 0.015 * max(max_value, 1e-6)),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
                clip_on=False,
            )

    ax.set_title("Best Detection Performance by Experiment")
    ax.set_xticks(x)
    ax.set_xticklabels(exp_names, rotation=15, ha="right")
    format_axes(ax, xlabel="Experiment", ylabel="Score")
    ax.legend(frameon=False, ncol=len(metric_names))
    ax.set_ylim(0, max(0.1, max_value * 1.22))
    save_figure(fig, output_dir, "best_metric_bars", rect=(0.0, 0.0, 1.0, 0.98))


def plot_scale_sensitivity(experiments: List[Dict], output_dir: Path) -> None:
    summaries = [summarize_experiment(exp) for exp in experiments]
    exp_names = [item["name"] for item in summaries]
    aps = [item.get("best_APs") or 0.0 for item in summaries]
    apm = [item.get("best_APm") or 0.0 for item in summaries]
    apl = [item.get("best_APl") or 0.0 for item in summaries]

    x = np.arange(len(exp_names))
    width = 0.24
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x - width, aps, width=width, label=COCO_METRIC_DISPLAY_NAMES[3], color="#4c78a8")
    ax.bar(x, apm, width=width, label=COCO_METRIC_DISPLAY_NAMES[4], color="#f58518")
    ax.bar(x + width, apl, width=width, label=COCO_METRIC_DISPLAY_NAMES[5], color="#54a24b")
    ax.set_title("Object Scale Sensitivity")
    ax.set_xticks(x)
    ax.set_xticklabels(exp_names, rotation=15, ha="right")
    format_axes(ax, xlabel="Experiment", ylabel="Average Precision")
    ax.legend(frameon=False)
    ax.set_ylim(0, max(0.1, max(aps + apm + apl) * 1.12 if (aps + apm + apl) else 0.1))
    save_figure(fig, output_dir, "scale_sensitivity", rect=(0.0, 0.0, 1.0, 0.98))


def print_console_summary(experiments: List[Dict]) -> None:
    summaries = [summarize_experiment(exp) for exp in experiments]
    print("=== Metrics Summary ===")
    for item in summaries:
        print(
            f"{item['name']}: "
            f"best mAP={item.get('best_mAP', 0):.4f}, "
            f"final mAP={item.get('final_mAP', 0):.4f}, "
            f"AP50={item.get('best_AP50', 0):.4f}, "
            f"AP75={item.get('best_AP75', 0):.4f}, "
            f"best epoch={item.get('best_epoch_by_AP')}"
        )


def get_args():
    parser = argparse.ArgumentParser(description="Extract CV metrics from log.txt and draw academic-style figures.")
    parser.add_argument("--logs", "-i", nargs="+", required=True, help="Paths like exp=output/xxx/log.txt or raw log path.")
    parser.add_argument("--output_dir", "-o", type=str, required=True, help="Directory to save csv/json/figures.")
    return parser.parse_args()


def main():
    args = get_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_paper_style()
    experiments = build_experiments(args.logs)
    save_summary(experiments, output_dir)
    save_map_history(experiments, output_dir)
    plot_learning_curves(experiments, output_dir)
    plot_learning_rate_panels(experiments, output_dir)
    plot_detection_metrics(experiments, output_dir)
    plot_recall_metrics(experiments, output_dir)
    plot_coco_full_metrics(experiments, output_dir)
    plot_best_metric_bars(experiments, output_dir)
    plot_scale_sensitivity(experiments, output_dir)
    print_console_summary(experiments)


if __name__ == "__main__":
    main()
