#!/usr/bin/env python3
"""Visualize class AP across rigs and center-distance thresholds."""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_THRESHOLDS = (0.5, 1.0, 2.0, 4.0)
RIG_PREFIX_ORDER = ("R1-f", "R1", "R1-c6", "R1-c10", "R1-r", "R1-t")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot per-rig class AP at PCCR distance thresholds.")
    parser.add_argument("--input", required=True, type=Path,
                        help="Standardized trained_on_*.json file.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--class-name", default="car")
    parser.add_argument("--train-rig", default=None,
                        help="Rig to highlight as in-domain.")
    parser.add_argument("--thresholds", nargs="+", type=float,
                        default=list(DEFAULT_THRESHOLDS))
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def threshold_text(value):
    return "{:.1f}".format(float(value))


def natural_rig_key(rig):
    if rig in RIG_PREFIX_ORDER:
        return (0, RIG_PREFIX_ORDER.index(rig))
    if rig.startswith("R") and rig[1:].isdigit():
        return (1, int(rig[1:]))
    return (2, rig)


def load_rows(path, class_name, thresholds, train_rig):
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    results = payload.get("results")
    if not isinstance(results, dict):
        raise ValueError("{} has no results dictionary".format(path))
    if train_rig is None:
        train_rig = payload.get("trained_on")

    rigs = sorted(results, key=natural_rig_key)
    rows = []
    for rig in rigs:
        raw = results[rig]["metrics"]["raw"]
        aps = []
        for threshold in thresholds:
            key = "object/{}_ap_dist_{}".format(
                class_name, threshold_text(threshold))
            if key not in raw:
                raise KeyError("Missing {} for rig {}".format(key, rig))
            aps.append(float(raw[key]))
        rows.append({
            "rig": rig,
            "in_domain": rig == train_rig,
            "aps": aps,
            "mean_ap": float(np.mean(aps)),
            "mATE": float(raw.get("object/mATE", float("nan"))),
            "class_trans_err": float(raw.get(
                "object/{}_trans_err".format(class_name), float("nan"))),
        })
    return payload, rows, train_rig


def write_tables(rows, thresholds, class_name, output_dir):
    fields = ["rig", "in_domain"]
    fields += ["AP@{}m".format(threshold_text(t)) for t in thresholds]
    fields += ["mean_class_AP", "class_translation_error", "overall_mATE"]
    csv_path = output_dir / "{}_ap_by_rig_threshold.csv".format(class_name)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            record = {"rig": row["rig"], "in_domain": row["in_domain"]}
            for threshold, value in zip(thresholds, row["aps"]):
                record["AP@{}m".format(threshold_text(threshold))] = value
            record["mean_class_AP"] = row["mean_ap"]
            record["class_translation_error"] = row["class_trans_err"]
            record["overall_mATE"] = row["mATE"]
            writer.writerow(record)

    md_path = output_dir / "{}_ap_by_rig_threshold.md".format(class_name)
    with md_path.open("w", encoding="utf-8") as handle:
        header = ["Rig"] + ["AP@{}m".format(threshold_text(t))
                            for t in thresholds] + ["Mean AP"]
        handle.write("# {} localization AP across rigs\n\n".format(
            class_name.replace("_", " ").title()))
        handle.write("AP values are percentages. `*` marks the training rig.\n\n")
        handle.write("| " + " | ".join(header) + " |\n")
        handle.write("|---" + "|---:" * (len(header) - 1) + "|\n")
        for row in rows:
            rig = row["rig"] + (" *" if row["in_domain"] else "")
            values = [100.0 * value for value in row["aps"]]
            values.append(100.0 * row["mean_ap"])
            handle.write("| {} | {} |\n".format(
                rig, " | ".join("{:.2f}".format(value) for value in values)))
    return csv_path, md_path


def plot_heatmap(rows, thresholds, class_name, train_rig, output_dir, dpi):
    matrix = 100.0 * np.asarray([row["aps"] for row in rows])
    height = max(6.2, 0.48 * len(rows) + 2.0)
    fig, axis = plt.subplots(figsize=(9.2, height), dpi=dpi)
    image = axis.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0,
                        vmax=max(1.0, float(matrix.max())))
    axis.set_xticks(np.arange(len(thresholds)))
    axis.set_xticklabels(["{} m".format(threshold_text(t))
                          for t in thresholds], fontweight="bold")
    axis.set_yticks(np.arange(len(rows)))
    labels = [row["rig"] + ("  ★" if row["in_domain"] else "")
              for row in rows]
    axis.set_yticklabels(labels)
    axis.set_xlabel("Ground-plane center-distance threshold",
                    fontweight="bold", labelpad=10)
    axis.set_ylabel("Test camera rig", fontweight="bold", labelpad=10)
    axis.set_title(
        "{} localization across camera rigs".format(
            class_name.replace("_", " ").title()),
        fontsize=16, fontweight="bold", pad=28)
    axis.text(
        0.5, 1.015,
        "PETR trained on {} | AP (%) | ★ in-domain".format(train_rig),
        transform=axis.transAxes, ha="center", va="bottom",
        fontsize=10.5, color="#4B5563")

    midpoint = float(matrix.max()) * 0.52
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            color = "white" if value > midpoint else "#111827"
            axis.text(column_index, row_index, "{:.2f}".format(value),
                      ha="center", va="center", color=color,
                      fontsize=9.5, fontweight="bold")

    colorbar = fig.colorbar(image, ax=axis, fraction=0.035, pad=0.035)
    colorbar.set_label("AP (%)", rotation=270, labelpad=18,
                       fontweight="bold")
    for spine in axis.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    save(fig, output_dir / "{}_ap_by_rig_threshold_heatmap".format(class_name))


def plot_curves(rows, thresholds, class_name, train_rig, output_dir, dpi):
    fig, axis = plt.subplots(figsize=(10.5, 6.3), dpi=dpi)
    x = np.asarray(thresholds, dtype=np.float64)
    cmap = plt.get_cmap("tab20")
    for index, row in enumerate(rows):
        if row["in_domain"]:
            axis.plot(x, 100.0 * np.asarray(row["aps"]), color="#DC2626",
                      linewidth=3.2, marker="o", markersize=7,
                      label=row["rig"] + " (in-domain)", zorder=5)
        else:
            axis.plot(x, 100.0 * np.asarray(row["aps"]),
                      color=cmap(index % 20), linewidth=1.35,
                      marker="o", markersize=3.5, alpha=0.78,
                      label=row["rig"])
    axis.set_xticks(x)
    axis.set_xticklabels(["{} m".format(threshold_text(t))
                          for t in thresholds])
    axis.set_xlabel("Ground-plane center-distance threshold",
                    fontweight="bold")
    axis.set_ylabel("{} AP (%)".format(class_name.replace("_", " ").title()),
                    fontweight="bold")
    axis.set_title("Strict-to-coarse localization profile across rigs",
                   fontsize=15, fontweight="bold")
    axis.grid(alpha=0.25)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(ncol=2, fontsize=8.5, frameon=False,
                bbox_to_anchor=(1.02, 1.0), loc="upper left")
    fig.tight_layout()
    save(fig, output_dir / "{}_ap_by_rig_threshold_curves".format(class_name))


def save(figure, stem):
    figure.savefig(str(stem) + ".png", dpi=figure.dpi, bbox_inches="tight")
    figure.savefig(str(stem) + ".svg", bbox_inches="tight")
    plt.close(figure)


def main():
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _, rows, train_rig = load_rows(
        args.input, args.class_name, args.thresholds, args.train_rig)
    csv_path, md_path = write_tables(
        rows, args.thresholds, args.class_name, args.output_dir)
    plot_heatmap(rows, args.thresholds, args.class_name, train_rig,
                 args.output_dir, args.dpi)
    plot_curves(rows, args.thresholds, args.class_name, train_rig,
                args.output_dir, args.dpi)
    print("Plotted {} rigs for class '{}'.".format(len(rows), args.class_name))
    print("CSV: {}".format(csv_path))
    print("Markdown: {}".format(md_path))


if __name__ == "__main__":
    main()
