#!/usr/bin/env python3
"""Plot PCCR-reported versus newly trained PETR per-class AP.

Each class AP is the unweighted mean of the AP values at the requested ground-
plane center-distance thresholds (0.5, 1, 2, and 4 meters by default). The
overall rig mAP shown in the subtitle is read from the standardized result file.
"""

import argparse
import csv
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_CLASSES = (
    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
    "adult",
    "child",
    "traffic_light",
    "traffic_sign",
)

COLORS = {
    "reference": "#2b67e0",
    "new": "#ef7c31",
    "positive": "#1b9e77",
    "negative": "#d84a4a",
    "text": "#253047",
    "muted": "#637089",
    "grid": "#dae0ea",
    "background": "#ffffff",
    "plot_background": "#f8fafd",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare PCCR and newly trained PETR per-class AP."
    )
    parser.add_argument(
        "--reference",
        required=True,
        type=Path,
        help="PCCR standardized trained_on_R1.json result file.",
    )
    parser.add_argument(
        "--new",
        required=True,
        type=Path,
        help="New checkpoint standardized trained_on_R1.json result file.",
    )
    parser.add_argument(
        "--rigs",
        nargs="+",
        default=["R1"],
        help="Test rigs to plot. Use 'all' to plot every common rig.",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=list(DEFAULT_CLASSES),
        help="Object classes and display order.",
    )
    parser.add_argument(
        "--distance-thresholds",
        nargs="+",
        type=float,
        default=[0.5, 1.0, 2.0, 4.0],
        help="Center-distance AP thresholds to average for each class.",
    )
    parser.add_argument("--reference-label", default="PCCR-reported PETR")
    parser.add_argument("--new-label", default="Newly trained PETR")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/per_class_ap_comparison"),
    )
    parser.add_argument("--width", type=int, default=1800)
    parser.add_argument("--height", type=int, default=1120)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument(
        "--font-scale",
        type=float,
        default=1.0,
        help="Scale every plot font. Increase this if labels are still too small.",
    )
    parser.add_argument(
        "--save-pdf",
        action="store_true",
        help="Also save a vector PDF suitable for papers and presentations.",
    )
    return parser.parse_args()


def validate_args(args):
    if args.width < 1000 or args.height < 700:
        raise ValueError("--width must be >= 1000 and --height must be >= 700")
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")
    if args.font_scale <= 0:
        raise ValueError("--font-scale must be positive")
    if not args.classes:
        raise ValueError("At least one class is required")
    if not args.distance_thresholds:
        raise ValueError("At least one distance threshold is required")


def load_results(path):
    path = path.expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    results = payload.get("results")
    if not isinstance(results, dict):
        raise ValueError("{} does not contain a results dictionary".format(path))
    return path, payload, results


def threshold_key(threshold):
    return "{:.1f}".format(float(threshold))


def metric_key(class_name, threshold):
    return "object/{}_ap_dist_{}".format(class_name, threshold_key(threshold))


def class_threshold_aps(rig_result, class_name, thresholds):
    raw = rig_result["metrics"]["raw"]
    values = []
    for threshold in thresholds:
        key = metric_key(class_name, threshold)
        if key not in raw:
            raise KeyError("Missing metric: {}".format(key))
        values.append(float(raw[key]))
    return values


def nice_axis_max(values):
    maximum = max(values) if values else 1.0
    padded = max(5.0, maximum * 1.18)
    return math.ceil(padded / 5.0) * 5.0


def display_class_name(class_name):
    return class_name.replace("_", " ")


def draw_plot(
    output_path,
    rig,
    classes,
    thresholds,
    reference_class_ap,
    new_class_ap,
    reference_map,
    new_map,
    reference_label,
    new_label,
    width,
    height,
    dpi,
    font_scale,
    save_pdf,
):
    scale = float(font_scale)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12 * scale,
            "axes.labelsize": 15 * scale,
            "xtick.labelsize": 12 * scale,
            "ytick.labelsize": 13 * scale,
            "legend.fontsize": 13 * scale,
        }
    )

    figure_width = float(width) / float(dpi)
    figure_height = float(height) / float(dpi)
    figure, axis = plt.subplots(
        figsize=(figure_width, figure_height),
        dpi=dpi,
    )
    figure.patch.set_facecolor(COLORS["background"])

    y_values = np.arange(len(classes), dtype=np.float64)
    bar_height = 0.34
    reference_percent = 100.0 * np.asarray(reference_class_ap)
    new_percent = 100.0 * np.asarray(new_class_ap)
    delta = new_percent - reference_percent
    ap_axis_max = nice_axis_max(list(reference_percent) + list(new_percent))
    delta_column_x = ap_axis_max + 3.6
    full_axis_max = ap_axis_max + 9.0

    axis.set_facecolor(COLORS["plot_background"])
    reference_bars = axis.barh(
        y_values - bar_height / 2.0,
        reference_percent,
        height=bar_height,
        color=COLORS["reference"],
        label=reference_label,
        zorder=3,
    )
    new_bars = axis.barh(
        y_values + bar_height / 2.0,
        new_percent,
        height=bar_height,
        color=COLORS["new"],
        label=new_label,
        zorder=3,
    )
    axis.set_xlim(0, full_axis_max)
    axis.set_xticks(np.arange(0, ap_axis_max + 0.1, 5.0))
    axis.set_xlabel("Class AP (%)", color=COLORS["text"], fontweight="bold")
    axis.set_yticks(y_values)
    axis.set_yticklabels(
        [display_class_name(class_name) for class_name in classes],
        fontweight="bold",
        color=COLORS["text"],
    )
    axis.invert_yaxis()
    axis.grid(axis="x", color=COLORS["grid"], linewidth=1.1, zorder=0)
    axis.tick_params(axis="x", colors=COLORS["muted"])
    axis.tick_params(axis="y", length=0, pad=12)
    for spine in axis.spines.values():
        spine.set_visible(False)

    value_font_size = 10.5 * scale
    value_offset = 0.35
    for bars, color in (
        (reference_bars, COLORS["reference"]),
        (new_bars, COLORS["new"]),
    ):
        for bar in bars:
            value = bar.get_width()
            axis.text(
                value + value_offset,
                bar.get_y() + bar.get_height() / 2.0,
                "{:.2f}".format(value),
                ha="left",
                va="center",
                fontsize=value_font_size,
                fontweight="bold",
                color=color,
            )

    title = "PETR per-class AP on {} test set".format(rig)
    subtitle = (
        "Mean over center-distance thresholds {} m | Overall mAP: "
        "PCCR {:.2f}% vs new {:.2f}%"
    ).format(
        ", ".join(threshold_key(value) for value in thresholds),
        reference_map * 100.0,
        new_map * 100.0,
    )
    figure.suptitle(
        title,
        x=0.12,
        y=0.975,
        ha="left",
        va="top",
        fontsize=22 * scale,
        color=COLORS["text"],
        fontweight="bold",
    )
    figure.text(
        0.12,
        0.918,
        subtitle,
        ha="left",
        va="top",
        fontsize=12.5 * scale,
        color=COLORS["muted"],
    )
    handles, labels = axis.get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        frameon=False,
        ncol=2,
        bbox_to_anchor=(0.56, 0.884),
        columnspacing=2.2,
        handlelength=2.4,
    )

    delta_colors = [
        COLORS["positive"]
        if value > 0
        else COLORS["negative"]
        if value < 0
        else COLORS["muted"]
        for value in delta
    ]
    axis.axvline(
        ap_axis_max + 1.2,
        color=COLORS["grid"],
        linewidth=1.2,
        zorder=1,
    )
    axis.text(
        delta_column_x,
        -0.82,
        "Delta",
        ha="center",
        va="center",
        fontsize=11.5 * scale,
        fontweight="bold",
        color=COLORS["text"],
    )
    for y_value, value, color in zip(y_values, delta, delta_colors):
        axis.text(
            delta_column_x,
            y_value,
            "{:+.2f} pp".format(value),
            ha="center",
            va="center",
            fontsize=10.5 * scale,
            fontweight="bold",
            color=color,
        )

    figure.text(
        0.5,
        0.025,
        "Delta = newly trained minus PCCR-reported; positive values favor the new checkpoint.",
        ha="center",
        va="bottom",
        fontsize=10.5 * scale,
        color=COLORS["muted"],
    )
    figure.subplots_adjust(left=0.19, right=0.975, top=0.81, bottom=0.105)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(str(output_path), dpi=dpi, facecolor=COLORS["background"])
    if save_pdf:
        figure.savefig(
            str(output_path.with_suffix(".pdf")),
            facecolor=COLORS["background"],
        )
    plt.close(figure)


def write_csv(
    output_path,
    rig,
    classes,
    thresholds,
    reference_threshold_aps,
    new_threshold_aps,
):
    threshold_names = [threshold_key(value) for value in thresholds]
    fieldnames = ["rig", "class"]
    fieldnames.extend("pccr_ap_dist_{}".format(value) for value in threshold_names)
    fieldnames.append("pccr_class_ap")
    fieldnames.extend("new_ap_dist_{}".format(value) for value in threshold_names)
    fieldnames.extend(["new_class_ap", "delta_percentage_points"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for class_name in classes:
            reference_values = reference_threshold_aps[class_name]
            new_values = new_threshold_aps[class_name]
            reference_ap = sum(reference_values) / float(len(reference_values))
            new_ap = sum(new_values) / float(len(new_values))
            row = {"rig": rig, "class": class_name}
            for threshold, value in zip(threshold_names, reference_values):
                row["pccr_ap_dist_{}".format(threshold)] = value
            row["pccr_class_ap"] = reference_ap
            for threshold, value in zip(threshold_names, new_values):
                row["new_ap_dist_{}".format(threshold)] = value
            row["new_class_ap"] = new_ap
            row["delta_percentage_points"] = (new_ap - reference_ap) * 100.0
            writer.writerow(row)


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def main():
    args = parse_args()
    validate_args(args)
    reference_path, _, reference_results = load_results(args.reference)
    new_path, _, new_results = load_results(args.new)

    common_rigs = [rig for rig in reference_results if rig in new_results]
    if args.rigs == ["all"]:
        rigs = common_rigs
    else:
        rigs = args.rigs
        missing = [rig for rig in rigs if rig not in reference_results or rig not in new_results]
        if missing:
            raise KeyError(
                "Rig(s) missing from one or both result files: {}".format(
                    ", ".join(missing)
                )
            )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "reference": str(reference_path),
        "new": str(new_path),
        "rigs": rigs,
        "classes": args.classes,
        "distance_thresholds": args.distance_thresholds,
        "outputs": [],
    }

    for rig in rigs:
        reference_threshold_aps = {
            class_name: class_threshold_aps(
                reference_results[rig], class_name, args.distance_thresholds
            )
            for class_name in args.classes
        }
        new_threshold_aps = {
            class_name: class_threshold_aps(
                new_results[rig], class_name, args.distance_thresholds
            )
            for class_name in args.classes
        }
        reference_class_ap = [
            sum(reference_threshold_aps[class_name])
            / float(len(args.distance_thresholds))
            for class_name in args.classes
        ]
        new_class_ap = [
            sum(new_threshold_aps[class_name]) / float(len(args.distance_thresholds))
            for class_name in args.classes
        ]

        stem = "{}_per_class_ap_pccr_vs_new".format(safe_name(rig))
        plot_path = output_dir / "{}.png".format(stem)
        csv_path = output_dir / "{}.csv".format(stem)
        draw_plot(
            plot_path,
            rig,
            args.classes,
            args.distance_thresholds,
            reference_class_ap,
            new_class_ap,
            float(reference_results[rig]["metrics"]["mAP"]),
            float(new_results[rig]["metrics"]["mAP"]),
            args.reference_label,
            args.new_label,
            args.width,
            args.height,
            args.dpi,
            args.font_scale,
            args.save_pdf,
        )
        write_csv(
            csv_path,
            rig,
            args.classes,
            args.distance_thresholds,
            reference_threshold_aps,
            new_threshold_aps,
        )
        output_entry = {"rig": rig, "plot": plot_path.name, "csv": csv_path.name}
        if args.save_pdf:
            output_entry["pdf"] = plot_path.with_suffix(".pdf").name
        manifest["outputs"].append(output_entry)
        print("Wrote {}".format(plot_path))
        print("Wrote {}".format(csv_path))

    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print("Wrote {}".format(manifest_path))


if __name__ == "__main__":
    main()
