#!/usr/bin/env python
"""Compare car localization on R1 and R1-f using saved evaluator outputs.

This script does not run PETR.  It consumes ``results_pccr.json`` and the
validation annotation pickle, reproduces the evaluator's confidence-ordered,
one-to-one center-distance matching, and creates one focused figure.
"""

from __future__ import print_function

import argparse
import csv
import json
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


STANDARD_THRESHOLDS = (0.5, 1.0, 2.0, 4.0)


def parse_args():
    root = "results/petr_r50dcn_gridmask_p4_800x320_pccr/R1/cross_rig/evaluations"
    parser = argparse.ArgumentParser(
        description="Plot validation-set car AP and BEV-localization recall for R1/R1-f")
    parser.add_argument("--r1-results", default=os.path.join(
        root, "R1/formatted/results_pccr.json"))
    parser.add_argument("--r1f-results", default=os.path.join(
        root, "R1-f/formatted/results_pccr.json"))
    parser.add_argument("--r1-metrics", default=os.path.join(
        root, "R1/formatted/metrics_summary.json"))
    parser.add_argument("--r1f-metrics", default=os.path.join(
        root, "R1-f/formatted/metrics_summary.json"))
    # The saved cross-rig evaluation command uses *_infos_test.pkl.  These must
    # describe exactly the same sample tokens as results_pccr.json.
    parser.add_argument("--r1-ann", default="data/pccr/R1/R1_infos_test.pkl")
    parser.add_argument("--r1f-ann", default="data/pccr/R1-f/R1-f_infos_test.pkl")
    parser.add_argument("--score-threshold", type=float, default=0.35,
                        help="Minimum car confidence used for localization recall")
    parser.add_argument("--max-distance", type=float, default=10.0)
    parser.add_argument("--output-dir", default=(
        "experiments/cross_rig_attention/outputs/car_validation_localization"))
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def quaternion_matrix(q):
    """Return a 3x3 rotation matrix for a w,x,y,z quaternion."""
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),
         2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z),
         2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w),
         1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def lidar_to_global(points, info):
    points = np.asarray(points, dtype=np.float64)
    lidar2ego_r = quaternion_matrix(info["lidar2ego_rotation"])
    ego2global_r = quaternion_matrix(info["ego2global_rotation"])
    ego = np.dot(points, lidar2ego_r.T)
    ego += np.asarray(info["lidar2ego_translation"], dtype=np.float64)
    glob = np.dot(ego, ego2global_r.T)
    glob += np.asarray(info["ego2global_translation"], dtype=np.float64)
    return glob


def load_gt(annotation_path):
    with open(annotation_path, "rb") as handle:
        payload = pickle.load(handle)
    infos = sorted(payload["infos"], key=lambda item: item["timestamp"])
    gt_by_token = {}
    for info in infos:
        names = np.asarray(info["gt_names"])
        boxes = np.asarray(info["gt_boxes"])
        keep = names == "car"

        # This matches the PCCR evaluator's zero-point exclusion when point
        # counts are present, while remaining compatible with camera-only info.
        if "num_lidar_pts" in info:
            num_pts = np.asarray(info["num_lidar_pts"], dtype=np.int64)
            if "num_radar_pts" in info:
                num_pts = num_pts + np.asarray(info["num_radar_pts"], dtype=np.int64)
            keep = np.logical_and(keep, num_pts > 0)

        centers_lidar = boxes[keep, :3]
        # The official evaluator filters class range in ego coordinates.
        lidar2ego_r = quaternion_matrix(info["lidar2ego_rotation"])
        centers_ego = np.dot(centers_lidar, lidar2ego_r.T)
        centers_ego += np.asarray(info["lidar2ego_translation"], dtype=np.float64)
        in_range = np.linalg.norm(centers_ego[:, :2], axis=1) < 50.0
        gt_by_token[info["token"]] = lidar_to_global(
            centers_lidar[in_range], info)[:, :2]
    return gt_by_token


def load_predictions(result_path, score_threshold):
    with open(result_path, "r") as handle:
        payload = json.load(handle)
    predictions = []
    for token, boxes in payload["results"].items():
        for box in boxes:
            score = float(box["detection_score"])
            if box["detection_name"] != "car" or score < score_threshold:
                continue
            predictions.append((score, token, np.asarray(box["translation"][:2],
                                                          dtype=np.float64)))
    predictions.sort(key=lambda item: item[0], reverse=True)
    return predictions


def evaluator_recall(gt_by_token, predictions, distance_threshold):
    """Final recall from the same greedy matching order used for AP."""
    num_gt = sum(len(centers) for centers in gt_by_token.values())
    taken = set()
    distances = []
    for _, token, pred_center in predictions:
        gt_centers = gt_by_token.get(token)
        if gt_centers is None or len(gt_centers) == 0:
            continue
        available = [idx for idx in range(len(gt_centers)) if (token, idx) not in taken]
        if not available:
            continue
        local_distances = np.linalg.norm(gt_centers[available] - pred_center[None, :], axis=1)
        nearest_local = int(np.argmin(local_distances))
        distance = float(local_distances[nearest_local])
        if distance < distance_threshold:
            gt_index = available[nearest_local]
            taken.add((token, gt_index))
            distances.append(distance)
    recall = len(taken) / float(num_gt) if num_gt else 0.0
    return recall, distances, num_gt


def load_car_ap(path):
    with open(path, "r") as handle:
        metrics = json.load(handle)
    car_ap = metrics["label_aps"]["car"]
    return {float(key): float(value) for key, value in car_ap.items()}


def evaluate_condition(label, ann_path, result_path, metrics_path, score_threshold,
                       distance_grid):
    gt = load_gt(ann_path)
    predictions = load_predictions(result_path, score_threshold)
    gt_tokens = set(gt)
    prediction_tokens = set(item[1] for item in predictions)
    common_tokens = gt_tokens.intersection(prediction_tokens)
    if not common_tokens:
        raise ValueError(
            "{} has no common sample tokens between {} and {}. Use the exact "
            "annotation split used to generate the predictions.".format(
                label, ann_path, result_path))
    if common_tokens != gt_tokens or common_tokens != prediction_tokens:
        print("WARNING: {} token coverage: GT={}, predictions={}, common={}".format(
            label, len(gt_tokens), len(prediction_tokens), len(common_tokens)))
    recall = []
    for threshold in distance_grid:
        recall.append(evaluator_recall(gt, predictions, threshold)[0])
    standard_recall = {}
    for threshold in STANDARD_THRESHOLDS:
        standard_recall[threshold] = evaluator_recall(gt, predictions, threshold)[0]
    num_gt = sum(len(value) for value in gt.values())
    return {
        "label": label,
        "recall": np.asarray(recall),
        "standard_recall": standard_recall,
        "ap": load_car_ap(metrics_path),
        "num_gt": num_gt,
        "num_predictions": len(predictions),
    }


def save_data(output_dir, conditions, score_threshold):
    csv_path = os.path.join(output_dir, "car_threshold_summary.csv")
    with open(csv_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["condition", "distance_threshold_m", "localization_recall",
                         "official_ap", "score_threshold", "num_gt",
                         "num_car_predictions"])
        for condition in conditions:
            for threshold in STANDARD_THRESHOLDS:
                writer.writerow([
                    condition["label"], threshold,
                    condition["standard_recall"][threshold],
                    condition["ap"].get(threshold, float("nan")),
                    score_threshold, condition["num_gt"],
                    condition["num_predictions"],
                ])
    return csv_path


def plot(output_dir, distance_grid, conditions, score_threshold, dpi):
    colors = {"R1": "#2066b3", "R1-F": "#d1493f"}
    fig, axis = plt.subplots(figsize=(11.5, 7.0))
    for condition in conditions:
        label = condition["label"]
        color = colors[label]
        axis.plot(distance_grid, 100.0 * condition["recall"], color=color,
                  linewidth=2.8, label="{} localization recall".format(label))
        ap_x = np.asarray(STANDARD_THRESHOLDS)
        ap_y = 100.0 * np.asarray([condition["ap"].get(x, np.nan) for x in ap_x])
        axis.plot(ap_x, ap_y, color=color, linestyle="--", marker="o",
                  markersize=6.5, linewidth=1.8,
                  label="{} official car AP".format(label))

    for threshold in STANDARD_THRESHOLDS:
        axis.axvline(threshold, color="#777777", linestyle=":", linewidth=1.0,
                     alpha=0.65)
        axis.text(threshold, 99.0, "{} m".format(threshold), rotation=90,
                  ha="right", va="top", fontsize=9, color="#555555")

    axis.set_xlim(0.0, float(distance_grid[-1]))
    axis.set_ylim(0.0, 100.0)
    axis.set_xlabel("Allowed BEV center error (m)")
    axis.set_ylabel("GT-car recall / official AP (%)")
    axis.set_title("Does BEV localization explain the R1 to R1-F car AP collapse?\n"
                   "Localization recall uses class-correct predictions with score >= {:.2f}"
                   .format(score_threshold))
    axis.grid(True, alpha=0.22)
    axis.legend(loc="lower right", frameon=True)

    rows = []
    for threshold in STANDARD_THRESHOLDS:
        values = ["{}: recall {:.1f}%, AP {:.1f}%".format(
            condition["label"],
            100.0 * condition["standard_recall"][threshold],
            100.0 * condition["ap"].get(threshold, float("nan")))
            for condition in conditions]
        rows.append("{:g} m  |  {}".format(threshold, "    ".join(values)))
    fig.text(0.5, 0.015, "\n".join(rows), ha="center", va="bottom", fontsize=8.5,
             family="monospace")
    fig.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=0.19)
    output_path = os.path.join(output_dir, "car_bev_localization_vs_ap.png")
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def main():
    args = parse_args()
    required = (args.r1_results, args.r1f_results, args.r1_metrics,
                args.r1f_metrics, args.r1_ann, args.r1f_ann)
    missing = [path for path in required if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError("Missing required files: {}".format(", ".join(missing)))
    if args.max_distance <= 0:
        raise ValueError("--max-distance must be positive")
    os.makedirs(args.output_dir, exist_ok=True)

    distance_grid = np.linspace(0.01, args.max_distance, 250)
    conditions = [
        evaluate_condition("R1", args.r1_ann, args.r1_results, args.r1_metrics,
                           args.score_threshold, distance_grid),
        evaluate_condition("R1-F", args.r1f_ann, args.r1f_results, args.r1f_metrics,
                           args.score_threshold, distance_grid),
    ]
    csv_path = save_data(args.output_dir, conditions, args.score_threshold)
    plot_path = plot(args.output_dir, distance_grid, conditions,
                     args.score_threshold, args.dpi)
    print("Wrote {}".format(plot_path))
    print("Wrote {}".format(csv_path))
    for condition in conditions:
        print("{}: {} GT cars, {} predictions at score >= {:.2f}".format(
            condition["label"], condition["num_gt"], condition["num_predictions"],
            args.score_threshold))


if __name__ == "__main__":
    main()
