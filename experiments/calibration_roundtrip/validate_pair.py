#!/usr/bin/env python
"""Visualize and numerically validate PCCR calibration for a paired rig frame.

For every GT gravity center visible in a camera, the script projects the point
with lidar2img and back-projects the resulting pixel with its camera-z depth.
It reports the 3D round-trip error, draws image-space GT boxes/centers, and
draws BEV camera rays, original centers, and reconstructed centers for R1 and
R1-f. Image overlays are important because an algebraic round trip alone
cannot detect a consistently wrong coordinate convention.
"""

import argparse
import csv
import importlib
import json
import os
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mmcv import Config  # noqa: E402
from mmdet3d.datasets import build_dataset  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EDGES = ((0, 1), (1, 2), (2, 3), (3, 0),
         (4, 5), (5, 6), (6, 7), (7, 4),
         (0, 4), (1, 5), (2, 6), (3, 7))
COLORS = ((31, 119, 180), (255, 127, 14), (44, 160, 44),
          (214, 39, 40), (148, 103, 189), (140, 86, 75),
          (227, 119, 194), (127, 127, 127), (188, 189, 34))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(
        REPO_ROOT / "projects/configs/petr/"
        "petr_r50dcn_gridmask_p4_800x320_pccr.py"))
    parser.add_argument("--scene-name", required=True)
    parser.add_argument("--scene-frame-index", required=True, type=int)
    parser.add_argument("--rig-a", default="R1")
    parser.add_argument("--rig-b", default="R1-f")
    parser.add_argument("--split", default="val")
    parser.add_argument("--output-dir", default=str(
        REPO_ROOT / "experiments/calibration_roundtrip/outputs"))
    parser.add_argument("--max-match-distance", type=float, default=0.25)
    parser.add_argument("--bev-limit", type=float, default=55.0)
    return parser.parse_args()


def import_plugin(cfg):
    if cfg.get("plugin", False):
        importlib.import_module(
            os.path.dirname(cfg.get("plugin_dir", "")).replace("/", "."))


def make_dataset(cfg, rig, split):
    root = REPO_ROOT / "data" / "pccr" / rig
    ann = root / "{}_infos_{}.pkl".format(rig, split)
    if not ann.is_file():
        raise FileNotFoundError(str(ann))
    data_cfg = cfg.data.test.copy()
    data_cfg.data_root = str(root) + "/"
    data_cfg.ann_file = str(ann)
    data_cfg.test_mode = True
    dataset = build_dataset(data_cfg)
    relocate_paths(dataset, root)
    return dataset


def relocate_paths(dataset, root):
    root = Path(root).resolve()
    for info in dataset.data_infos:
        for camera in info.get("cams", {}).values():
            value = camera.get("data_path")
            if not value or os.path.exists(value):
                continue
            normalized = value.replace("\\", "/")
            candidates = [root / normalized.lstrip("/")]
            for directory in ("samples", "sweeps"):
                marker = "/{}/".format(directory)
                if marker in normalized:
                    candidates.append(
                        root / directory / normalized.split(marker, 1)[1])
            for candidate in candidates:
                if candidate.exists():
                    camera["data_path"] = str(candidate)
                    break


def load_scene_tables(dataset):
    table_root = Path(dataset.data_root) / dataset.version
    with (table_root / "sample.json").open("r") as handle:
        samples = {row["token"]: row for row in json.load(handle)}
    with (table_root / "scene.json").open("r") as handle:
        scenes = {row["token"]: row for row in json.load(handle)}
    return samples, scenes


def resolve_frame(dataset, scene_name, frame_index):
    if frame_index < 0:
        raise ValueError("--scene-frame-index must be non-negative")
    samples, scenes = load_scene_tables(dataset)
    indices = []
    for index, info in enumerate(dataset.data_infos):
        sample = samples.get(info["token"])
        if sample and scenes[sample["scene_token"]]["name"] == scene_name:
            indices.append(index)
    indices.sort(key=lambda i: dataset.data_infos[i]["timestamp"])
    if frame_index >= len(indices):
        raise IndexError("{} has {} frames, requested {}".format(
            scene_name, len(indices), frame_index))
    return indices[frame_index]


def camera_matrices(camera):
    camera2lidar = np.eye(4, dtype=np.float64)
    camera2lidar[:3, :3] = np.asarray(
        camera["sensor2lidar_rotation"], dtype=np.float64)
    camera2lidar[:3, 3] = np.asarray(
        camera["sensor2lidar_translation"], dtype=np.float64)
    lidar2camera = np.linalg.inv(camera2lidar)
    intrinsic = np.eye(4, dtype=np.float64)
    intrinsic[:3, :3] = np.asarray(camera["cam_intrinsic"], dtype=np.float64)
    return intrinsic @ lidar2camera, camera2lidar, intrinsic


def project(points, matrix):
    points1 = np.concatenate(
        (points, np.ones((len(points), 1), dtype=np.float64)), axis=1)
    projected = points1 @ matrix.T
    depth = projected[:, 2]
    uv = projected[:, :2] / np.maximum(depth[:, None], 1e-12)
    return uv, depth, projected


def backproject(uv, depth, lidar2img):
    image_h = np.column_stack((uv[:, 0] * depth,
                               uv[:, 1] * depth,
                               depth,
                               np.ones(len(depth), dtype=np.float64)))
    lidar_h = image_h @ np.linalg.inv(lidar2img).T
    return lidar_h[:, :3] / lidar_h[:, 3:4]


def draw_overlay(image, boxes, centers, names, matrix, camera_name):
    result = image.copy()
    corners = boxes.corners.detach().cpu().numpy()
    center_uv, center_depth, _ = project(centers, matrix)
    height, width = result.shape[:2]
    for index, xyz in enumerate(corners):
        uv, depth, _ = project(xyz, matrix)
        color = COLORS[index % len(COLORS)]
        bgr = (color[2], color[1], color[0])
        for start, end in EDGES:
            if depth[start] <= 0.1 or depth[end] <= 0.1:
                continue
            cv2.line(result, tuple(np.round(uv[start]).astype(int)),
                     tuple(np.round(uv[end]).astype(int)), bgr, 2,
                     cv2.LINE_AA)
        u, v = center_uv[index]
        if (center_depth[index] > 0.1 and 0 <= u < width and 0 <= v < height):
            point = tuple(np.round((u, v)).astype(int))
            cv2.drawMarker(result, point, bgr, cv2.MARKER_CROSS, 18, 2)
            cv2.putText(result, "{}:{} z={:.2f}m".format(
                index, names[index], center_depth[index]),
                (point[0] + 7, point[1] - 7), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, bgr, 1, cv2.LINE_AA)
    cv2.putText(result, camera_name, (15, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return result


def tile_images(images, columns=4, tile_width=480):
    resized = []
    for image in images:
        height = int(round(image.shape[0] * tile_width / image.shape[1]))
        resized.append(cv2.resize(image, (tile_width, height)))
    tile_height = max(image.shape[0] for image in resized)
    rows = []
    for start in range(0, len(resized), columns):
        row = resized[start:start + columns]
        row += [np.zeros((tile_height, tile_width, 3), np.uint8)] * (
            columns - len(row))
        row = [cv2.copyMakeBorder(x, 0, tile_height - x.shape[0], 0, 0,
                                  cv2.BORDER_CONSTANT) for x in row]
        rows.append(np.concatenate(row, axis=1))
    return np.concatenate(rows, axis=0)


def petr_geometry_config(cfg):
    head = cfg.model.pts_bbox_head
    pipeline = cfg.test_pipeline
    resize_step = next(step for step in pipeline
                       if step["type"] == "ResizeCropFlipImage")
    aug = resize_step["data_aug_conf"]
    raw_h, raw_w = aug["H"], aug["W"]
    final_h, final_w = aug["final_dim"]
    resize = max(final_h / raw_h, final_w / raw_w)
    resized_w, resized_h = int(raw_w * resize), int(raw_h * resize)
    crop_h = int((1 - np.mean(aug["bot_pct_lim"])) * resized_h) - final_h
    crop_w = int(max(0, resized_w - final_w) / 2)
    ida = np.eye(4, dtype=np.float64)
    ida[0, 0] = resize
    ida[1, 1] = resize
    ida[0, 3] = -crop_w
    ida[1, 3] = -crop_h
    depth_num = int(head.get("depth_num", 64))
    depth_start = float(head.get("depth_start", 1.0))
    position_range = list(head["position_range"])
    indices = np.arange(depth_num, dtype=np.float64)
    if head.get("LID", False):
        bin_size = ((position_range[3] - depth_start) /
                    (depth_num * (1 + depth_num)))
        depths = depth_start + bin_size * indices * (indices + 1)
    else:
        bin_size = (position_range[3] - depth_start) / depth_num
        depths = depth_start + bin_size * indices
    # PCCR PETR level-0 CPFPN output is stride 16: 320x800 -> 20x50.
    feature_h, feature_w = final_h // 16, final_w // 16
    return {"ida": ida, "resize": resize, "crop": (crop_w, crop_h),
            "image_shape": (final_h, final_w),
            "feature_shape": (feature_h, feature_w), "depths": depths,
            "LID": bool(head.get("LID", False)),
            "depth_start": depth_start, "position_range": position_range}


def transform_image_for_petr(image, geometry):
    final_h, final_w = geometry["image_shape"]
    resized = cv2.resize(image, None, fx=geometry["resize"],
                         fy=geometry["resize"], interpolation=cv2.INTER_LINEAR)
    crop_w, crop_h = geometry["crop"]
    return resized[crop_h:crop_h + final_h, crop_w:crop_w + final_w].copy()


def draw_token_overlay(image, observations, feature_shape):
    result = image.copy()
    feature_h, feature_w = feature_shape
    stride_x = image.shape[1] / feature_w
    stride_y = image.shape[0] / feature_h
    for row in observations:
        color = COLORS[row["gt_index"] % len(COLORS)]
        bgr = (color[2], color[1], color[0])
        col, feat_row = row["token_col"], row["token_row"]
        cv2.rectangle(result,
                      (int(round(col * stride_x)),
                       int(round(feat_row * stride_y))),
                      (int(round((col + 1) * stride_x)),
                       int(round((feat_row + 1) * stride_y))), bgr, 2)
        cv2.drawMarker(result,
                       (int(round(row["gt_u"])), int(round(row["gt_v"]))),
                       bgr, cv2.MARKER_CROSS, 16, 2)
        cv2.circle(result,
                   (int(round(row["token_u"])), int(round(row["token_v"]))),
                   5, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(result,
                    "{} ray={:.2f}m bin={:.2f}m".format(
                        row["gt_index"], row["ray_error_at_gt_depth_m"],
                        row["nearest_candidate_error_m"]),
                    (int(round(row["gt_u"])) + 6,
                     int(round(row["gt_v"])) - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, bgr, 1, cv2.LINE_AA)
    return result


def analyze_rig(dataset, index, rig, output_dir, bev_limit, geometry):
    info = dataset.data_infos[index]
    annotation = dataset.get_ann_info(index)
    labels = np.asarray(annotation["gt_labels_3d"], dtype=np.int64)
    valid = labels >= 0
    boxes = annotation["gt_bboxes_3d"][valid]
    centers = boxes.gravity_center.detach().cpu().numpy().astype(np.float64)
    labels = labels[valid]
    names = [dataset.CLASSES[label] for label in labels]
    rows, rays, overlays = [], [], []
    token_rows, token_rays, token_overlays = [], [], []

    for camera_index, (camera_name, camera) in enumerate(info["cams"].items()):
        matrix, camera2lidar, intrinsic = camera_matrices(camera)
        image = cv2.imread(camera["data_path"])
        if image is None:
            raise FileNotFoundError(camera["data_path"])
        uv, depth, _ = project(centers, matrix)
        inside = ((depth > 0.1) & (uv[:, 0] >= 0) &
                  (uv[:, 0] < image.shape[1]) & (uv[:, 1] >= 0) &
                  (uv[:, 1] < image.shape[0]))
        reconstructed = backproject(uv, depth, matrix)
        errors = np.linalg.norm(reconstructed - centers, axis=1)
        origin = camera2lidar[:3, 3]
        for gt_index in np.flatnonzero(inside):
            rows.append({
                "rig": rig, "camera_index": camera_index,
                "camera": camera_name, "gt_index": int(gt_index),
                "class": names[gt_index], "u": float(uv[gt_index, 0]),
                "v": float(uv[gt_index, 1]),
                "camera_depth_m": float(depth[gt_index]),
                "gt_x": float(centers[gt_index, 0]),
                "gt_y": float(centers[gt_index, 1]),
                "gt_z": float(centers[gt_index, 2]),
                "reconstructed_x": float(reconstructed[gt_index, 0]),
                "reconstructed_y": float(reconstructed[gt_index, 1]),
                "reconstructed_z": float(reconstructed[gt_index, 2]),
                "roundtrip_error_m": float(errors[gt_index]),
            })
            rays.append((origin.copy(), reconstructed[gt_index].copy(),
                         gt_index, camera_name))
        overlays.append(draw_overlay(
            image, boxes, centers, names, matrix, camera_name))

        # Reproduce the deterministic PETR test-time resize/crop and the exact
        # token coordinates/depth bins used by PETRHead.position_embeding().
        petr_matrix = geometry["ida"] @ matrix
        petr_image = transform_image_for_petr(image, geometry)
        petr_uv, petr_depth, _ = project(centers, petr_matrix)
        final_h, final_w = geometry["image_shape"]
        feature_h, feature_w = geometry["feature_shape"]
        stride_x, stride_y = final_w / feature_w, final_h / feature_h
        petr_inside = ((petr_depth > 0.1) & (petr_uv[:, 0] >= 0) &
                       (petr_uv[:, 0] < final_w) & (petr_uv[:, 1] >= 0) &
                       (petr_uv[:, 1] < final_h))
        camera_token_rows = []
        for gt_index in np.flatnonzero(petr_inside):
            # A CNN token owns this image cell. PETR associates it with the
            # top-left grid coordinate col*stride,row*stride (not cell centre).
            token_col = min(int(petr_uv[gt_index, 0] // stride_x), feature_w - 1)
            token_row = min(int(petr_uv[gt_index, 1] // stride_y), feature_h - 1)
            token_uv = np.asarray([[token_col * stride_x,
                                    token_row * stride_y]], dtype=np.float64)
            true_depth = np.asarray([petr_depth[gt_index]], dtype=np.float64)
            ray_at_true_depth = backproject(token_uv, true_depth, petr_matrix)[0]
            candidate_uv = np.repeat(token_uv, len(geometry["depths"]), axis=0)
            candidates = backproject(candidate_uv, geometry["depths"], petr_matrix)
            candidate_errors = np.linalg.norm(candidates - centers[gt_index], axis=1)
            nearest_index = int(np.argmin(candidate_errors))
            true_depth_bin_index = int(np.argmin(
                np.abs(geometry["depths"] - true_depth[0])))
            row = {
                "rig": rig, "camera_index": camera_index,
                "camera": camera_name, "gt_index": int(gt_index),
                "class": names[gt_index],
                "gt_u": float(petr_uv[gt_index, 0]),
                "gt_v": float(petr_uv[gt_index, 1]),
                "gt_camera_depth_m": float(true_depth[0]),
                "token_row": token_row, "token_col": token_col,
                "token_u": float(token_uv[0, 0]),
                "token_v": float(token_uv[0, 1]),
                "pixel_to_token_offset_px": float(np.linalg.norm(
                    petr_uv[gt_index] - token_uv[0])),
                "ray_error_at_gt_depth_m": float(np.linalg.norm(
                    ray_at_true_depth - centers[gt_index])),
                "nearest_depth_bin_index": nearest_index,
                "nearest_depth_bin_m": float(geometry["depths"][nearest_index]),
                "nearest_candidate_error_m": float(candidate_errors[nearest_index]),
                "closest_depth_value_bin_index": true_depth_bin_index,
                "closest_depth_value_bin_m": float(
                    geometry["depths"][true_depth_bin_index]),
                "depth_quantization_error_m": float(abs(
                    geometry["depths"][true_depth_bin_index] - true_depth[0])),
            }
            token_rows.append(row)
            camera_token_rows.append(row)
            token_rays.append((camera2lidar[:3, 3].copy(),
                               ray_at_true_depth.copy(),
                               candidates[nearest_index].copy(), gt_index,
                               camera_name))
        token_overlays.append(draw_token_overlay(
            petr_image, camera_token_rows, geometry["feature_shape"]))

    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir / "{}_camera_overlays.jpg".format(rig)),
                tile_images(overlays))
    cv2.imwrite(str(output_dir / "{}_petr_token_overlays.jpg".format(rig)),
                tile_images(token_overlays))
    plot_bev(rig, centers, names, rays, output_dir, bev_limit)
    plot_token_bev(rig, centers, names, token_rays, output_dir, bev_limit)
    return {"rig": rig, "token": info["token"], "centers": centers,
            "labels": labels, "names": names, "rows": rows, "rays": rays,
            "token_rows": token_rows, "token_rays": token_rays}


def plot_token_bev(rig, centers, names, rays, output_dir, limit):
    fig, ax = plt.subplots(figsize=(10, 10))
    for origin, ray_point, candidate, _, _ in rays:
        ax.plot([origin[0], ray_point[0]], [origin[1], ray_point[1]],
                color="0.72", linewidth=0.7, alpha=0.55)
        ax.scatter(ray_point[0], ray_point[1], marker="x", s=25,
                   color="tab:orange")
        ax.scatter(candidate[0], candidate[1], marker="+", s=28,
                   color="tab:red")
    for index, (center, name) in enumerate(zip(centers, names)):
        ax.scatter(center[0], center[1], s=38,
                   color=np.asarray(COLORS[index % len(COLORS)]) / 255.0)
        ax.text(center[0] + 0.4, center[1] + 0.4,
                "{}:{}".format(index, name), fontsize=7)
    ax.set(xlim=(-limit, limit), ylim=(-limit, limit), xlabel="LiDAR x (m)",
           ylabel="LiDAR y (m)",
           title=("{} PETR tokens: GT dot, token ray at true depth x, "
                  "nearest 64-depth candidate +").format(rig))
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(str(output_dir / "{}_petr_token_geometry.png".format(rig)),
                dpi=180)
    plt.close(fig)


def plot_bev(rig, centers, names, rays, output_dir, limit):
    fig, ax = plt.subplots(figsize=(10, 10))
    for origin, reconstructed, gt_index, camera in rays:
        ax.plot([origin[0], reconstructed[0]], [origin[1], reconstructed[1]],
                color="0.75", linewidth=0.6, alpha=0.45)
        ax.scatter(reconstructed[0], reconstructed[1], marker="x", s=24,
                   color="tab:red")
    for index, (center, name) in enumerate(zip(centers, names)):
        ax.scatter(center[0], center[1], s=35,
                   color=np.asarray(COLORS[index % len(COLORS)]) / 255.0)
        ax.text(center[0] + 0.4, center[1] + 0.4,
                "{}:{}".format(index, name), fontsize=7)
    camera_origins = {}
    for origin, _, _, camera in rays:
        camera_origins[camera] = origin
    for camera, origin in camera_origins.items():
        ax.scatter(origin[0], origin[1], marker="^", s=80, color="black")
        ax.text(origin[0] + 0.3, origin[1] + 0.3, camera, fontsize=7)
    ax.set(xlim=(-limit, limit), ylim=(-limit, limit), xlabel="LiDAR x (m)",
           ylabel="LiDAR y (m)", title="{} calibration round trip".format(rig))
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(str(output_dir / "{}_bev_roundtrip.png".format(rig)), dpi=180)
    plt.close(fig)


def match_centers(a, b, max_distance):
    matches, used = [], set()
    for ai, (center, label) in enumerate(zip(a["centers"], a["labels"])):
        candidates = [(float(np.linalg.norm(center - other)), bi)
                      for bi, (other, other_label) in enumerate(
                          zip(b["centers"], b["labels"]))
                      if bi not in used and label == other_label]
        if not candidates:
            continue
        distance, bi = min(candidates)
        if distance <= max_distance:
            matches.append((ai, bi, distance))
            used.add(bi)
    return matches


def plot_pair(a, b, matches, output_dir, limit):
    fig, axes = plt.subplots(1, 2, figsize=(18, 9), sharex=True, sharey=True)
    for ax, result in zip(axes, (a, b)):
        for origin, reconstructed, gt_index, _ in result["rays"]:
            ax.plot([origin[0], reconstructed[0]],
                    [origin[1], reconstructed[1]], color="0.8",
                    linewidth=0.5, alpha=0.35)
        for index, (center, name) in enumerate(
                zip(result["centers"], result["names"])):
            ax.scatter(center[0], center[1], s=30,
                       color=np.asarray(COLORS[index % len(COLORS)]) / 255.0)
            ax.text(center[0] + 0.35, center[1] + 0.35,
                    "{}:{}".format(index, name), fontsize=6)
        origins = {camera: origin for origin, _, _, camera in result["rays"]}
        for camera, origin in origins.items():
            ax.scatter(origin[0], origin[1], marker="^", s=70, color="black")
            ax.text(origin[0] + 0.25, origin[1] + 0.25, camera, fontsize=6)
        ax.set_title(result["rig"])
        ax.set(xlim=(-limit, limit), ylim=(-limit, limit), xlabel="LiDAR x (m)")
        ax.set_aspect("equal")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("LiDAR y (m)")
    fig.suptitle("Paired frame: {} matched GT centers; max mismatch {:.6f} m".format(
        len(matches), max([x[2] for x in matches] or [float("nan")])))
    fig.tight_layout()
    fig.savefig(str(output_dir / "paired_bev_comparison.png"), dpi=180)
    plt.close(fig)


def write_outputs(a, b, matches, output_dir, args):
    rows = a["rows"] + b["rows"]
    fields = list(rows[0]) if rows else []
    if fields:
        with (output_dir / "roundtrip_measurements.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    token_rows = a["token_rows"] + b["token_rows"]
    token_fields = list(token_rows[0]) if token_rows else []
    if token_fields:
        with (output_dir / "petr_token_geometry.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=token_fields)
            writer.writeheader()
            writer.writerows(token_rows)
    errors = [row["roundtrip_error_m"] for row in rows]
    summary = {
        "scene_name": args.scene_name,
        "scene_frame_index": args.scene_frame_index,
        "rigs": {result["rig"]: {
            "sample_token": result["token"],
            "num_gt": len(result["centers"]),
            "num_visible_center_observations": len(result["rows"]),
            "max_roundtrip_error_m": max(
                [x["roundtrip_error_m"] for x in result["rows"]] or [0.0]),
        } for result in (a, b)},
        "matched_gt_count": len(matches),
        "matched_gt_max_center_difference_m": max(
            [match[2] for match in matches] or [0.0]),
        "overall_max_roundtrip_error_m": max(errors or [0.0]),
        "petr_token_geometry": {result["rig"]: {
            "num_visible_token_observations": len(result["token_rows"]),
            "mean_ray_error_at_gt_depth_m": float(np.mean(
                [x["ray_error_at_gt_depth_m"] for x in result["token_rows"]]
                or [0.0])),
            "mean_nearest_candidate_error_m": float(np.mean(
                [x["nearest_candidate_error_m"] for x in result["token_rows"]]
                or [0.0])),
            "max_nearest_candidate_error_m": max(
                [x["nearest_candidate_error_m"] for x in result["token_rows"]]
                or [0.0]),
        } for result in (a, b)},
        "warning": ("A small algebraic round-trip error only proves matrix "
                    "self-consistency. Inspect camera overlays to validate "
                    "coordinate conventions and image/calibration alignment."),
    }
    with (output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    import_plugin(cfg)
    geometry = petr_geometry_config(cfg)
    datasets = [make_dataset(cfg, rig, args.split)
                for rig in (args.rig_a, args.rig_b)]
    indices = [resolve_frame(dataset, args.scene_name, args.scene_frame_index)
               for dataset in datasets]
    output_dir = (Path(args.output_dir) /
                  "{}_frame_{:03d}".format(
                      args.scene_name, args.scene_frame_index))
    results = [analyze_rig(dataset, index, rig, output_dir, args.bev_limit,
                           geometry)
               for dataset, index, rig in zip(
                   datasets, indices, (args.rig_a, args.rig_b))]
    matches = match_centers(results[0], results[1], args.max_match_distance)
    plot_pair(results[0], results[1], matches, output_dir, args.bev_limit)
    summary = write_outputs(results[0], results[1], matches, output_dir, args)
    print(json.dumps(summary, indent=2))
    print("Outputs written to: {}".format(output_dir))


if __name__ == "__main__":
    main()
