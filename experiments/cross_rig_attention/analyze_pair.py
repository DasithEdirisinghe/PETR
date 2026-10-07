#!/usr/bin/env python
"""Compare paired PETR query traces from a source rig and a target rig."""

import argparse
import csv
import json
import math
import re
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402


TERM_ORDER = ('H-X', 'H-G3D', 'H-GMV',
              'E-X', 'E-G3D', 'E-GMV')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-trace', required=True,
                        help='R1 trace directory containing trace_summary.json.')
    parser.add_argument('--target-object-trace', required=True,
                        help='R1-F trace using its independently matched query.')
    parser.add_argument('--target-fixed-trace', required=True,
                        help='R1-F trace retaining the R1 query index.')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--box-expansion', type=float, default=0.5,
                        help='Metres added to every GT-box half extent when '
                             'defining GT-consistent feature-cell rays.')
    parser.add_argument('--top-k', type=int, default=200)
    parser.add_argument('--canvas-height', type=int, default=2400)
    return parser.parse_args()


def load_trace(directory):
    directory = Path(directory)
    with (directory / 'trace_summary.json').open('r') as handle:
        summary = json.load(handle)
    tensors = torch.load(
        str(directory / 'trace_tensors.pt'), map_location='cpu')
    return dict(directory=directory, summary=summary, tensors=tensors)


def find_related_target(summary, source_name):
    targets = summary.get('targets', {})
    if source_name in targets:
        return source_name, targets[source_name]
    matches = [(name, row) for name, row in targets.items()
               if row.get('source_target_name') == source_name]
    if len(matches) != 1:
        raise ValueError(
            'Expected exactly one target related to {}, found {}'.format(
                source_name, len(matches)))
    return matches[0]


def feature_rays(tensors):
    missing = [key for key in ('key_layout', 'lidar2img', 'pad_shape')
               if key not in tensors]
    if missing:
        raise ValueError(
            'Trace is missing {}. Regenerate it with the current '
            'trace_query.py rather than --skip-existing.'.format(missing))
    layout = tensors['key_layout']
    cameras = int(layout['num_cameras'])
    height = int(layout['height'])
    width = int(layout['width'])
    projections = np.asarray(tensors['lidar2img'], dtype=np.float64)
    pad_shapes = tensors['pad_shape']
    origins, directions, camera_ids = [], [], []
    for camera in range(cameras):
        pad_h, pad_w = float(pad_shapes[camera][0]), float(
            pad_shapes[camera][1])
        inverse = np.linalg.inv(projections[camera])
        origin_h = inverse.dot(np.asarray([0.0, 0.0, 0.0, 1.0]))
        origin = origin_h[:3] / max(abs(origin_h[3]), 1e-12)
        rows, columns = np.meshgrid(
            np.arange(height), np.arange(width), indexing='ij')
        u = (columns.reshape(-1) + 0.5) * pad_w / width
        v = (rows.reshape(-1) + 0.5) * pad_h / height
        depth = np.full_like(u, 10.0, dtype=np.float64)
        pixels = np.stack(
            (u * depth, v * depth, depth, np.ones_like(depth)), axis=1)
        far_h = pixels.dot(inverse.T)
        far = far_h[:, :3] / np.maximum(
            np.abs(far_h[:, 3:4]), 1e-12)
        direction = far - origin[None]
        direction /= np.maximum(
            np.linalg.norm(direction, axis=1, keepdims=True), 1e-12)
        origins.append(np.repeat(origin[None], height * width, axis=0))
        directions.append(direction)
        camera_ids.append(np.full(height * width, camera, dtype=np.int64))
    return (np.concatenate(origins), np.concatenate(directions),
            np.concatenate(camera_ids))


def rays_intersect_box(origins, directions, box, expansion):
    """Ray intersection with an oriented LiDAR box using a slab test."""
    box = np.asarray(box, dtype=np.float64)
    center = box[:3]
    half = np.maximum(box[3:6] * 0.5 + float(expansion), 1e-4)
    yaw = float(box[6])
    cosine, sine = math.cos(yaw), math.sin(yaw)
    world_to_box = np.asarray([[cosine, sine, 0.0],
                               [-sine, cosine, 0.0],
                               [0.0, 0.0, 1.0]])
    local_origins = (origins - center[None]).dot(world_to_box.T)
    local_directions = directions.dot(world_to_box.T)
    t_min = np.full(len(origins), -np.inf)
    t_max = np.full(len(origins), np.inf)
    possible = np.ones(len(origins), dtype=np.bool_)
    for axis in range(3):
        origin_axis = local_origins[:, axis]
        direction_axis = local_directions[:, axis]
        parallel = np.abs(direction_axis) < 1e-10
        possible &= ~(parallel & (np.abs(origin_axis) > half[axis]))
        safe_direction = np.where(parallel, 1.0, direction_axis)
        first = (-half[axis] - origin_axis) / safe_direction
        second = (half[axis] - origin_axis) / safe_direction
        near = np.minimum(first, second)
        far = np.maximum(first, second)
        near[parallel] = -np.inf
        far[parallel] = np.inf
        t_min = np.maximum(t_min, near)
        t_max = np.minimum(t_max, far)
    return possible & (t_max >= np.maximum(t_min, 0.0))


def normalized_entropy(attention, valid):
    count = int(valid.sum())
    if count <= 1:
        return float('nan')
    probability = attention[:, valid].double().clamp_min(1e-12)
    entropy = -(probability * probability.log()).sum(dim=-1)
    return float((entropy / math.log(count)).mean())


def standardized_discrimination(component, gt_mask, valid):
    values = []
    for head in range(component.shape[0]):
        gt = component[head, gt_mask & valid].double()
        background = component[head, (~gt_mask) & valid].double()
        all_values = component[head, valid].double()
        if gt.numel() == 0 or background.numel() == 0:
            continue
        scale = float(all_values.std(unbiased=False))
        if scale <= 1e-12:
            continue
        values.append(float((gt.mean() - background.mean()) / scale))
    return float(np.mean(values)) if values else float('nan')


def finite_or_none(value):
    value = float(value)
    return value if math.isfinite(value) else None


def target_metrics(label, trace, target_name, target, box_expansion, top_k):
    tensors = trace['tensors']
    slot = int(target['trace_slot'])
    origins, directions, camera_ids = feature_rays(tensors)
    exact_mask = rays_intersect_box(
        origins, directions, target['gt_box'], 0.0)
    gt_mask = rays_intersect_box(
        origins, directions, target['gt_box'], box_expansion)
    layers = []
    confidences = target['layer_confidence']
    box_errors = target['layer_prediction_to_gt_bev_m']
    for layer_index, trace_layer in enumerate(tensors['traces']):
        attention = trace_layer['attention'][:, slot].float()
        logits = trace_layer['logits'][:, slot].float()
        valid = torch.isfinite(logits).all(dim=0).numpy()
        usable_gt = gt_mask & valid
        usable_exact = exact_mask & valid
        mean_attention = attention.mean(dim=0).numpy()
        full_mass = float(attention[:, usable_gt].sum(dim=-1).mean())
        exact_mass = float(attention[:, usable_exact].sum(dim=-1).mean())
        valid_indices = np.flatnonzero(valid)
        ordered = valid_indices[np.argsort(mean_attention[valid_indices])[::-1]]
        selected = ordered[:min(top_k, len(ordered))]
        top_k_fraction = (float(gt_mask[selected].mean())
                          if len(selected) else float('nan'))
        gt_ranks = np.flatnonzero(gt_mask[ordered])
        best_rank = int(gt_ranks[0] + 1) if len(gt_ranks) else None
        camera_mass = []
        for camera in range(int(tensors['key_layout']['num_cameras'])):
            camera_mask = camera_ids == camera
            camera_mass.append(float(
                attention[:, torch.from_numpy(camera_mask)].sum(-1).mean()))
        row = dict(
            condition=label,
            target_name=target_name,
            query_index=int(target['query_index']),
            gt_index=int(target['gt_index']),
            gt_class=target['gt_class'],
            layer=layer_index + 1,
            confidence=float(confidences[layer_index]),
            bev_error_m=float(box_errors[layer_index]),
            gt_ray_count=int(usable_gt.sum()),
            exact_gt_ray_count=int(usable_exact.sum()),
            valid_token_count=int(valid.sum()),
            gt_attention_mass=full_mass,
            exact_gt_attention_mass=exact_mass,
            normalized_attention_entropy=normalized_entropy(attention, valid),
            top_k_gt_ray_fraction=top_k_fraction,
            best_gt_ray_rank=best_rank,
            camera_attention_mass=camera_mass)
        for term, component_all in trace_layer['component_logits'].items():
            component = component_all[:, slot].float()
            discrimination = standardized_discrimination(
                component, gt_mask, valid)
            removed_attention = torch.softmax(logits - component, dim=-1)
            removed_mass = float(
                removed_attention[:, usable_gt].sum(dim=-1).mean())
            utility = full_mass - removed_mass
            impact = float(trace_layer['component_impact_js'][term][slot])
            percentage = float(
                trace_layer['component_impact_percent'][term][slot])
            prefix = term.replace('-', '_')
            row[prefix + '_discrimination'] = discrimination
            row[prefix + '_directional_utility'] = utility
            row[prefix + '_impact_js'] = impact
            row[prefix + '_impact_percent'] = percentage
        layers.append(row)
    return layers


def safe_name(value):
    return re.sub(r'[^A-Za-z0-9._-]+', '_', value)


def write_csv(path, rows):
    fields = []
    for row in rows:
        for key in row:
            if key not in fields and key != 'camera_attention_mass':
                fields.append(key)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def plot_outcomes(path, condition_rows):
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=180)
    definitions = (
        ('gt_attention_mass', 'GT-consistent attention mass'),
        ('normalized_attention_entropy', 'Normalized attention entropy'),
        ('confidence', 'GT-class confidence'),
        ('bev_error_m', 'Prediction-to-GT BEV error (m)'),
    )
    for axis, (field, title) in zip(axes.flat, definitions):
        for condition, rows in condition_rows.items():
            axis.plot([row['layer'] for row in rows],
                      [row[field] for row in rows], marker='o',
                      linewidth=2, label=condition)
        axis.set_title(title)
        axis.set_xlabel('Decoder layer')
        axis.set_xticks(range(1, 7))
        axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(str(path), bbox_inches='tight')
    plt.close(figure)


def plot_terms(path, condition_rows, terms):
    figure, axes = plt.subplots(
        len(terms), 2, figsize=(14, 3.0 * len(terms)), dpi=180,
        squeeze=False)
    for row_index, term in enumerate(terms):
        prefix = term.replace('-', '_')
        for condition, rows in condition_rows.items():
            layers = [row['layer'] for row in rows]
            axes[row_index, 0].plot(
                layers, [row.get(prefix + '_discrimination', np.nan)
                         for row in rows], marker='o', label=condition)
            axes[row_index, 1].plot(
                layers, [row.get(prefix + '_directional_utility', np.nan)
                         for row in rows], marker='o', label=condition)
        axes[row_index, 0].axhline(0.0, color='black', linewidth=0.8)
        axes[row_index, 1].axhline(0.0, color='black', linewidth=0.8)
        axes[row_index, 0].set_ylabel(term)
        axes[row_index, 0].set_title('GT discrimination')
        axes[row_index, 1].set_title('Directional GT-mass utility')
        for axis in axes[row_index]:
            axis.set_xticks(range(1, 7))
            axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    figure.suptitle(
        'Positive values help select GT-consistent rays; negative values hurt',
        y=1.002)
    figure.tight_layout()
    figure.savefig(str(path), bbox_inches='tight')
    plt.close(figure)


def resize_to_height(image, target_height):
    scale = float(target_height) / image.shape[0]
    return cv2.resize(
        image, (max(1, int(round(image.shape[1] * scale))), target_height),
        interpolation=cv2.INTER_AREA)


def comparison_canvas(path, items, relative_path, target_height):
    panels = []
    for label, trace, target_name in items:
        image_path = trace['directory'] / target_name / relative_path
        image = cv2.imread(str(image_path))
        if image is None:
            return False
        image = resize_to_height(image, target_height)
        header = np.full((64, image.shape[1], 3), (26, 32, 44), np.uint8)
        cv2.putText(header, label, (18, 42), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (255, 255, 255), 2, cv2.LINE_AA)
        panels.append(np.concatenate((header, image), axis=0))
    cv2.imwrite(str(path), np.concatenate(panels, axis=1))
    return True


def write_markdown_summary(path, source_name, condition_rows, terms):
    labels = list(condition_rows)
    with path.open('w') as handle:
        handle.write('# Paired diagnostic summary: {}\n\n'.format(source_name))
        handle.write('## Final-layer outcomes\n\n')
        handle.write('| Condition | Query | GT mass | Entropy | Confidence | '
                     'BEV error (m) |\n')
        handle.write('|---|---:|---:|---:|---:|---:|\n')
        for label in labels:
            row = condition_rows[label][-1]
            handle.write('| {} | {} | {:.4f} | {:.4f} | {:.4f} | {:.3f} |\n'.format(
                label, row['query_index'], row['gt_attention_mass'],
                row['normalized_attention_entropy'], row['confidence'],
                row['bev_error_m']))
        handle.write('\n## First-layer term diagnosis\n\n')
        handle.write('| Term | ' + ' | '.join(
            '{} discrimination / utility'.format(label) for label in labels) +
            ' |\n')
        handle.write('|---|' + '|'.join(['---:'] * len(labels)) + '|\n')
        for term in terms:
            prefix = term.replace('-', '_')
            values = []
            for label in labels:
                row = condition_rows[label][0]
                values.append('{:.3f} / {:+.4f}'.format(
                    row.get(prefix + '_discrimination', float('nan')),
                    row.get(prefix + '_directional_utility', float('nan'))))
            handle.write('| {} | {} |\n'.format(term, ' | '.join(values)))
        handle.write(
            '\nPositive discrimination means the term scores GT-consistent '
            'rays above other valid rays. Positive utility means removing the '
            'term reduces GT attention mass; negative utility means the term '
            'is currently directing attention away from the GT.\n')


def main():
    args = parse_args()
    if args.box_expansion < 0:
        raise ValueError('--box-expansion must be non-negative')
    if args.top_k < 1 or args.canvas_height < 100:
        raise ValueError('--top-k and --canvas-height must be positive')
    source = load_trace(args.source_trace)
    target_object = load_trace(args.target_object_trace)
    target_fixed = load_trace(args.target_fixed_trace)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    all_rows = []
    report = dict(
        source_trace=str(source['directory']),
        target_object_trace=str(target_object['directory']),
        target_fixed_trace=str(target_fixed['directory']),
        box_expansion_m=args.box_expansion,
        targets={})
    source_rig = source['summary'].get('dataset_name') or 'source rig'
    target_rig = target_object['summary'].get('dataset_name') or 'target rig'
    for source_name, source_target in source['summary']['targets'].items():
        object_name, object_target = find_related_target(
            target_object['summary'], source_name)
        fixed_name, fixed_target = find_related_target(
            target_fixed['summary'], source_name)
        conditions = (
            ('{} object-matched'.format(source_rig),
             source, source_name, source_target),
            ('{} object-matched'.format(target_rig),
             target_object, object_name, object_target),
            ('{} fixed {} query'.format(target_rig, source_rig),
             target_fixed, fixed_name, fixed_target),
        )
        condition_rows = {}
        condition_targets = {}
        for label, trace, target_name, target in conditions:
            rows = target_metrics(
                label, trace, target_name, target,
                args.box_expansion, args.top_k)
            condition_rows[label] = rows
            condition_targets[label] = target
            all_rows.extend(rows)
        destination = output / safe_name(source_name)
        destination.mkdir(parents=True, exist_ok=True)
        plot_outcomes(destination / 'outcomes_over_layers.png', condition_rows)
        available_terms = [term for term in TERM_ORDER if any(
            term.replace('-', '_') + '_discrimination' in row
            for rows in condition_rows.values() for row in rows)]
        plot_terms(destination / 'term_diagnostics.png',
                   condition_rows, available_terms)
        write_markdown_summary(
            destination / 'diagnostic_summary.md', source_name,
            condition_rows, available_terms)
        canvas_items = [(label, trace, target_name)
                        for label, trace, target_name, _ in conditions]
        comparison_canvas(
            destination / 'paired_attention_all_layers.png', canvas_items,
            'attention_all_layers.png', args.canvas_height)
        comparison_canvas(
            destination / 'paired_bev_rays_L06.png', canvas_items,
            'bev_ray_attention_L06.png', min(args.canvas_height, 1800))
        report['targets'][source_name] = {}
        for label, rows in condition_rows.items():
            target = condition_targets[label]
            report['targets'][source_name][label] = {
                'query_index': int(target['query_index']),
                'gt_index': int(target['gt_index']),
                'final_confidence': finite_or_none(rows[-1]['confidence']),
                'final_bev_error_m': finite_or_none(rows[-1]['bev_error_m']),
                'final_gt_attention_mass': finite_or_none(
                    rows[-1]['gt_attention_mass']),
                'layer_metrics': [
                    {key: (finite_or_none(value)
                           if isinstance(value, (float, np.floating)) else value)
                     for key, value in row.items()}
                    for row in rows],
            }
    write_csv(output / 'paired_layer_metrics.csv', all_rows)
    with (output / 'paired_diagnostics.json').open('w') as handle:
        json.dump(report, handle, indent=2)
    print('Paired cross-rig diagnostics written to:', output)


if __name__ == '__main__':
    main()
