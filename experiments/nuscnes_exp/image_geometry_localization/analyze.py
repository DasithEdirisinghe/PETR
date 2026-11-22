#!/usr/bin/env python3
"""Relate nuScenes image geometry to localization of official 4 m matches.

Run with the nuScenes devkit, numpy, pyquaternion, and matplotlib installed.
The checkpoint is recorded as provenance; inference is not rerun.
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


RANGES = ((0, 10), (10, 20), (20, 30), (30, 40), (40, 50))
THRESHOLDS = (0.5, 1.0, 2.0, 4.0)
CAMERA_COLUMNS = (
    'sample_token', 'scene_token', 'gt_token', 'class', 'gt_range_m', 'range_bin',
    'gt_length_m', 'gt_width_m', 'gt_height_m',
    'camera', 'camera_depth_m', 'projected_width_px', 'projected_height_px',
    'projected_area_px2', 'viewing_angle_deg', 'off_axis_angle_deg',
    'visibility_token', 'matched_at_4m', 'score', 'center_error_m',
    'radial_error_m', 'tangential_error_m',
)
GT_COLUMNS = (
    'sample_token', 'scene_token', 'gt_token', 'class', 'gt_range_m', 'range_bin',
    'gt_length_m', 'gt_width_m', 'gt_height_m',
    'visibility_token', 'camera_count', 'best_camera', 'camera_depth_m',
    'projected_width_px', 'projected_height_px', 'projected_area_px2',
    'footprint_bin', 'viewing_angle_deg', 'angle_bin', 'off_axis_angle_deg',
    'matched_at_4m', 'score', 'center_error_m', 'radial_error_m',
    'tangential_error_m',
)
SUMMARY_COLUMNS = (
    'class', 'range_bin', 'factor', 'factor_bin', 'gt_count', 'tp_at_4m',
    'coverage_at_4m_pct', 'tp_lt_0.5m', 'retained_lt_0.5m_pct',
    'tp_lt_1m', 'retained_lt_1m_pct', 'tp_lt_2m', 'retained_lt_2m_pct',
    'mean_center_error_m', 'mean_abs_radial_m', 'mean_abs_tangential_m',
    'mean_camera_depth_m', 'mean_projected_width_px',
    'mean_projected_height_px',
)


def range_bin(distance):
    for low, high in RANGES:
        if low <= distance < high:
            return '{}-{}'.format(low, high)
    return 'out_of_range'


def footprint_bin(width, height):
    side = math.sqrt(width * height)
    if side < 32:
        return '<32 px'
    if side < 96:
        return '32-96 px'
    return '>=96 px'


def angle_bin(angle):
    if angle < 30:
        return '0-30 deg'
    if angle < 60:
        return '30-60 deg'
    return '60-90 deg'


def match_class(predictions, gt_by_sample, distance_fn):
    """Replay confidence-ordered nearest-available same-class 4 m matching."""
    ranked = sorted(enumerate(predictions),
                    key=lambda item: (item[1].detection_score, item[0]),
                    reverse=True)
    taken = set()
    for _, pred in ranked:
        token = pred.sample_token
        nearest = None
        distance = float('inf')
        for index, gt in enumerate(gt_by_sample.get(token, ())):
            if (token, index) in taken:
                continue
            value = distance_fn(gt, pred)
            if value < distance:
                nearest, distance = index, value
        if nearest is not None and distance < 4.0:
            taken.add((token, nearest))
            yield pred, gt_by_sample[token][nearest], distance


def annotation_for_gt(nusc, gt):
    """Link a filtered DetectionBox back to its exact nuScenes annotation."""
    sample = nusc.get('sample', gt.sample_token)
    candidates = []
    for token in sample['anns']:
        ann = nusc.get('sample_annotation', token)
        delta = math.sqrt(sum((a - b) ** 2 for a, b in
                              zip(ann['translation'], gt.translation)))
        if delta < 1e-4:
            candidates.append(ann)
    if len(candidates) != 1:
        raise ValueError('GT annotation lookup is {}-way for sample {} at {}'.format(
            len(candidates), gt.sample_token, gt.translation))
    return candidates[0]


def projected_view(nusc, ann, camera_name, sample_data_token):
    """Project a 3D GT box into one camera, clipping its footprint to the image."""
    import numpy as np
    from pyquaternion import Quaternion

    sd = nusc.get('sample_data', sample_data_token)
    calibration = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
    pose = nusc.get('ego_pose', sd['ego_pose_token'])
    box = nusc.get_box(ann['token'])
    box.translate(-np.asarray(pose['translation']))
    box.rotate(Quaternion(pose['rotation']).inverse)
    box.translate(-np.asarray(calibration['translation']))
    box.rotate(Quaternion(calibration['rotation']).inverse)

    center = box.center
    if center[2] <= 0.1:
        return None
    corners = box.corners().T
    # nuScenes Box.corners(): four corners on each face, corresponding corners
    # across faces. Clip edges crossing the near plane before projection.
    edges = ((0, 1), (1, 2), (2, 3), (3, 0),
             (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7))
    near = 0.1
    points = [corner for corner in corners if corner[2] >= near]
    for i, j in edges:
        a, b = corners[i], corners[j]
        if (a[2] < near) != (b[2] < near):
            t = (near - a[2]) / (b[2] - a[2])
            points.append(a + t * (b - a))
    if not points:
        return None
    xyz = np.asarray(points).T
    intrinsic = np.asarray(calibration['camera_intrinsic'])
    uvw = intrinsic @ xyz
    u, v = uvw[0] / uvw[2], uvw[1] / uvw[2]
    xmin, xmax = max(0.0, float(u.min())), min(float(sd['width']), float(u.max()))
    ymin, ymax = max(0.0, float(v.min())), min(float(sd['height']), float(v.max()))
    if xmax <= xmin or ymax <= ymin:
        return None
    width, height = xmax - xmin, ymax - ymin
    # Axial angle treats front/rear symmetrically: 0 deg axial, 90 deg side-on.
    heading = box.orientation.rotate([1, 0, 0])
    heading_xy = np.asarray(heading[:2])
    ray_xy = np.asarray(center[:2])
    denom = np.linalg.norm(heading_xy) * np.linalg.norm(ray_xy)
    if denom < 1e-8:
        return None
    cosine = abs(float(np.dot(heading_xy, ray_xy) / denom))
    view_angle = math.degrees(math.acos(min(1.0, max(-1.0, cosine))))
    off_axis = math.degrees(math.atan2(math.hypot(center[0], center[1]), center[2]))
    return {
        'camera': camera_name,
        'camera_depth_m': float(center[2]),
        'projected_width_px': width,
        'projected_height_px': height,
        'projected_area_px2': width * height,
        'viewing_angle_deg': view_angle,
        'off_axis_angle_deg': off_axis,
    }


def mean(values):
    return sum(values) / len(values) if values else ''


def summarize(gt_rows):
    groups = defaultdict(list)
    for row in gt_rows:
        for name in ('all classes', row['class']):
            for factor, value in (
                ('all', 'all GT'),
                ('footprint', row['footprint_bin']),
                ('view_angle', row['angle_bin']),
                ('visibility', row['visibility_token']),
            ):
                groups[(name, row['range_bin'], factor, value)].append(row)
    output = []
    for (name, distance, factor, factor_bin), rows in sorted(groups.items()):
        matched = [r for r in rows if r['matched_at_4m']]
        record = {'class': name, 'range_bin': distance, 'factor': factor,
                  'factor_bin': factor_bin, 'gt_count': len(rows),
                  'tp_at_4m': len(matched),
                  'coverage_at_4m_pct': 100 * len(matched) / len(rows)}
        for threshold, label in ((0.5, '0.5'), (1, '1'), (2, '2')):
            count = sum(float(r['center_error_m']) < threshold for r in matched)
            record['tp_lt_{}m'.format(label)] = count
            record['retained_lt_{}m_pct'.format(label)] = (
                100 * count / len(matched) if matched else '')
        record.update({
            'mean_center_error_m': mean([float(r['center_error_m']) for r in matched]),
            'mean_abs_radial_m': mean([abs(float(r['radial_error_m'])) for r in matched]),
            'mean_abs_tangential_m': mean([abs(float(r['tangential_error_m'])) for r in matched]),
            'mean_camera_depth_m': mean([float(r['camera_depth_m']) for r in rows
                                         if r['camera_depth_m'] != '']),
            'mean_projected_width_px': mean([float(r['projected_width_px']) for r in rows
                                             if r['projected_width_px'] != '']),
            'mean_projected_height_px': mean([float(r['projected_height_px']) for r in rows
                                              if r['projected_height_px'] != '']),
        })
        output.append(record)
    return output


def save_plots(summary, output_dir, model):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    plot_dir = output_dir / 'plots'
    plot_dir.mkdir(exist_ok=True)
    names = sorted({r['class'] for r in summary})
    range_labels = ['{}-{}'.format(a, b) for a, b in RANGES]
    factors = (
        ('footprint', ['<32 px', '32-96 px', '>=96 px']),
        ('view_angle', ['0-30 deg', '30-60 deg', '60-90 deg']),
    )
    lookup = {(r['class'], r['range_bin'], r['factor'], r['factor_bin']): r
              for r in summary}
    for name in names:
        for factor, bins in factors:
            fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), dpi=160)
            specs = (('retained_lt_0.5m_pct', '4 m TPs within 0.5 m (%)', 0, 100, 'viridis'),
                     ('mean_abs_radial_m', 'Mean absolute radial error (m)', 0, 4, 'magma'))
            for ax, (key, title, vmin, vmax, cmap) in zip(axes, specs):
                matrix = np.full((len(bins), len(range_labels)), np.nan)
                counts = np.zeros_like(matrix, dtype=int)
                for i, bin_name in enumerate(bins):
                    for j, distance in enumerate(range_labels):
                        row = lookup.get((name, distance, factor, bin_name))
                        if row and row[key] != '':
                            matrix[i, j] = float(row[key])
                            counts[i, j] = int(row['tp_at_4m'])
                masked = np.ma.masked_invalid(matrix)
                image = ax.imshow(masked, vmin=vmin, vmax=vmax, cmap=cmap, aspect='auto')
                ax.set_xticks(range(len(range_labels)))
                ax.set_xticklabels(range_labels)
                ax.set_yticks(range(len(bins)))
                ax.set_yticklabels(bins)
                ax.set_xlabel('GT ego range (m)')
                ax.set_title(title)
                for i in range(len(bins)):
                    for j in range(len(range_labels)):
                        if counts[i, j]:
                            value = '{:.0f}%'.format(matrix[i, j]) if key.endswith('pct') else '{:.2f}'.format(matrix[i, j])
                            ax.text(j, i, '{}\nn={:,}'.format(value, counts[i, j]),
                                    ha='center', va='center', fontsize=8, color='black',
                                    bbox={'facecolor': 'white', 'alpha': 0.72,
                                          'edgecolor': 'none', 'pad': 1})
                fig.colorbar(image, ax=ax, shrink=0.8)
            fig.suptitle('{}: {} vs GT range and {}'.format(model, name, factor.replace('_', ' ')),
                         fontsize=14, fontweight='bold')
            fig.tight_layout()
            fig.savefig(plot_dir / '{}_{}.png'.format(name.replace(' ', '_'), factor),
                        bbox_inches='tight')
            plt.close(fig)


def write_csv(path, columns, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path, model, summary, gt_rows):
    lookup = {(r['class'], r['range_bin'], r['factor'], r['factor_bin']): r
              for r in summary}
    classes = sorted({r['class'] for r in gt_rows})
    lines = [
        '# {} image geometry and localization'.format(model), '',
        'The analysis replays official 4 m matching on saved validation predictions. '
        'Every evaluator-eligible GT contributes once to `gt_geometry.csv`; '
        'every geometrically visible GT-camera pair contributes once to '
        '`camera_views.csv`. The camera with the largest clipped GT box area '
        'supplies each GT-level image feature.', '',
        'Retention columns use the original 4 m TPs as denominator. Coverage '
        'uses all evaluator-eligible GT as denominator. These values are not AP. '
        'An empty cell means no matched 4 m TP.', '',
        '| Class | GT range | GT | 4 m TPs | <0.5 m / 4 m TP | <1 m / 4 m TP | '
        'Mean |radial| (m) | Mean |tangential| (m) |',
        '|---|---|---:|---:|---:|---:|---:|---:|',
    ]
    for name in ['all classes'] + classes:
        for low, high in RANGES:
            label = '{}-{}'.format(low, high)
            row = lookup.get((name, label, 'all', 'all GT'))
            if not row:
                continue
            def fmt(key, suffix='', digits=1):
                value = row[key]
                return '—' if value == '' else '{:.{}f}{}'.format(float(value), digits, suffix)
            lines.append('| {} | {} m | {:,} | {:,} | {} | {} | {} | {} |'.format(
                name, label, row['gt_count'], row['tp_at_4m'],
                fmt('retained_lt_0.5m_pct', '%'), fmt('retained_lt_1m_pct', '%'),
                fmt('mean_abs_radial_m', digits=2),
                fmt('mean_abs_tangential_m', digits=2)))
    lines.extend(('', 'See `geometry_summary.csv` for the same class/range '
                  'breakdown by footprint, viewing angle, and annotation '
                  'visibility. Heatmaps in `plots/` show retention and radial '
                  'error by range and geometry bin.', ''))
    path.write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, help='Label, e.g. PETR or StreamPETR')
    parser.add_argument('--checkpoint', help='Checkpoint path or identifier, for provenance only')
    parser.add_argument('--results-json', required=True, type=Path,
                        help='Saved nuScenes validation results_nusc.json')
    parser.add_argument('--data-root', required=True, type=Path,
                        help='nuScenes root containing v1.0-trainval/')
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()
    if not args.results_json.is_file():
        parser.error('Results file not found: {}'.format(args.results_json))
    if not (args.data_root / 'v1.0-trainval' / 'sample_annotation.json').is_file():
        parser.error('nuScenes v1.0-trainval annotations not found under {}'.format(args.data_root))

    from nuscenes import NuScenes
    from nuscenes.eval.detection.evaluate import NuScenesEval
    try:
        from nuscenes.eval.detection.config import config_factory
    except ImportError:
        from nuscenes.eval.common.config import config_factory

    args.output_dir.mkdir(parents=True, exist_ok=True)
    nusc = NuScenes(version='v1.0-trainval', dataroot=str(args.data_root), verbose=False)
    evaluator = NuScenesEval(
        nusc, config=config_factory('detection_cvpr_2019'),
        result_path=str(args.results_json), eval_set='val',
        output_dir=str(args.output_dir / 'devkit_work'), verbose=False)
    gt_by_class = {name: defaultdict(list) for name in evaluator.cfg.class_names}
    pred_by_class = {name: [] for name in evaluator.cfg.class_names}
    for sample_token in evaluator.gt_boxes.sample_tokens:
        for gt in evaluator.gt_boxes[sample_token]:
            gt_by_class[gt.detection_name][sample_token].append(gt)
    for sample_token in evaluator.pred_boxes.sample_tokens:
        for pred in evaluator.pred_boxes[sample_token]:
            pred_by_class[pred.detection_name].append(pred)
    matches = {}
    for name in evaluator.cfg.class_names:
        for pred, gt, distance in match_class(
                pred_by_class[name], gt_by_class[name],
                evaluator.cfg.dist_fcn_callable):
            gx, gy = gt.ego_translation[:2]
            radius = math.hypot(gx, gy)
            if radius < 1e-8:
                continue
            dx, dy = pred.translation[0] - gt.translation[0], pred.translation[1] - gt.translation[1]
            radial = (dx * gx + dy * gy) / radius
            tangential = (-dx * gy + dy * gx) / radius
            matches[id(gt)] = (pred.detection_score, distance, radial, tangential)
    print('Reconstructed {:,} original 4 m matches'.format(len(matches)), flush=True)

    gt_rows, camera_rows = [], []
    for sample_token in evaluator.gt_boxes.sample_tokens:
        sample = nusc.get('sample', sample_token)
        for gt in evaluator.gt_boxes[sample_token]:
            ann = annotation_for_gt(nusc, gt)
            gt_range = math.hypot(*gt.ego_translation[:2])
            found = matches.get(id(gt))
            common = {
                'sample_token': sample_token, 'scene_token': sample['scene_token'],
                'gt_token': ann['token'],
                'class': gt.detection_name, 'gt_range_m': gt_range,
                'range_bin': range_bin(gt_range),
                # nuScenes annotation size is [width, length, height].
                'gt_width_m': ann['size'][0],
                'gt_length_m': ann['size'][1],
                'gt_height_m': ann['size'][2],
                'visibility_token': ann['visibility_token'],
                'matched_at_4m': int(found is not None),
                'score': found[0] if found else '',
                'center_error_m': found[1] if found else '',
                'radial_error_m': found[2] if found else '',
                'tangential_error_m': found[3] if found else '',
            }
            views = []
            for camera_name, sd_token in sample['data'].items():
                if not camera_name.startswith('CAM_'):
                    continue
                view = projected_view(nusc, ann, camera_name, sd_token)
                if view:
                    views.append(view)
                    camera_rows.append(dict(common, **view))
            best = max(views, key=lambda x: x['projected_area_px2']) if views else None
            row = dict(common, camera_count=len(views), best_camera=best['camera'] if best else '',
                       camera_depth_m=best['camera_depth_m'] if best else '',
                       projected_width_px=best['projected_width_px'] if best else '',
                       projected_height_px=best['projected_height_px'] if best else '',
                       projected_area_px2=best['projected_area_px2'] if best else '',
                       footprint_bin=footprint_bin(best['projected_width_px'], best['projected_height_px'])
                       if best else 'outside_cameras',
                       viewing_angle_deg=best['viewing_angle_deg'] if best else '',
                       angle_bin=angle_bin(best['viewing_angle_deg']) if best else 'outside_cameras',
                       off_axis_angle_deg=best['off_axis_angle_deg'] if best else '')
            gt_rows.append(row)
        if len(gt_rows) and len(gt_rows) % 10000 < len(evaluator.gt_boxes[sample_token]):
            print('Processed {:,} GT'.format(len(gt_rows)), flush=True)
    if sum(row['matched_at_4m'] for row in gt_rows) != len(matches):
        raise AssertionError('4 m match count changed during GT join')
    summary = summarize(gt_rows)
    write_csv(args.output_dir / 'gt_geometry.csv', GT_COLUMNS, gt_rows)
    write_csv(args.output_dir / 'camera_views.csv', CAMERA_COLUMNS, camera_rows)
    write_csv(args.output_dir / 'geometry_summary.csv', SUMMARY_COLUMNS, summary)
    write_report(args.output_dir / 'summary.md', args.model, summary, gt_rows)
    from apparent_size_analysis import analyze_csv
    analyze_csv(args.output_dir / 'gt_geometry.csv', args.output_dir, args.model)
    from experiment2_apparent_area import analyze_csv as analyze_experiment2
    analyze_experiment2(args.output_dir / 'gt_geometry.csv',
                        args.output_dir / 'experiment2', args.model)
    if not args.no_plots:
        save_plots(summary, args.output_dir, args.model)
    (args.output_dir / 'run.json').write_text(json.dumps({
        'model': args.model, 'checkpoint': args.checkpoint,
        'results_json': str(args.results_json.resolve()),
        'data_root': str(args.data_root.resolve()),
        'gt_count': len(gt_rows), 'camera_view_count': len(camera_rows),
        'tp_at_4m': len(matches),
        'notes': 'Checkpoint is provenance only; predictions were read from results_json.'
    }, indent=2) + '\n')
    print('Wrote analysis to {}'.format(args.output_dir))


if __name__ == '__main__':
    main()
