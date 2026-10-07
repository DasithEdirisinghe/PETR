#!/usr/bin/env python3
"""Trace one frozen R1 PETR checkpoint on paired R1/R1-f frames.

Uses the existing GT-object-matched traces and adds a fixed-R1-query trace
on R1-f. GT selects and scores the physical car, never enters PETR inference.
"""

import argparse
import csv
import json
import pickle
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from analyze_pair import (feature_rays, load_trace, rays_intersect_box,
                          target_metrics)
from plot_car_bev_localization import quaternion_matrix


ROOT = Path(__file__).resolve().parents[2]
TRACER = ROOT / 'experiments/query_trace/trace_query.py'
CONFIG = ROOT / 'projects/configs/petr/petr_r50dcn_gridmask_p4_800x320_pccr.py'
R1_CHECKPOINT = ROOT / 'results/petr_r50dcn_gridmask_p4_800x320_pccr/R1/latest.pth'
COLORS = {'r1_native': '#2563eb', 'r1f_same_query': '#dc3f45',
          'r1f_object_query': '#ef9b3a'}
LABELS = {'r1_native': 'R1 image · native query',
          'r1f_same_query': 'R1-f image · same R1 query',
          'r1f_object_query': 'R1-f image · reassigned query'}


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--max-examples', type=int, default=8)
    parser.add_argument('--paired-selection', default=None,
                        help='Alternative paired_selection.json, e.g. balanced cohort.')
    parser.add_argument('--trace-missing', action='store_true',
                        help='Generate missing R1 and R1-f object-query traces.')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--analyze-only', action='store_true',
                        help='Do not run missing fixed-query traces.')
    parser.add_argument('--force', action='store_true',
                        help='Re-run fixed-query traces even when cached.')
    parser.add_argument('--box-expansion', type=float, default=.5)
    parser.add_argument('--output-dir', default=None)
    return parser.parse_args()


def info_by_token(rig, split):
    path = ROOT / 'data/pccr' / rig / '{}_infos_{}.pkl'.format(rig, split)
    with path.open('rb') as handle:
        return {row['token']: row for row in pickle.load(handle)['infos']}


def radial_error(box, gt, info):
    rotation = quaternion_matrix(info['lidar2ego_rotation'])
    translation = np.asarray(info['lidar2ego_translation'], dtype=np.float64)
    ego_lidar = -rotation.T @ translation
    ray = np.asarray(gt[:2]) - ego_lidar[:2]
    ray /= max(np.linalg.norm(ray), 1e-8)
    return float(np.dot(np.asarray(box[:2])-np.asarray(gt[:2]), ray))


def trace_path(root, model, token, gt_index):
    return root / 'traces' / model / '{}_gt{:03d}'.format(token, gt_index)


def only_target(trace, expected_gt=None):
    targets = trace['summary']['targets']
    if len(targets) != 1:
        raise ValueError('Expected exactly one traced GT target')
    name, target = next(iter(targets.items()))
    if target['gt_class'] != 'car' or (
            expected_gt is not None and target['gt_index'] != expected_gt):
        raise ValueError('Trace selects a different GT car')
    return name, target


def ensure_fixed_trace(source_dir, target_dir, target_token, split,
                       device, force, analyze_only):
    summary_path = target_dir / 'trace_summary.json'
    tensor_path = target_dir / 'trace_tensors.pt'
    if not force and summary_path.is_file() and tensor_path.is_file():
        return
    if analyze_only:
        raise FileNotFoundError('Missing fixed-query trace: ' + str(target_dir))
    target_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, str(TRACER), '--config', str(CONFIG),
        '--checkpoint', str(R1_CHECKPOINT),
        '--data-root', 'data/pccr/R1-f/',
        '--ann-file', 'data/pccr/R1-f/R1-f_infos_{}.pkl'.format(split),
        '--sample-token', target_token,
        '--fixed-query-targets-from', str(source_dir / 'trace_summary.json'),
        '--max-gt-match-distance', '0.25',
        '--classes', 'car', '--device', device,
        '--run-label', '.', '--output-dir', str(target_dir),
    ]
    print('Tracing fixed R1 query on R1-f:', target_token, flush=True)
    with (target_dir / 'trace.log').open('w') as handle:
        subprocess.run(command, cwd=str(ROOT), stdout=handle,
                       stderr=subprocess.STDOUT, check=True)


def ensure_object_trace(rig, directory, token, gt_index, split, device,
                        trace_missing):
    if ((directory / 'trace_summary.json').is_file() and
            (directory / 'trace_tensors.pt').is_file()):
        return
    if not trace_missing:
        raise FileNotFoundError('Missing object-query trace; pass '
                                '--trace-missing: ' + str(directory))
    directory.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, str(TRACER), '--config', str(CONFIG),
        '--checkpoint', str(R1_CHECKPOINT),
        '--data-root', 'data/pccr/{}/'.format(rig),
        '--ann-file', 'data/pccr/{0}/{0}_infos_{1}.pkl'.format(rig, split),
        '--sample-token', token, '--gt-indices', str(gt_index),
        '--classes', 'car', '--device', device,
        '--run-label', '.', '--output-dir', str(directory),
    ]
    print('Tracing R1 model on', rig, token, flush=True)
    with (directory / 'trace.log').open('w') as handle:
        subprocess.run(command, cwd=str(ROOT), stdout=handle,
                       stderr=subprocess.STDOUT, check=True)


def layer_rows(condition, trace, info, pair_index, box_expansion):
    name, target = only_target(trace)
    rows = target_metrics(condition, trace, name, target,
                          box_expansion, top_k=200)
    for layer_index, row in enumerate(rows):
        row.update({
            'pair_index': pair_index, 'condition': condition,
            'query_index': target['query_index'],
            'sample_token': trace['summary']['sample_token'],
            'radial_ego_m': radial_error(
                target['layer_box_decoded'][layer_index], target['gt_box'], info),
            'gt_ray_enrichment': (
                row['gt_attention_mass'] /
                (row['gt_ray_count']/row['valid_token_count'])
                if row['gt_ray_count'] and row['valid_token_count'] else None),
            'geometry_utility': sum(row.get(term+'_directional_utility', 0.0)
                                    for term in ('H_G3D', 'E_G3D')),
            'appearance_utility': sum(row.get(term+'_directional_utility', 0.0)
                                      for term in ('H_X', 'E_X')),
        })
        row.pop('camera_attention_mass', None)
    return rows


def feature_evidence(source, target, source_target, target_target,
                     box_expansion):
    """Same-model source-car prototype versus GT-ray and same-camera background."""
    source_tensors = source['tensors']
    source_features = source_tensors['traces'][0]['image_features'].float().numpy()
    origins, directions, cameras = feature_rays(source_tensors)
    source_mask = rays_intersect_box(
        origins, directions, source_target['gt_box'], box_expansion)
    if not source_mask.any():
        return {'source_gt_tokens': 0, 'target_gt_tokens': 0,
                'source_to_target_object_cosine': None,
                'source_template_auc': None, 'target_template_auc': None}
    prototype = source_features[source_mask].mean(axis=0)
    prototype /= max(np.linalg.norm(prototype), 1e-8)

    def evidence(trace, target_row):
        tensors = trace['tensors']
        features = tensors['traces'][0]['image_features'].float().numpy()
        ray_origins, ray_directions, camera_ids = feature_rays(tensors)
        mask = rays_intersect_box(ray_origins, ray_directions,
                                  target_row['gt_box'], box_expansion)
        if not mask.any():
            return None, 0, None
        visible_cameras = np.unique(camera_ids[mask])
        background = (~mask) & np.isin(camera_ids, visible_cameras)
        # Exclude padded/invalid tokens using the traced query's attention mask.
        slot = int(target_row['trace_slot'])
        valid = np.isfinite(tensors['traces'][0]['logits'][:, slot].numpy()).all(0)
        mask &= valid
        background &= valid
        if not mask.any() or not background.any():
            return None, int(mask.sum()), None
        normalized = features / np.maximum(np.linalg.norm(
            features, axis=1, keepdims=True), 1e-8)
        scores = normalized @ prototype
        positive = scores[mask]
        negative = scores[background]
        # Exact Mann-Whitney AUC; no fitted probe or labels enter inference.
        auc = float(((positive[:, None] > negative[None]).mean() +
                     .5*(positive[:, None] == negative[None]).mean()))
        object_vector = features[mask].mean(0)
        cosine = float(np.dot(object_vector, prototype) /
                       max(np.linalg.norm(object_vector), 1e-8))
        return auc, int(mask.sum()), cosine

    source_auc, source_count, _ = evidence(source, source_target)
    target_auc, target_count, cosine = evidence(target, target_target)
    return {'source_gt_tokens': source_count,
            'target_gt_tokens': target_count,
            'source_to_target_object_cosine': cosine,
            'source_template_auc': source_auc,
            'target_template_auc': target_auc}


def finite_median(values):
    values = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    return float(np.median(values)) if values else None


def main():
    args = arguments()
    if args.max_examples < 1 or args.box_expansion < 0:
        raise ValueError('Need positive --max-examples and nonnegative --box-expansion')
    base = ROOT / 'experiments/cross_rig_attention/output'
    r1 = base / 'r1_{}'.format(args.split)
    r1f = base / 'r1f_{}'.format(args.split)
    selection_path = (Path(args.paired_selection) if args.paired_selection else
                      r1 / 'traces/paired_selection.json')
    with selection_path.open('r') as handle:
        pairs = json.load(handle)[:args.max_examples]
    if not pairs:
        raise ValueError('Paired selection is empty: {}'.format(selection_path))
    with (r1f / 'analysis/trace_candidates.json').open('r') as handle:
        source_candidates = {(r['sample_token'], r['gt_index']): r
                             for r in json.load(handle)}
    infos = {'R1': info_by_token('R1', args.split),
             'R1-f': info_by_token('R1-f', args.split)}
    output = Path(args.output_dir or (
        selection_path.parent / 'internal' if args.paired_selection else
        r1f / 'traces/r1_unwarped_internal'))
    output.mkdir(parents=True, exist_ok=True)
    records = []
    cases = []
    feature_rows = []
    for pair_index, pair in enumerate(pairs):
        source_token = pair['sample_token']
        target_token = pair['source_sample_token']
        source_gt = pair['tracer_gt_index']
        target_gt = pair.get('source_tracer_gt_index')
        if target_gt is None:
            target_gt = source_candidates[(target_token, pair['source_gt_index'])][
                'tracer_gt_index']
        source_dir = trace_path(r1, 'r1', source_token, source_gt)
        object_dir = trace_path(r1f, 'r1', target_token, target_gt)
        fixed_dir = trace_path(r1f, 'r1_fixed_source_query', target_token, target_gt)
        ensure_object_trace('R1', source_dir, source_token, source_gt,
                            args.split, args.device, args.trace_missing)
        ensure_object_trace('R1-f', object_dir, target_token, target_gt,
                            args.split, args.device, args.trace_missing)
        ensure_fixed_trace(source_dir, fixed_dir, target_token, args.split,
                           args.device, args.force, args.analyze_only)
        source = load_trace(source_dir)
        object_trace = load_trace(object_dir)
        fixed = load_trace(fixed_dir)
        source_name, source_target = only_target(source, source_gt)
        _, object_target = only_target(object_trace, target_gt)
        _, fixed_target = only_target(fixed, target_gt)
        if fixed_target['query_index'] != source_target['query_index']:
            raise ValueError('Fixed-query trace changed query index')
        for candidate in (object_target, fixed_target):
            if np.linalg.norm(np.asarray(candidate['gt_box'][:3]) -
                              np.asarray(source_target['gt_box'][:3])) > .25:
                raise ValueError('Paired traces do not contain the same physical GT')
        cases.append({'pair_index': pair_index, 'scene': pair['scene_name'],
                      'scene_frame_index': pair['scene_frame_index'],
                      'r1_token': source_token, 'r1f_token': target_token,
                      'r1_query': source_target['query_index'],
                      'r1f_object_query': object_target['query_index'],
                      'query_changed': bool(source_target['query_index'] !=
                                            object_target['query_index'])})
        feature_rows.append(dict(
            pair_index=pair_index, scene=pair['scene_name'],
            scene_frame_index=pair['scene_frame_index'],
            **feature_evidence(source, object_trace, source_target,
                               object_target, args.box_expansion)))
        for condition, trace, info in [
                ('r1_native', source, infos['R1'][source_token]),
                ('r1f_same_query', fixed, infos['R1-f'][target_token]),
                ('r1f_object_query', object_trace, infos['R1-f'][target_token])]:
            records.extend(layer_rows(condition, trace, info,
                                      pair_index, args.box_expansion))
        print('Analyzed paired car', pair_index+1, '/', len(pairs), flush=True)

    fields = sorted(set().union(*(row.keys() for row in records)))
    with (output / 'per_car_per_layer.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    with (output / 'paired_cases.json').open('w') as handle:
        json.dump(cases, handle, indent=2)
    with (output / 'image_feature_car_evidence.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(feature_rows[0]))
        writer.writeheader()
        writer.writerows(feature_rows)

    metrics = [
        ('bev_error_m', 'BEV center error (m)'),
        ('radial_ego_m', 'Signed ego-radial error (m)'),
        ('confidence', 'Car confidence'),
        ('gt_ray_enrichment', 'GT-ray attention enrichment'),
        ('geometry_utility', 'G3D term GT-ray utility'),
        ('appearance_utility', 'X term GT-ray utility'),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    medians = {}
    for condition in LABELS:
        medians[condition] = {}
        for axis, (field, title) in zip(axes.flat, metrics):
            values = [finite_median(r[field] for r in records
                                    if r['condition'] == condition and
                                    r['layer'] == layer)
                      for layer in range(1, 7)]
            medians[condition][field] = values
            axis.plot(range(1, 7), [np.nan if v is None else v for v in values],
                      '-o', linewidth=2, color=COLORS[condition],
                      label=LABELS[condition])
            axis.set_title(title, loc='left')
            axis.set_xlabel('Decoder layer')
            axis.set_xticks(range(1, 7))
            axis.grid(alpha=.2)
            axis.spines['top'].set_visible(False)
            axis.spines['right'].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle('Frozen R1 PETR on the same cars: unchanged R1 vs R1-f images',
                 x=.03, ha='left', fontsize=17, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(output / 'internal_divergence.png', dpi=220)
    fig.savefig(output / 'internal_divergence.svg')
    plt.close(fig)
    with (output / 'summary.json').open('w') as handle:
        json.dump({'num_paired_cars': len(cases),
                   'query_changed_count': sum(c['query_changed'] for c in cases),
                   'image_feature_evidence': {
                       'source_template_auc_r1_median': finite_median(
                           r['source_template_auc'] for r in feature_rows),
                       'source_template_auc_r1f_median': finite_median(
                           r['target_template_auc'] for r in feature_rows),
                       'object_feature_cosine_median': finite_median(
                           r['source_to_target_object_cosine'] for r in feature_rows),
                   },
                   'median_by_layer': medians,
                   'notes': ('Object-query and same-query traces are observational. '
                             'GT-ray attention utility is change in attention mass '
                             'when one logit term is removed, not AP causality.')},
                  handle, indent=2)
    print('Saved unwarped internal comparison:', output)


if __name__ == '__main__':
    main()
