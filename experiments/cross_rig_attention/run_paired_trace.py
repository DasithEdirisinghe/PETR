#!/usr/bin/env python
"""Run and compare paired PETR traces for one PCC-R frame across two rigs."""

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TRACER = REPO_ROOT / 'experiments/query_trace/trace_query.py'
ANALYZER = Path(__file__).resolve().parent / 'analyze_pair.py'


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene-name', required=True)
    parser.add_argument('--scene-frame-index', required=True, type=int,
                        help='Zero-based synchronized frame index.')
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--gt-indices', nargs='+', type=int)
    selection.add_argument('--query-indices', nargs='+', type=int)
    parser.add_argument('--list-source-queries', action='store_true',
                        help='Write the R1 Hungarian-match table and stop.')
    parser.add_argument('--source-rig', default='R1')
    parser.add_argument('--target-rig', default='R1-f')
    parser.add_argument('--split', default='val')
    parser.add_argument('--data-root', default='data/pccr')
    parser.add_argument('--config', default=(
        'projects/configs/petr/'
        'petr_r50dcn_gridmask_p4_800x320_pccr.py'))
    parser.add_argument('--checkpoint', default=(
        'results/petr_r50dcn_gridmask_p4_800x320_pccr/R1/latest.pth'))
    parser.add_argument('--classes', nargs='+', default=None)
    parser.add_argument('--score-threshold', type=float, default=0.35)
    parser.add_argument('--max-gt-match-distance', type=float, default=0.25)
    parser.add_argument('--box-expansion', type=float, default=0.5)
    parser.add_argument('--bev-ray-top-k', type=int, default=200)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output-dir', default=(
        'experiments/cross_rig_attention/outputs'))
    parser.add_argument('--skip-existing', action='store_true')
    return parser.parse_args()


def run(command, env=None):
    print('+', ' '.join(str(value) for value in command), flush=True)
    subprocess.check_call([str(value) for value in command], env=env)


def trace_complete(directory):
    return ((directory / 'trace_summary.json').is_file() and
            (directory / 'trace_tensors.pt').is_file())


def main():
    args = parse_args()
    if args.scene_frame_index < 0:
        raise ValueError('--scene-frame-index must be non-negative')
    if not args.list_source_queries and not (
            args.gt_indices or args.query_indices):
        raise ValueError(
            'Select --gt-indices/--query-indices, or use '
            '--list-source-queries first')
    os.chdir(str(REPO_ROOT))
    pair_name = '{}_frame_{:03d}_{}_vs_{}'.format(
        args.scene_name, args.scene_frame_index,
        args.source_rig, args.target_rig)
    root = Path(args.output_dir) / pair_name
    source_dir = root / '{}_object_matched'.format(args.source_rig)
    target_object_dir = root / '{}_object_matched'.format(args.target_rig)
    target_fixed_dir = root / '{}_fixed_source_query'.format(args.target_rig)
    source_data = Path(args.data_root) / args.source_rig
    target_data = Path(args.data_root) / args.target_rig
    source_ann = source_data / '{}_infos_{}.pkl'.format(
        args.source_rig, args.split)
    target_ann = target_data / '{}_infos_{}.pkl'.format(
        args.target_rig, args.split)
    for required in (Path(args.config), Path(args.checkpoint),
                     source_ann, target_ann):
        if not required.is_file():
            raise FileNotFoundError(str(required))
    common = [
        sys.executable, str(TRACER),
        '--config', args.config,
        '--checkpoint', args.checkpoint,
        '--scene-name', args.scene_name,
        '--scene-frame-index', str(args.scene_frame_index),
        '--score-threshold', str(args.score_threshold),
        '--device', args.device,
        '--max-gt-match-distance', str(args.max_gt_match_distance),
    ]
    if args.classes:
        common += ['--classes'] + args.classes
    source_command = common + [
        '--data-root', str(source_data) + '/',
        '--ann-file', str(source_ann),
        '--dataset-name', args.source_rig,
        '--run-label', '.',
        '--output-dir', str(source_dir),
    ]
    if args.list_source_queries:
        run(source_command + ['--list-queries'])
        print('Source query table:', source_dir / 'query_candidates.csv')
        return
    if args.gt_indices:
        source_command += ['--gt-indices'] + [
            str(value) for value in args.gt_indices]
    else:
        source_command += ['--query-indices'] + [
            str(value) for value in args.query_indices]
    visualization = [
        '--plot-bev-rays', '--bev-ray-layers',
        '1', '2', '3', '4', '5', '6',
        '--bev-ray-top-k', str(args.bev_ray_top_k),
    ]
    source_command += visualization
    if not (args.skip_existing and trace_complete(source_dir)):
        run(source_command)

    source_summary = source_dir / 'trace_summary.json'
    target_base = common + [
        '--data-root', str(target_data) + '/',
        '--ann-file', str(target_ann),
        '--dataset-name', args.target_rig,
    ]
    object_command = target_base + [
        '--run-label', '.',
        '--output-dir', str(target_object_dir),
        '--match-targets-from', str(source_summary),
    ] + visualization
    if not (args.skip_existing and trace_complete(target_object_dir)):
        run(object_command)

    fixed_command = target_base + [
        '--run-label', '.',
        '--output-dir', str(target_fixed_dir),
        '--fixed-query-targets-from', str(source_summary),
    ] + visualization
    if not (args.skip_existing and trace_complete(target_fixed_dir)):
        run(fixed_command)

    comparison_dir = root / 'comparison'
    run([
        sys.executable, str(ANALYZER),
        '--source-trace', str(source_dir),
        '--target-object-trace', str(target_object_dir),
        '--target-fixed-trace', str(target_fixed_dir),
        '--output-dir', str(comparison_dir),
        '--box-expansion', str(args.box_expansion),
        '--top-k', str(args.bev_ray_top_k),
    ])
    print('Complete paired experiment:', root)


if __name__ == '__main__':
    main()
