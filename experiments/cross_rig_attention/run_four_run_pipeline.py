#!/usr/bin/env python3
"""Ensure all four traces/captures, then produce the paired-car stage comparison."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from paired_selection_indices import ensure_source_tracer_indices

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CHECKPOINTS = {
    'r1': ROOT / 'results/petr_r50dcn_gridmask_p4_800x320_pccr/R1/latest.pth',
    'r1f': ROOT / 'results/petr_r50dcn_gridmask_p4_800x320_pccr/R1-f/latest.pth',
}
CONFIG = ROOT / 'projects/configs/petr/petr_r50dcn_gridmask_p4_800x320_pccr.py'


def run_logged(command, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    print('Running:', ' '.join(map(str, command)), flush=True)
    with log.open('w') as handle:
        subprocess.run(list(map(str, command)), cwd=str(ROOT), stdout=handle,
                       stderr=subprocess.STDOUT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--selection', type=Path, default=None)
    parser.add_argument('--max-examples', type=int, default=8)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    if args.max_examples < 1:
        raise ValueError('--max-examples must be positive')
    base = ROOT / 'experiments/cross_rig_attention/output'
    selection = args.selection or base / 'r1_{}'.format(args.split) / (
        'traces/paired_selection.json')
    with selection.open() as handle:
        pairs = json.load(handle)[:args.max_examples]
    if not pairs:
        raise ValueError('No paired cars in {}'.format(selection))
    pairs = ensure_source_tracer_indices(pairs, args.split)
    output = args.output_dir or base / 'four_run_{}'.format(args.split)
    for index, pair in enumerate(pairs):
        for rig in ('R1', 'R1-f'):
            token = pair['sample_token'] if rig == 'R1' else pair['source_sample_token']
            gt_index = int(pair['tracer_gt_index'] if rig == 'R1' else
                           pair['source_tracer_gt_index'])
            rig_dir = base / ('r1_{}'.format(args.split) if rig == 'R1'
                              else 'r1f_{}'.format(args.split)) / 'traces'
            for model in ('r1', 'r1f'):
                trace_dir = rig_dir / model / '{}_gt{:03d}'.format(token, gt_index)
                if args.force or not (trace_dir / 'trace_summary.json').is_file() or not (
                        trace_dir / 'trace_tensors.pt').is_file():
                    command = [sys.executable, HERE.parent / 'query_trace/trace_query.py',
                               '--config', CONFIG, '--checkpoint', CHECKPOINTS[model],
                               '--data-root', 'data/pccr/{}/'.format(rig),
                               '--ann-file', 'data/pccr/{0}/{0}_infos_{1}.pkl'.format(
                                   rig, args.split), '--sample-token', token,
                               '--gt-indices', gt_index, '--classes', 'car',
                               '--device', args.device, '--run-label', '.',
                               '--output-dir', trace_dir]
                    run_logged(command, trace_dir / 'four_run_trace.log')
                capture = output / 'tensors' / rig.replace('-', '') / model / (
                    token + '.pt')
                if args.force or not capture.is_file():
                    command = [sys.executable, HERE / 'four_run_stage_capture.py',
                               '--rig', rig, '--model', model, '--split', args.split,
                               '--sample-token', token, '--device', args.device,
                               '--output', capture]
                    run_logged(command, capture.with_suffix('.log'))
        print('Prepared four-run car', index+1, '/', len(pairs), flush=True)
    subprocess.run([sys.executable, str(HERE / 'analyze_four_run_pipeline.py'),
                    '--selection', str(selection), '--split', args.split,
                    '--max-examples', str(args.max_examples),
                    '--output-dir', str(output)], cwd=str(ROOT), check=True)
    subprocess.run([sys.executable, str(HERE / 'report_four_run_pipeline.py'),
                    '--output-dir', str(output)], cwd=str(ROOT), check=True)


if __name__ == '__main__':
    main()
