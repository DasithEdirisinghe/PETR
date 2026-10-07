#!/usr/bin/env python3
"""Collect per-rig PCCR metrics in the cross-rig plot schema."""

import argparse
import json
from pathlib import Path


RIGS = (
    'R1', 'R1-c10', 'R1-c6', 'R1-f', 'R1-r', 'R1-t',
    'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9',
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluations', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', default='PETR')
    parser.add_argument('--trained-on', default='R1_R1f')
    parser.add_argument('--split', choices=('val', 'test'), default='val')
    args = parser.parse_args()

    results = {}
    for rig in RIGS:
        metrics_path = args.evaluations / rig / 'formatted' / 'metrics_summary.json'
        metrics = json.loads(metrics_path.read_text())
        mean_ap = metrics.get('mean_ap')
        if not isinstance(mean_ap, (int, float)) or not 0 <= mean_ap <= 1:
            raise ValueError('Missing or invalid mean_ap in {}'.format(metrics_path))
        results[rig] = dict(
            model=args.model,
            trained_on=args.trained_on,
            tested_on=rig,
            split=args.split,
            metrics=dict(mAP=mean_ap, raw=metrics),
            source=dict(metrics_summary=str(metrics_path)))

    payload = dict(
        model=args.model,
        trained_on=args.trained_on,
        split=args.split,
        results=results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    print(args.output)


if __name__ == '__main__':
    main()
