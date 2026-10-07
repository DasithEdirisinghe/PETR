#!/usr/bin/env python3
"""Compare nuScenes validation AP by center-distance threshold and full mAP."""

import argparse
import csv
import json
import math
from pathlib import Path


CLASSES = (
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
    'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone',
)
THRESHOLDS = ('0.5', '1.0', '2.0', '4.0')
RUNS = (
    ('Baseline PETR', 'petr_r50dcn_gridmask_p4_nuscenes_fresh_2gpu_global_batch8',
     '20260911_144132.log.json', '#2878B5'),
    ('URoPE PETR', 'petr_r50dcn_gridmask_p4_nuscenes_fresh_2gpu_global_batch8_urope_multiview',
     '20260916_142210.log.json', '#25A18E'),
    ('PETR without 3DPE', 'petr_r50dcn_gridmask_p4_nuscenes_fresh_2gpu_global_batch8_no_3dpe',
     '20260913_185743.log.json', '#D9534F'),
)


def final_validation(path):
    validation = None
    with path.open() as handle:
        for line in handle:
            record = json.loads(line)
            if record.get('mode') == 'val':
                validation = record
    if validation is None:
        raise ValueError('No validation record in {}'.format(path))
    return validation


def main():
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=repo_root / 'results')
    parser.add_argument('--output-dir', type=Path,
                        default=Path(__file__).resolve().parent / 'outputs' / 'plots')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for label, directory, filename, color in RUNS:
        source = args.results_dir / directory / filename
        record = final_validation(source)
        row = {'model': label, 'epoch': record.get('epoch'), 'source': str(source)}
        for threshold in THRESHOLDS:
            key = 'pts_bbox_NuScenes/{}_AP_dist_{}'
            row['AP_{}m'.format(threshold)] = 100 * sum(
                float(record[key.format(class_name, threshold)])
                for class_name in CLASSES) / len(CLASSES)
        row['mAP'] = 100 * float(record['pts_bbox_NuScenes/mAP'])
        derived_map = sum(row['AP_{}m'.format(t)] for t in THRESHOLDS) / len(THRESHOLDS)
        if abs(derived_map - row['mAP']) > 0.01:
            raise ValueError('Threshold AP and reported mAP disagree in {}'.format(source))
        rows.append(row)

    csv_path = args.output_dir / 'threshold_ap_comparison.csv'
    with csv_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    categories = ('0.5 m', '1 m', '2 m', '4 m', 'Full mAP')
    keys = ['AP_{}m'.format(t) for t in THRESHOLDS] + ['mAP']
    left, right, top, bottom = 100, 1030, 150, 525
    chart_width, chart_height = right - left, bottom - top
    maximum = math.ceil(max(row[key] for row in rows for key in keys) / 10) * 10
    elements = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="640" '
        'viewBox="0 0 1120 640">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#25313b}'
        '.title{font-size:24px;font-weight:700}.legend{font-size:15px}'
        '.tick{font-size:13px;fill:#5f6870}.value{font-size:12px;font-weight:700}'
        '.note{font-size:13px;fill:#5f6870}</style>',
        '<text x="560" y="42" text-anchor="middle" class="title">'
        'nuScenes validation: AP by center-distance threshold</text>',
    ]
    for index, (label, _, _, color) in enumerate(RUNS):
        x = 155 + index * 300
        elements.extend((
            '<rect x="{}" y="72" width="17" height="17" rx="2" fill="{}"/>'
            .format(x, color),
            '<text x="{}" y="86" class="legend">{}</text>'.format(x + 25, label),
        ))
    for tick in range(0, maximum + 1, 10):
        y = bottom - chart_height * tick / maximum
        elements.extend((
            '<line x1="{}" y1="{:.1f}" x2="{}" y2="{:.1f}" '
            'stroke="#dce2e6"/>'.format(left, y, right, y),
            '<text x="{}" y="{:.1f}" text-anchor="end" class="tick">{}%'
            '</text>'.format(left - 12, y + 4, tick),
        ))
    group_width = chart_width / len(categories)
    bar_width = 42
    for group, (category, key) in enumerate(zip(categories, keys)):
        center = left + (group + 0.5) * group_width
        elements.append('<text x="{:.1f}" y="550" text-anchor="middle" '
                        'class="legend">{}</text>'.format(center, category))
        for index, (row, run) in enumerate(zip(rows, RUNS)):
            value = row[key]
            height = chart_height * value / maximum
            x = center + (index - 1) * 50 - bar_width / 2
            y = bottom - height
            elements.extend((
                '<rect x="{:.1f}" y="{:.1f}" width="{}" height="{:.1f}" '
                'rx="2" fill="{}"/>'.format(x, y, bar_width, height, run[3]),
                '<text x="{:.1f}" y="{:.1f}" text-anchor="middle" '
                'class="value">{:.1f}</text>'.format(x + bar_width / 2, y - 7, value),
            ))
    elements.extend((
        '<text x="560" y="602" text-anchor="middle" class="note">'
        'Threshold AP averages 10 classes; full mAP averages all four thresholds.</text>',
        '</svg>',
    ))
    svg_path = args.output_dir / 'threshold_ap_comparison.svg'
    svg_path.write_text('\n'.join(elements) + '\n')
    print(csv_path)
    print(svg_path)
    for row in rows:
        print('{}: {}'.format(row['model'], ', '.join(
            '{}={:.2f}%'.format(key, row[key]) for key in keys)))


if __name__ == '__main__':
    main()
