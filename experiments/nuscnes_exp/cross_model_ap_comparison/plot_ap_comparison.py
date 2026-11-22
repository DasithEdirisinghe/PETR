#!/usr/bin/env python3
"""Plot official nuScenes validation AP for PETR, StreamPETR, OPEN, and EVA."""

import argparse
import csv
import json
import math
import shutil
import subprocess
from pathlib import Path


THRESHOLDS = ('0.5', '1.0', '2.0', '4.0')
MODELS = (
    ('PETR', 'PETR/experiments/nuscnes_exp/localization_oracle/outputs/baseline/baseline/metrics_summary.json', '#3677B9'),
    ('PETR EVA02-L (epoch 16)', 'PETR/results/petr_eva02_l_p4_800x320_nuscenes/epoch_16_nuscenes_val_metrics.json', '#D15B32'),
    ('StreamPETR', 'StreamPETR/test/stream_petr_r50_noflash_428q_nui_60e/Fri_Oct__9_14_02_06_2026/pts_bbox/metrics_summary.json', '#1B9E77'),
    ('OPEN', 'OPEN/test/open_r50_704_90e/Sun_Oct_11_14_07_58_2026/pts_bbox/metrics_summary.json', '#D95F02'),
    ('RepDETR3D EVA02-L', 'StreamPETR/experiments/eva02_gallery/outputs/inference/formatted/pts_bbox/metrics_summary.json', '#8B5CF6'),
)


def read_metrics(name, path):
    data = json.loads(path.read_text())
    if 'label_aps' in data:
        aps = data['label_aps']
        reported_map = float(data['mean_ap'])
    else:
        # Training's official validation log stores class AP rounded to 4 decimals.
        values = data['metrics']
        classes = ('car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                   'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone')
        aps = {name: {threshold: values[f'{name}_AP_dist_{threshold}']
                      for threshold in THRESHOLDS} for name in classes}
        reported_map = float(values['mAP'])
    if len(aps) != 10:
        raise ValueError('{} has {} classes, expected 10'.format(path, len(aps)))
    row = {'model': name, 'source': str(path)}
    for threshold in THRESHOLDS:
        row['AP@{}m'.format(threshold)] = 100 * sum(
            float(class_aps[threshold]) for class_aps in aps.values()) / len(aps)
    row['mAP'] = 100 * reported_map
    computed = sum(row['AP@{}m'.format(t)] for t in THRESHOLDS) / len(THRESHOLDS)
    tolerance = 0.05 if 'metrics' in data else 1e-6
    if abs(computed - row['mAP']) > tolerance:
        raise ValueError('{}: reported mAP differs from mean threshold AP'.format(path))
    return row


def draw_svg(rows, output):
    categories = [('AP@0.5m', '0.5 m'), ('AP@1.0m', '1 m'),
                  ('AP@2.0m', '2 m'), ('AP@4.0m', '4 m'), ('mAP', 'mAP')]
    width, height = 1500, 690
    left, right, top, bottom = 105, 1435, 150, 560
    maximum = max(10, math.ceil(max(row[key] for row in rows for key, _ in categories) / 10) * 10)
    plot_height = bottom - top
    elements = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">'.format(width, height, width, height),
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#27323a}.title{font-size:25px;font-weight:700}'
        '.sub{font-size:15px;fill:#53616b}.legend{font-size:17px}.tick{font-size:14px;fill:#66747e}'
        '.category{font-size:18px;font-weight:600}.value{font-size:14px;font-weight:700}'
        '.note{font-size:14px;fill:#53616b}</style>',
        '<text x="750" y="43" text-anchor="middle" class="title">nuScenes validation detection AP by center-distance threshold</text>',
        '<text x="750" y="71" text-anchor="middle" class="sub">Official validation metrics; PETR EVA02-L is epoch 16</text>',
    ]
    for index, (_, _, color) in enumerate(MODELS):
        x = 95 + index * 280
        elements.append('<rect x="{}" y="100" width="20" height="20" rx="3" fill="{}"/>'.format(x, color))
        elements.append('<text x="{}" y="117" class="legend">{}</text>'.format(x + 29, rows[index]['model']))
    for tick in range(0, maximum + 1, 10):
        y = bottom - plot_height * tick / maximum
        elements.append('<line x1="{}" y1="{:.1f}" x2="{}" y2="{:.1f}" stroke="#e0e6e9"/>'.format(left, y, right, y))
        elements.append('<text x="{}" y="{:.1f}" text-anchor="end" class="tick">{}%</text>'.format(left - 13, y + 5, tick))
    group_width = (right - left) / len(categories)
    for group, (key, label) in enumerate(categories):
        center = left + (group + 0.5) * group_width
        elements.append('<text x="{:.1f}" y="595" text-anchor="middle" class="category">{}</text>'.format(center, label))
        for index, (row, model) in enumerate(zip(rows, MODELS)):
            value = row[key]
            bar_height = plot_height * value / maximum
            bar_width = 40
            x = center + (index - (len(MODELS) - 1) / 2) * 47 - bar_width / 2
            y = bottom - bar_height
            elements.append('<rect x="{:.1f}" y="{:.1f}" width="{}" height="{:.1f}" rx="3" fill="{}"/>'.format(x, y, bar_width, bar_height, model[2]))
            elements.append('<text x="{:.1f}" y="{:.1f}" text-anchor="middle" class="value">{:.1f}</text>'.format(x + bar_width / 2, y - 8, value))
    elements.append('<text x="750" y="640" text-anchor="middle" class="note">Each threshold AP is averaged over 10 classes; mAP averages the four threshold AP values.</text>')
    elements.append('</svg>')
    output.write_text('\n'.join(elements) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent / 'outputs')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [read_metrics(name, args.workspace / source) for name, source, _ in MODELS]
    csv_path = args.output_dir / 'cross_model_ap_comparison.csv'
    with csv_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    svg_path = args.output_dir / 'cross_model_ap_comparison.svg'
    draw_svg(rows, svg_path)
    converter = shutil.which('rsvg-convert')
    if converter:
        png_path = args.output_dir / 'cross_model_ap_comparison.png'
        subprocess.run([converter, '-o', str(png_path), str(svg_path)], check=True)
        print(png_path)
    for row in rows:
        print('{}: {}'.format(row['model'], ', '.join('{}={:.2f}%'.format(key, row[key]) for key in row if key.startswith('AP@') or key == 'mAP')))
    print(svg_path)
    print(csv_path)


if __name__ == '__main__':
    main()
