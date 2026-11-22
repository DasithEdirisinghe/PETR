#!/usr/bin/env python3
"""Plot official PETR validation AP for each class and distance threshold."""

import argparse
import csv
import html
import json
import math
import shutil
import subprocess
from pathlib import Path


THRESHOLDS = ('0.5', '1.0', '2.0', '4.0')


def color(value):
    """A light-to-dark teal scale for AP in [0, 100] percent."""
    low, high = (239, 246, 249), (15, 118, 110)
    fraction = max(0.0, min(1.0, value / 100.0))
    return '#{:02x}{:02x}{:02x}'.format(*(
        round(a + (b - a) * fraction) for a, b in zip(low, high)))


def main():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics', type=Path, default=base / 'outputs' /
                        'formatted' / 'pts_bbox' / 'metrics_summary.json')
    parser.add_argument('--output-dir', type=Path, default=base / 'outputs' / 'plots')
    args = parser.parse_args()
    metrics = json.loads(args.metrics.read_text())
    classes = list(metrics['label_aps'])
    if len(classes) != 10:
        raise ValueError('Expected 10 official nuScenes detection classes')
    rows = []
    for name in classes:
        aps = metrics['label_aps'][name]
        values = [100.0 * float(aps[threshold]) for threshold in THRESHOLDS]
        if not all(math.isfinite(value) and 0 <= value <= 100 for value in values):
            raise ValueError('Invalid AP for ' + name)
        rows.append({'class': name, **{'AP@{}m (%)'.format(threshold): value
                                      for threshold, value in zip(THRESHOLDS, values)},
                     'class mAP (%)': sum(values) / len(values)})
        official_class_mean = 100.0 * float(metrics['mean_dist_aps'][name])
        if abs(rows[-1]['class mAP (%)'] - official_class_mean) > 1e-6:
            raise ValueError('Class mean AP disagrees with official value for ' + name)
    overall = [sum(row['AP@{}m (%)'.format(t)] for row in rows) / len(rows)
               for t in THRESHOLDS]
    mean_ap = sum(overall) / len(overall)
    if abs(mean_ap - 100.0 * float(metrics['mean_ap'])) > 1e-6:
        raise ValueError('Class × threshold AP mean disagrees with official mAP')

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / 'per_class_ap_by_threshold.csv'
    with csv_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow({'class': 'All-class mean',
                         **{'AP@{}m (%)'.format(t): value
                            for t, value in zip(THRESHOLDS, overall)},
                         'class mAP (%)': mean_ap})

    left, top, cell_w, cell_h = 260, 168, 170, 48
    width, height = 1180, 835
    headers = ('AP @ 0.5 m', 'AP @ 1 m', 'AP @ 2 m', 'AP @ 4 m', 'Class mAP')
    elements = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">'.format(
            width, height, width, height),
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#17212b}'
        '.title{font-size:27px;font-weight:700}.sub{font-size:16px;fill:#52606d}'
        '.head{font-size:17px;font-weight:700}.label{font-size:18px}'
        '.value{font-size:17px;font-weight:700}.note{font-size:15px;fill:#52606d}</style>',
        '<text x="55" y="55" class="title">PETR validation AP by class and center-distance threshold</text>',
        '<text x="55" y="85" class="sub">Official nuScenes detection metrics · higher AP is better · values in percent</text>',
    ]
    for column, label in enumerate(headers):
        x = left + column * cell_w + cell_w / 2
        elements.append('<text x="{:.1f}" y="137" text-anchor="middle" class="head">{}</text>'.format(
            x, label))
    for row_index, row in enumerate(rows):
        y = top + row_index * cell_h
        label = html.escape(row['class'].replace('_', ' '))
        elements.append('<text x="242" y="{}" text-anchor="end" class="label">{}</text>'.format(
            y + 31, label))
        values = [row['AP@{}m (%)'.format(t)] for t in THRESHOLDS] + [row['class mAP (%)']]
        for column, value in enumerate(values):
            x = left + column * cell_w
            fill = color(value)
            text_color = '#ffffff' if value >= 57 else '#17212b'
            elements.append('<rect x="{}" y="{}" width="{}" height="{}" fill="{}" stroke="#ffffff" stroke-width="3"/>'.format(
                x, y, cell_w, cell_h, fill))
            elements.append('<text x="{:.1f}" y="{}" text-anchor="middle" class="value" style="fill:{}">{:.1f}</text>'.format(
                x + cell_w / 2, y + 31, text_color, value))
    summary_y = top + len(rows) * cell_h + 10
    elements.append('<text x="242" y="{}" text-anchor="end" class="label" font-weight="bold">All-class mean</text>'.format(
        summary_y + 31))
    for column, value in enumerate(overall + [mean_ap]):
        x = left + column * cell_w
        elements.append('<rect x="{}" y="{}" width="{}" height="{}" fill="#dbeafe" stroke="#ffffff" stroke-width="3"/>'.format(
            x, summary_y, cell_w, cell_h))
        elements.append('<text x="{:.1f}" y="{}" text-anchor="middle" class="value">{:.1f}</text>'.format(
            x + cell_w / 2, summary_y + 31, value))
    elements.extend([
        '<text x="55" y="755" class="note">Class mAP = mean of that class’s four threshold AP values.</text>',
        '<text x="55" y="780" class="note">Bottom-right value reproduces official PETR overall mAP: {:.2f}%.</text>'.format(mean_ap),
        '</svg>',
    ])
    svg_path = args.output_dir / 'per_class_ap_by_threshold.svg'
    svg_path.write_text('\n'.join(elements) + '\n')
    print(svg_path)
    print(csv_path)
    converter = shutil.which('rsvg-convert')
    if converter:
        png_path = args.output_dir / 'per_class_ap_by_threshold.png'
        with png_path.open('wb') as handle:
            subprocess.run([converter, '-f', 'png', str(svg_path)],
                           check=True, stdout=handle)
        print(png_path)
    print('Official mAP: {:.2f}%'.format(mean_ap))


if __name__ == '__main__':
    main()
