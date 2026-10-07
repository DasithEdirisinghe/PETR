#!/usr/bin/env python3
"""Plot per-class nuScenes mean AP for PETR positional-encoding variants."""

import argparse
import json
from pathlib import Path


CLASSES = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
DISTANCES = ('0.5', '1.0', '2.0', '4.0')


def parse_args():
    repo_root = Path(__file__).resolve().parents[2]
    results = repo_root / 'results'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full-log', type=Path, default=results /
                        'petr_r50dcn_gridmask_p4_nuscenes_fresh_2gpu_global_batch8' /
                        '20260911_144132.log.json')
    parser.add_argument('--no-3dpe-log', type=Path, default=results /
                        'petr_r50dcn_gridmask_p4_nuscenes_fresh_2gpu_global_batch8_no_3dpe' /
                        '20260913_185743.log.json')
    parser.add_argument('--urope-log', type=Path, default=results /
                        'petr_r50dcn_gridmask_p4_nuscenes_fresh_2gpu_global_batch8_urope_multiview' /
                        '20260916_142210.log.json')
    parser.add_argument('--output', type=Path, default=repo_root /
                        'experiments/nuscenes_reproduction/outputs/plots/'
                        'per_class_map_3dpe_ablation.svg')
    return parser.parse_args()


def read_validation(path):
    validation = None
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row.get('mode') == 'val':
                validation = row
    if validation is None:
        raise RuntimeError('No validation record found in {}'.format(path))
    return validation


def class_mean_aps(validation):
    values = []
    for class_name in CLASSES:
        prefix = 'pts_bbox_NuScenes/{}_AP_dist_'.format(class_name)
        values.append(sum(
            validation[prefix + distance] for distance in DISTANCES
        ) / len(DISTANCES) * 100.0)
    return values


def main():
    args = parse_args()
    full = read_validation(args.full_log)
    no_3dpe = read_validation(args.no_3dpe_log)
    urope = read_validation(args.urope_log)
    full_ap = class_mean_aps(full)
    no_3dpe_ap = class_mean_aps(no_3dpe)
    urope_ap = class_mean_aps(urope)

    labels = [name.replace('_', ' ').title() for name in CLASSES]
    width, height = 1200, 790
    left, right, top = 230, 80, 135
    chart_width = width - left - right
    row_height, bar_height = 57, 15
    max_value = 60.0
    elements = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" '
        'viewBox="0 0 {} {}">'.format(width, height, width, height),
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#20252b}'
        '.title{font-size:25px;font-weight:700}.subtitle{font-size:15px}'
        '.label{font-size:15px}.value{font-size:13px;font-weight:700}'
        '.tick{font-size:12px;fill:#626b73}.legend{font-size:14px}</style>',
        '<text x="600" y="38" text-anchor="middle" class="title">'
        'nuScenes Validation: Effect of Removing PETR 3DPE</text>',
        '<text x="600" y="65" text-anchor="middle" class="subtitle">'
        'Per-class mean AP across 0.5, 1, 2 and 4 m thresholds</text>',
        '<rect x="350" y="84" width="18" height="18" rx="2" fill="#2878B5"/>',
        '<text x="378" y="98" class="legend">Full PETR (3DPE)</text>',
        '<rect x="555" y="84" width="18" height="18" rx="2" fill="#25A18E"/>',
        '<text x="583" y="98" class="legend">URoPE + multiview PE</text>',
        '<rect x="805" y="84" width="18" height="18" rx="2" fill="#D9534F"/>',
        '<text x="833" y="98" class="legend">PETR without 3DPE</text>',
    ]
    for tick in range(0, 61, 10):
        x = left + chart_width * tick / max_value
        elements.append(
            '<line x1="{0:.1f}" y1="115" x2="{0:.1f}" y2="700" '
            'stroke="#d9dee3" stroke-dasharray="4 5"/>'.format(x))
        elements.append('<text x="{:.1f}" y="720" text-anchor="middle" '
                        'class="tick">{}%</text>'.format(x, tick))
    for index, (label, full_value, urope_value, ablation_value) in enumerate(
            zip(labels, full_ap, urope_ap, no_3dpe_ap)):
        center = top + index * row_height
        full_width = chart_width * full_value / max_value
        urope_width = chart_width * urope_value / max_value
        ablation_width = chart_width * ablation_value / max_value
        elements.extend([
            '<text x="215" y="{}" text-anchor="end" dominant-baseline="middle" '
            'class="label">{}</text>'.format(center, label),
            '<rect x="{}" y="{}" width="{:.1f}" height="{}" rx="3" '
            'fill="#2878B5"/>'.format(left, center - 24, full_width, bar_height),
            '<rect x="{}" y="{}" width="{:.1f}" height="{}" rx="3" '
            'fill="#25A18E"/>'.format(left, center - 7, urope_width, bar_height),
            '<rect x="{}" y="{}" width="{:.1f}" height="{}" rx="3" '
            'fill="#D9534F"/>'.format(left, center + 10, ablation_width, bar_height),
            '<text x="{:.1f}" y="{}" dominant-baseline="middle" '
            'class="value">{:.1f}</text>'.format(
                left + full_width + 7, center - 16, full_value),
            '<text x="{:.1f}" y="{}" dominant-baseline="middle" '
            'class="value">{:.1f}</text>'.format(
                left + urope_width + 7, center + 1, urope_value),
            '<text x="{:.1f}" y="{}" dominant-baseline="middle" '
            'class="value">{:.1f}</text>'.format(
                left + ablation_width + 7, center + 18, ablation_value),
        ])
    elements.extend([
        '<text x="600" y="755" text-anchor="middle" class="subtitle">'
        'Overall mAP: Full PETR {:.2f}%  |  URoPE {:.2f}%  |  Without 3DPE {:.2f}%</text>'.format(
            full['pts_bbox_NuScenes/mAP'] * 100.0,
            urope['pts_bbox_NuScenes/mAP'] * 100.0,
            no_3dpe['pts_bbox_NuScenes/mAP'] * 100.0),
        '</svg>',
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text('\n'.join(elements))
    print(args.output)


if __name__ == '__main__':
    main()
