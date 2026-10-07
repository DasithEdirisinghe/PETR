#!/usr/bin/env python3
"""Plot four nuScenes validation variants without external plot dependencies."""

import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULTS = {
    'petr_original': REPO_ROOT / 'results/petr_r50dcn_gridmask_p4_nuscenes_fresh_2gpu_global_batch8/20260911_144132.log.json',
    'urope_original': REPO_ROOT / 'results/petr_r50dcn_gridmask_p4_nuscenes_fresh_2gpu_global_batch8_urope_multiview/20260916_142210.log.json',
    'petr_oracle': REPO_ROOT / 'experiments/gt_center_oracle/outputs/petr_gt_center_3d/formatted/pts_bbox/metrics_summary.json',
    'urope_oracle': REPO_ROOT / 'experiments/gt_center_oracle/outputs/urope_gt_center_3d/formatted/pts_bbox/metrics_summary.json',
}
DISTANCES = ('0.5', '1.0', '2.0', '4.0')
CLASSES = (
    ('car', 'Car'), ('truck', 'Truck'), ('bus', 'Bus'),
    ('trailer', 'Trailer'), ('construction_vehicle', 'Constr. veh.'),
    ('pedestrian', 'Pedestrian'), ('motorcycle', 'Motorcycle'),
    ('bicycle', 'Bicycle'), ('traffic_cone', 'Traffic cone'),
    ('barrier', 'Barrier'),
)
VARIANTS = (
    ('petr_original', 'PETR', '#2563eb'),
    ('urope_original', 'PETR + URoPE', '#ea580c'),
    ('petr_oracle', 'PETR + GT refs', '#93c5fd'),
    ('urope_oracle', 'URoPE + GT refs', '#fdba74'),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in DEFAULTS.items():
        parser.add_argument('--' + name.replace('_', '-'), type=Path,
                            default=default)
    parser.add_argument('--output', type=Path, default=REPO_ROOT /
                        'experiments/gt_center_oracle/outputs/four_variant_comparison.svg')
    return parser.parse_args()


def load_metrics(path):
    if path.name.endswith('.log.json'):
        with path.open() as handle:
            records = [json.loads(line) for line in handle if line.strip()]
        records = [row for row in records if row.get('mode') == 'val'
                   and row.get('epoch') == 24
                   and 'pts_bbox_NuScenes/NDS' in row]
        if not records:
            raise ValueError('No epoch-24 validation metrics in {}'.format(path))
        row = records[-1]
        prefix = 'pts_bbox_NuScenes/'
        result = {
            'NDS': float(row[prefix + 'NDS']),
            'mAP': float(row[prefix + 'mAP']),
            'mATE': float(row[prefix + 'mATE']),
            'class_ap': {
                name: [float(row[prefix + name + '_AP_dist_' + distance])
                       for distance in DISTANCES]
                for name, _ in CLASSES
            },
        }
    else:
        with path.open() as handle:
            row = json.load(handle)
        result = {
            'NDS': float(row['nd_score']),
            'mAP': float(row['mean_ap']),
            'mATE': float(row['tp_errors']['trans_err']),
            'class_ap': {
                name: [float(row['label_aps'][name][distance])
                       for distance in DISTANCES]
                for name, _ in CLASSES
            },
        }
    return result


def text(x, y, value, size=17, color='#263446', anchor='start', extra=''):
    return ('<text x="{:.1f}" y="{:.1f}" fill="{}" font-size="{}" '
            'text-anchor="{}" {}>{}</text>').format(
                x, y, color, size, anchor, extra, escape(str(value)))


def rect(x, y, width, height, fill, extra=''):
    return ('<rect x="{:.1f}" y="{:.1f}" width="{:.1f}" height="{:.1f}" '
            'fill="{}" {} />').format(x, y, width, height, fill, extra)


def line(x1, y1, x2, y2, color='#d9e2ec', width=1):
    return ('<line x1="{:.1f}" y1="{:.1f}" x2="{:.1f}" y2="{:.1f}" '
            'stroke="{}" stroke-width="{}" />').format(
                x1, y1, x2, y2, color, width)


def bar_panel(parts, x, y, width, height, title, categories, series,
              maximum, unit='%', annotate=False, rotate_labels=False):
    parts.append(rect(x, y, width, height, '#ffffff',
                      'rx="16" stroke="#dce5ee"'))
    parts.append(text(x + 22, y + 33, title, 20, '#152435',
                      extra='font-weight="700"'))
    left, right = x + 68, x + width - 22
    top, bottom = y + 58, y + height - (95 if rotate_labels else 65)
    plot_width, plot_height = right - left, bottom - top
    for tick in range(5):
        value = maximum * tick / 4.0
        yy = bottom - plot_height * tick / 4.0
        parts.append(line(left, yy, right, yy))
        label = '{:.0f}%'.format(value * 100) if unit == '%' else '{:.2f}'.format(value)
        parts.append(text(left - 10, yy + 5, label, 13, '#53687c', 'end'))
    group_width = plot_width / len(categories)
    cluster_width = min(group_width * 0.75, 120)
    gap = 3 if len(categories) > 5 else 5
    bar_width = (cluster_width - gap * 3) / 4
    for category_index, category in enumerate(categories):
        centre = left + (category_index + 0.5) * group_width
        start = centre - cluster_width / 2
        for variant_index, (_, _, color) in enumerate(VARIANTS):
            value = series[variant_index][category_index]
            bar_height = max(0.0, min(value / maximum, 1.0)) * plot_height
            bx = start + variant_index * (bar_width + gap)
            by = bottom - bar_height
            parts.append(rect(bx, by, bar_width, bar_height, color,
                              'rx="2"'))
            if annotate:
                value_label = '{:.1f}'.format(value * 100) if unit == '%' else '{:.3f}'.format(value)
                parts.append(text(bx + bar_width / 2, by - 7,
                                  value_label, 11, '#35485a', 'middle'))
        if rotate_labels:
            parts.append(text(centre + 27, bottom + 25, category, 14,
                              '#35485a', 'end',
                              'transform="rotate(-32 {:.1f} {:.1f})"'.format(
                                  centre + 27, bottom + 25)))
        else:
            parts.append(text(centre, bottom + 25, category, 15,
                              '#35485a', 'middle'))


def make_svg(metrics):
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1800" height="1250" '
        'viewBox="0 0 1800 1250">',
        rect(0, 0, 1800, 1250, '#f5f8fc'),
        text(72, 54, 'nuScenes validation: query reference-point oracle',
             31, '#142335', extra='font-weight="700"'),
        text(72, 86,
             'Same two 24-epoch checkpoints; normal inference versus GT-centered learned references',
             17, '#53687c'),
    ]
    legend_x = 72
    for _, label, color in VARIANTS:
        parts.append(rect(legend_x, 108, 23, 16, color, 'rx="2"'))
        parts.append(text(legend_x + 32, 122, label, 16))
        legend_x += 32 + max(175, len(label) * 10)

    def values(metric):
        return [[metrics[name][metric] for _ in (0,)]
                for name, _, _ in VARIANTS]

    bar_panel(parts, 72, 160, 480, 350, 'Overall detection scores',
              ['NDS', 'mAP'],
              [[metrics[name]['NDS'], metrics[name]['mAP']]
               for name, _, _ in VARIANTS], 0.48, annotate=True)
    bar_panel(parts, 576, 160, 400, 350, 'Translation error (lower is better)',
              ['mATE'], values('mATE'), 1.05, unit='m', annotate=True)
    bar_panel(parts, 1000, 160, 728, 350, 'Car AP by center-distance threshold',
              ['0.5 m', '1 m', '2 m', '4 m'],
              [metrics[name]['class_ap']['car'] for name, _, _ in VARIANTS],
              0.95)
    bar_panel(parts, 72, 534, 1656, 618, 'Class-wise AP averaged over 0.5, 1, 2, and 4 m',
              [label for _, label in CLASSES],
              [[sum(metrics[name]['class_ap'][class_name]) / len(DISTANCES)
                for class_name, _ in CLASSES]
               for name, _, _ in VARIANTS], 0.62, rotate_labels=True)
    parts.append(text(72, 1200,
                      'GT-reference runs use validation ground-truth centers during inference; '
                      'they are diagnostic, not deployable results.',
                      17, '#5e7184'))
    parts.append('</svg>')
    return '\n'.join(parts)


def main():
    args = parse_args()
    metrics = {name: load_metrics(getattr(args, name))
               for name, _, _ in VARIANTS}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(make_svg(metrics))
    print('Plot written to:', args.output)
    for name, label, _ in VARIANTS:
        row = metrics[name]
        print('{:<20} NDS={:.4f} mAP={:.4f} mATE={:.4f}'.format(
            label, row['NDS'], row['mAP'], row['mATE']))


if __name__ == '__main__':
    main()
