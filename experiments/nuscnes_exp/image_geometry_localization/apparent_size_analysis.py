#!/usr/bin/env python3
"""Compare apparent-size tertiles within each object class and 5 m GT-range bin."""

import argparse
import csv
import html
import math
from collections import defaultdict
from pathlib import Path
from statistics import median


CLASSES = ('car', 'truck', 'bus', 'trailer', 'construction_vehicle',
           'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier')
SIZES = ('small', 'medium', 'large')
FIELDS = ('class', 'range_bin_m', 'size_tertile', 'gt_count', 'tp_at_4m',
          'coverage_at_4m_pct', 'median_apparent_side_px',
          'mean_gt_range_m', 'mean_viewing_angle_deg',
          'tp_lt_0.5m', 'retained_lt_0.5m_pct',
          'tp_lt_1m', 'retained_lt_1m_pct',
          'tp_lt_2m', 'retained_lt_2m_pct',
          'mean_center_error_m', 'mean_abs_radial_m',
          'mean_abs_tangential_m', 'mean_signed_radial_m')
CONTRAST_FIELDS = ('class', 'range_bin_m', 'small_gt', 'large_gt',
                   'small_tp_at_4m', 'large_tp_at_4m',
                   'small_median_side_px', 'large_median_side_px',
                   'small_retained_lt_0.5m_pct', 'large_retained_lt_0.5m_pct',
                   'large_minus_small_retention_pp',
                   'small_mean_abs_radial_m', 'large_mean_abs_radial_m',
                   'large_minus_small_abs_radial_m',
                   'small_mean_abs_tangential_m', 'large_mean_abs_tangential_m',
                   'large_minus_small_abs_tangential_m',
                   'at_least_30_tp_each')


def average(values):
    return sum(values) / len(values) if values else ''


def range_label(distance, width):
    low = int(distance // width) * width
    return '{}-{}'.format(low, low + width) if 0 <= low < 50 else None


def read_gt(path, width):
    groups = defaultdict(list)
    outside = 0
    with path.open(newline='') as handle:
        for row in csv.DictReader(handle):
            if not row['projected_width_px'] or not row['projected_height_px']:
                outside += 1
                continue
            distance = float(row['gt_range_m'])
            label = range_label(distance, width)
            if label is None:
                continue
            row['apparent_side_px'] = math.sqrt(
                float(row['projected_width_px']) *
                float(row['projected_height_px']))
            groups[(row['class'], label)].append(row)
    return groups, outside


def summary_row(name, distance, size, rows):
    matches = [r for r in rows if r['matched_at_4m'] == '1']
    count = len(matches)
    result = {
        'class': name, 'range_bin_m': distance, 'size_tertile': size,
        'gt_count': len(rows), 'tp_at_4m': count,
        'coverage_at_4m_pct': 100 * count / len(rows),
        'median_apparent_side_px': median(float(r['apparent_side_px']) for r in rows),
        'mean_gt_range_m': average([float(r['gt_range_m']) for r in rows]),
        'mean_viewing_angle_deg': average([float(r['viewing_angle_deg']) for r in rows]),
        'mean_center_error_m': average([float(r['center_error_m']) for r in matches]),
        'mean_abs_radial_m': average([abs(float(r['radial_error_m'])) for r in matches]),
        'mean_abs_tangential_m': average([abs(float(r['tangential_error_m'])) for r in matches]),
        'mean_signed_radial_m': average([float(r['radial_error_m']) for r in matches]),
    }
    for threshold, label in ((0.5, '0.5'), (1.0, '1'), (2.0, '2')):
        retained = sum(float(r['center_error_m']) < threshold for r in matches)
        result['tp_lt_{}m'.format(label)] = retained
        result['retained_lt_{}m_pct'.format(label)] = (
            100 * retained / count if count else '')
    return result


def summaries(groups):
    records = []
    for (name, distance), rows in sorted(groups.items()):
        rows.sort(key=lambda r: float(r['apparent_side_px']))
        size_groups = defaultdict(list)
        for index, row in enumerate(rows):
            size_groups[SIZES[min(2, 3 * index // len(rows))]].append(row)
        for size in SIZES:
            if size_groups[size]:
                records.append(summary_row(name, distance, size, size_groups[size]))
    return records


def contrasts(records):
    lookup = {(r['class'], r['range_bin_m'], r['size_tertile']): r for r in records}
    result = []
    for name, distance in sorted({(r['class'], r['range_bin_m']) for r in records}):
        small = lookup.get((name, distance, 'small'))
        large = lookup.get((name, distance, 'large'))
        if not small or not large:
            continue
        have_tps = small['tp_at_4m'] and large['tp_at_4m']
        result.append({
            'class': name, 'range_bin_m': distance,
            'small_gt': small['gt_count'], 'large_gt': large['gt_count'],
            'small_tp_at_4m': small['tp_at_4m'],
            'large_tp_at_4m': large['tp_at_4m'],
            'small_median_side_px': small['median_apparent_side_px'],
            'large_median_side_px': large['median_apparent_side_px'],
            'small_retained_lt_0.5m_pct': small['retained_lt_0.5m_pct'],
            'large_retained_lt_0.5m_pct': large['retained_lt_0.5m_pct'],
            'large_minus_small_retention_pp': (
                large['retained_lt_0.5m_pct'] - small['retained_lt_0.5m_pct']
                if have_tps else ''),
            'small_mean_abs_radial_m': small['mean_abs_radial_m'],
            'large_mean_abs_radial_m': large['mean_abs_radial_m'],
            'large_minus_small_abs_radial_m': (
                large['mean_abs_radial_m'] - small['mean_abs_radial_m']
                if have_tps else ''),
            'small_mean_abs_tangential_m': small['mean_abs_tangential_m'],
            'large_mean_abs_tangential_m': large['mean_abs_tangential_m'],
            'large_minus_small_abs_tangential_m': (
                large['mean_abs_tangential_m'] - small['mean_abs_tangential_m']
                if have_tps else ''),
            'at_least_30_tp_each': int(
                small['tp_at_4m'] >= 30 and large['tp_at_4m'] >= 30),
        })
    return result


def write_csv(path, fields, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def color(delta):
    # Teal: larger image footprint has better retention. Orange: worse.
    magnitude = min(abs(delta) / 20, 1)
    target = (21, 139, 128) if delta >= 0 else (215, 109, 52)
    rgb = tuple(round(242 * (1 - magnitude) + channel * magnitude)
                for channel in target)
    return '#{:02x}{:02x}{:02x}'.format(*rgb)


def draw_svg(path, records, width, model):
    bins = ['{}-{}'.format(start, start + width)
            for start in range(0, 50, width)]
    left, top, cell_w, cell_h = 195, 148, 99, 43
    svg_width = left + len(bins) * cell_w + 40
    svg_height = top + len(CLASSES) * cell_h + 85
    lookup = {(r['class'], r['range_bin_m']): r for r in records}
    items = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">'.format(
            svg_width, svg_height, svg_width, svg_height),
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#263238}.title{font-size:21px;font-weight:bold}'
        '.sub{font-size:13px;fill:#55636b}.label{font-size:12px}.val{font-size:13px;font-weight:bold}'
        '.n{font-size:10px;fill:#46545c}</style>',
        '<text x="{}" y="34" text-anchor="middle" class="title">Apparent size and {} localization within each class</text>'.format(
            svg_width / 2, html.escape(model)),
        '<text x="{}" y="60" text-anchor="middle" class="sub">Cell = large minus small projected-size tertile in 0.5 m retention among original 4 m TPs</text>'.format(svg_width / 2),
        '<text x="{}" y="81" text-anchor="middle" class="sub">Compared within the same class and 5 m GT-range bin; n shows small / large 4 m TP counts</text>'.format(svg_width / 2),
    ]
    for j, label in enumerate(bins):
        items.append('<text x="{}" y="125" text-anchor="middle" class="label">{} m</text>'.format(
            left + j * cell_w + cell_w / 2, label))
    for i, name in enumerate(CLASSES):
        y = top + i * cell_h
        items.append('<text x="{}" y="{}" text-anchor="end" class="label">{}</text>'.format(
            left - 12, y + 25, html.escape(name.replace('_', ' '))))
        for j, label in enumerate(bins):
            x = left + j * cell_w
            row = lookup.get((name, label))
            stable = row and row['at_least_30_tp_each'] and row['large_minus_small_retention_pp'] != ''
            fill = color(row['large_minus_small_retention_pp']) if stable else '#edf0f1'
            items.append('<rect x="{}" y="{}" width="{}" height="{}" fill="{}" stroke="white"/>'.format(
                x, y, cell_w - 2, cell_h - 2, fill))
            if stable:
                items.append('<text x="{}" y="{}" text-anchor="middle" class="val">{:+.1f} pp</text>'.format(
                    x + cell_w / 2, y + 18, row['large_minus_small_retention_pp']))
                items.append('<text x="{}" y="{}" text-anchor="middle" class="n">n={:,} / {:,}</text>'.format(
                    x + cell_w / 2, y + 34, row['small_tp_at_4m'], row['large_tp_at_4m']))
    items.append('<text x="{}" y="{}" text-anchor="middle" class="sub">Teal: larger footprint has better retention · Orange: smaller footprint has better retention · Grey: &lt;30 TPs in a comparison group</text>'.format(
        svg_width / 2, svg_height - 28))
    items.append('</svg>')
    path.write_text('\n'.join(items) + '\n')


def write_report(path, model, records, contrasts, outside, width):
    lookup = defaultdict(list)
    for row in records:
        lookup[row['class']].append(row)
    lines = [
        '# {}: localization by apparent object size within class'.format(model), '',
        'Apparent size is `sqrt(projected GT width × projected GT height)` in '
        'the camera with the largest clipped GT box area. GT objects are divided '
        'into equal-count small, medium, and large groups **separately for each '
        'class and {} m GT-range bin**. This compares sizes at approximately '
        'the same distance. {} GT objects outside all camera images are omitted '
        'from the size analysis.'.format(width, outside), '',
        'Retention uses only the original 4 m true positives; coverage uses all '
        'GT in the group. These are not AP values. The heatmap greys out '
        'comparisons with fewer than 30 original 4 m TPs in either tertile.', '',
        '![Large minus small 0.5 m retention](apparent_size_retention_delta.svg)', '',
    ]
    for name in CLASSES:
        rows = sorted(lookup[name], key=lambda r: (int(r['range_bin_m'].split('-')[0]),
                                                     SIZES.index(r['size_tertile'])))
        if not rows:
            continue
        lines += ['', '## {}'.format(name.replace('_', ' ')), '',
                  '| GT range | Size | Median image side | GT | 4 m TP | <0.5 m / 4 m TP | Mean |radial| | Mean |tangential| |',
                  '|---|---|---:|---:|---:|---:|---:|---:|']
        for row in rows:
            fmt = lambda key, digits: '—' if row[key] == '' else '{:.{}f}'.format(row[key], digits)
            lines.append('| {} m | {} | {} px | {:,} | {:,} | {}% | {} m | {} m |'.format(
                row['range_bin_m'], row['size_tertile'],
                fmt('median_apparent_side_px', 1), row['gt_count'],
                row['tp_at_4m'], fmt('retained_lt_0.5m_pct', 1),
                fmt('mean_abs_radial_m', 2), fmt('mean_abs_tangential_m', 2)))
    lines += ['', 'Physical 3D box dimensions are not included in this report. '
              'Projected area also depends on viewing angle, occlusion, and '
              'clipping; the table is descriptive rather than causal.', '']
    path.write_text('\n'.join(lines))


def analyze_csv(input_path, output_dir, model, range_width=5):
    if 50 % range_width:
        raise ValueError('range_width must divide 50')
    groups, outside = read_gt(input_path, range_width)
    records = summaries(groups)
    comparison = contrasts(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / 'apparent_size_tertiles.csv', FIELDS, records)
    write_csv(output_dir / 'apparent_size_contrasts.csv', CONTRAST_FIELDS, comparison)
    draw_svg(output_dir / 'apparent_size_retention_delta.svg', comparison,
             range_width, model)
    write_report(output_dir / 'apparent_size_summary.md', model, records,
                 comparison, outside, range_width)
    print('Apparent-size analysis: {} class/range contrasts; {} GT outside camera views'.format(
        len(comparison), outside))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='gt_geometry.csv')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--model', default='PETR')
    parser.add_argument('--range-width', type=int, default=5)
    args = parser.parse_args()
    analyze_csv(args.input, args.output_dir or args.input.parent,
                args.model, args.range_width)


if __name__ == '__main__':
    main()
