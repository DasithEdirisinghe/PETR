#!/usr/bin/env python3
"""Experiment 2: apparent area versus center error within class, range, visibility.

Quartiles are assigned using all evaluator-eligible GT with a camera footprint.
Errors are measured on original 4 m matched predictions; coverage is reported
alongside them. The scene-cluster interval describes association, not causality.
"""

import argparse
import csv
import html
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median


CLASSES = ('car', 'truck', 'bus', 'trailer', 'construction_vehicle',
           'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier')
COLORS = ('#3178B8', '#E69F00', '#009E73', '#CC6677', '#7855A6')
OBJECT_FIELDS = ('sample_token', 'scene_token', 'gt_token', 'class',
                 'gt_range_m', 'range_bin_m', 'visibility_token',
                 'best_camera', 'camera_depth_m', 'apparent_width_px',
                 'apparent_height_px', 'apparent_area_px2',
                 'relative_orientation_deg', 'off_axis_angle_deg',
                 'gt_length_m', 'gt_width_m', 'gt_height_m',
                 'quartile_all_visibility', 'quartile_within_visibility',
                 'matched_at_4m', 'score', 'center_error_m',
                 'radial_error_m', 'tangential_error_m')
SUMMARY_FIELDS = ('class', 'range_bin_m', 'visibility', 'area_quartile',
                  'gt_count', 'tp_at_4m', 'coverage_at_4m_pct',
                  'median_apparent_area_px2', 'median_apparent_width_px',
                  'median_apparent_height_px', 'mean_gt_range_m',
                  'mean_camera_depth_m', 'mean_relative_orientation_deg',
                  'mean_gt_length_m', 'mean_gt_width_m', 'mean_gt_height_m',
                  'mean_center_error_m', 'mean_abs_radial_m',
                  'mean_abs_tangential_m', 'tp_lt_0.5m',
                  'strict_gt_success_pct', 'retained_lt_0.5m_pct')
CONTRAST_FIELDS = ('class', 'range_bin_m', 'visibility',
                   'q1_gt', 'q4_gt', 'q1_tp_at_4m', 'q4_tp_at_4m',
                   'q1_mean_center_error_m', 'q4_mean_center_error_m',
                   'q4_minus_q1_center_error_m', 'scene_cluster_se_m',
                   'ci95_low_m', 'ci95_high_m', 'scene_count',
                   'ci95_excludes_zero', 'q4_minus_q1_abs_radial_m',
                   'q4_minus_q1_abs_tangential_m',
                   'q1_coverage_at_4m_pct', 'q4_coverage_at_4m_pct',
                   'q1_mean_gt_range_m', 'q4_mean_gt_range_m',
                   'q1_mean_gt_length_m', 'q4_mean_gt_length_m',
                   'at_least_30_tp_each')
ADJUSTED_FIELDS = ('class', 'matched_4m_count', 'scene_count',
                   'center_error_change_per_area_doubling_m',
                   'scene_cluster_se_m', 'ci95_low_m', 'ci95_high_m',
                   'adjusted_for')


def iter_json_array(path):
    """Stream a large top-level JSON array without loading it into RAM."""
    decoder = json.JSONDecoder()
    with path.open('r', encoding='utf-8', buffering=2 * 1024 * 1024) as handle:
        buffer, position, finished = '', 0, False
        while True:
            if position >= len(buffer):
                chunk = handle.read(2 * 1024 * 1024)
                if not chunk:
                    break
                buffer, position = chunk, 0
            while position < len(buffer) and buffer[position] in ' \r\n\t,[]':
                position += 1
            if position >= len(buffer):
                continue
            try:
                value, end = decoder.raw_decode(buffer, position)
            except json.JSONDecodeError:
                chunk = handle.read(2 * 1024 * 1024)
                if not chunk:
                    raise ValueError('Incomplete JSON array in {}'.format(path))
                buffer, position = buffer[position:] + chunk, 0
                continue
            position = end
            yield value
            if position > 2 * 1024 * 1024:
                buffer, position = buffer[position:], 0


def metadata_maps(data_root, rows):
    needed_samples = {r['sample_token'] for r in rows}
    needed_annotations = {r['gt_token'] for r in rows}
    metadata_dir = data_root / 'v1.0-trainval'
    scenes = {}
    for record in iter_json_array(metadata_dir / 'sample.json'):
        token = record['token']
        if token in needed_samples:
            scenes[token] = record['scene_token']
    dimensions = {}
    for record in iter_json_array(metadata_dir / 'sample_annotation.json'):
        token = record['token']
        if token in needed_annotations:
            width, length, height = record['size']
            dimensions[token] = (length, width, height)
    if len(scenes) != len(needed_samples) or len(dimensions) != len(needed_annotations):
        raise ValueError('Could not map all sample/annotation tokens to metadata: '
                         '{} / {} scenes, {} / {} boxes'.format(
                             len(scenes), len(needed_samples),
                             len(dimensions), len(needed_annotations)))
    return scenes, dimensions


def range_bin(distance, width):
    low = int(distance // width) * width
    return '{}-{}'.format(low, low + width) if 0 <= low < 50 else None


def read_objects(path, range_width, data_root=None):
    rows, outside = [], 0
    with path.open(newline='', buffering=2 * 1024 * 1024) as handle:
        for source in csv.DictReader(handle):
            enriched = 'apparent_area_px2' in source
            area_key = 'apparent_area_px2' if enriched else 'projected_area_px2'
            if not source[area_key]:
                outside += 1
                continue
            label = range_bin(float(source['gt_range_m']), range_width)
            if label is None:
                continue
            rows.append({
                'sample_token': source['sample_token'],
                'scene_token': source.get('scene_token', ''),
                'gt_token': source['gt_token'], 'class': source['class'],
                'gt_range_m': float(source['gt_range_m']),
                'range_bin_m': label,
                'visibility_token': str(source['visibility_token']),
                'best_camera': source['best_camera'],
                'camera_depth_m': float(source['camera_depth_m']),
                'apparent_width_px': float(source[
                    'apparent_width_px' if enriched else 'projected_width_px']),
                'apparent_height_px': float(source[
                    'apparent_height_px' if enriched else 'projected_height_px']),
                'apparent_area_px2': float(source[area_key]),
                'relative_orientation_deg': float(source[
                    'relative_orientation_deg' if enriched else 'viewing_angle_deg']),
                'off_axis_angle_deg': float(source['off_axis_angle_deg']),
                'gt_length_m': source.get('gt_length_m', ''),
                'gt_width_m': source.get('gt_width_m', ''),
                'gt_height_m': source.get('gt_height_m', ''),
                'quartile_all_visibility': '',
                'quartile_within_visibility': '',
                'matched_at_4m': int(source['matched_at_4m']),
                'score': source['score'],
                'center_error_m': source['center_error_m'],
                'radial_error_m': source['radial_error_m'],
                'tangential_error_m': source['tangential_error_m'],
            })
    missing_scenes = any(not r['scene_token'] for r in rows)
    missing_dimensions = any(r['gt_length_m'] == '' for r in rows)
    if missing_scenes or missing_dimensions:
        if data_root is None:
            raise ValueError('Existing GT CSV lacks scene IDs or physical dimensions; '
                             'pass --data-root to backfill them')
        scenes, dimensions = metadata_maps(data_root, rows)
        for row in rows:
            row['scene_token'] = scenes[row['sample_token']]
            row['gt_length_m'], row['gt_width_m'], row['gt_height_m'] = (
                dimensions[row['gt_token']])
    for row in rows:
        for key in ('gt_length_m', 'gt_width_m', 'gt_height_m'):
            row[key] = float(row[key])
    return rows, outside


def assign_quartiles(rows, key):
    ordered = sorted(rows, key=lambda r: r['apparent_area_px2'])
    for index, row in enumerate(ordered):
        row[key] = 'Q{}'.format(min(4, 4 * index // len(ordered) + 1))


def avg(values):
    return sum(values) / len(values) if values else ''


def summarize_group(name, distance, visibility, quartile, rows):
    matches = [r for r in rows if r['matched_at_4m']]
    strict = sum(r['center_error_m'] != '' and
                 float(r['center_error_m']) < 0.5 for r in matches)
    return {
        'class': name, 'range_bin_m': distance, 'visibility': visibility,
        'area_quartile': quartile, 'gt_count': len(rows),
        'tp_at_4m': len(matches),
        'coverage_at_4m_pct': 100 * len(matches) / len(rows),
        'median_apparent_area_px2': median(r['apparent_area_px2'] for r in rows),
        'median_apparent_width_px': median(r['apparent_width_px'] for r in rows),
        'median_apparent_height_px': median(r['apparent_height_px'] for r in rows),
        'mean_gt_range_m': avg([r['gt_range_m'] for r in rows]),
        'mean_camera_depth_m': avg([r['camera_depth_m'] for r in rows]),
        'mean_relative_orientation_deg': avg([r['relative_orientation_deg'] for r in rows]),
        'mean_gt_length_m': avg([r['gt_length_m'] for r in rows]),
        'mean_gt_width_m': avg([r['gt_width_m'] for r in rows]),
        'mean_gt_height_m': avg([r['gt_height_m'] for r in rows]),
        'mean_center_error_m': avg([float(r['center_error_m']) for r in matches]),
        'mean_abs_radial_m': avg([abs(float(r['radial_error_m'])) for r in matches]),
        'mean_abs_tangential_m': avg([abs(float(r['tangential_error_m'])) for r in matches]),
        'tp_lt_0.5m': strict,
        'strict_gt_success_pct': 100 * strict / len(rows),
        'retained_lt_0.5m_pct': 100 * strict / len(matches) if matches else '',
    }


def cluster_interval(q1, q4):
    """Scene-cluster robust 95% CI for mean(error Q4) minus mean(error Q1)."""
    q1 = [r for r in q1 if r['matched_at_4m']]
    q4 = [r for r in q4 if r['matched_at_4m']]
    if len(q1) < 2 or len(q4) < 2:
        return ('', '', '', 0)
    m1 = avg([float(r['center_error_m']) for r in q1])
    m4 = avg([float(r['center_error_m']) for r in q4])
    influence = defaultdict(float)
    for row in q1:
        influence[row['scene_token']] -= (float(row['center_error_m']) - m1) / len(q1)
    for row in q4:
        influence[row['scene_token']] += (float(row['center_error_m']) - m4) / len(q4)
    count = len(influence)
    if count < 2:
        return ('', '', '', count)
    se = math.sqrt(count / (count - 1) * sum(value * value for value in influence.values()))
    delta = m4 - m1
    return (se, delta - 1.96 * se, delta + 1.96 * se, count)


def summarize(rows):
    strata = defaultdict(list)
    for row in rows:
        strata[(row['class'], row['range_bin_m'], 'all')].append(row)
        strata[(row['class'], row['range_bin_m'], row['visibility_token'])].append(row)
    summary, contrasts = [], []
    for (name, distance, visibility), members in sorted(strata.items()):
        key = ('quartile_all_visibility' if visibility == 'all'
               else 'quartile_within_visibility')
        assign_quartiles(members, key)
        by_quartile = defaultdict(list)
        for row in members:
            by_quartile[row[key]].append(row)
        quartet = {}
        for number in range(1, 5):
            label = 'Q{}'.format(number)
            if by_quartile[label]:
                record = summarize_group(name, distance, visibility, label,
                                         by_quartile[label])
                summary.append(record)
                quartet[label] = record
        if 'Q1' not in quartet or 'Q4' not in quartet:
            continue
        q1, q4 = quartet['Q1'], quartet['Q4']
        se, low, high, scenes = cluster_interval(by_quartile['Q1'],
                                                  by_quartile['Q4'])
        enough = q1['tp_at_4m'] >= 30 and q4['tp_at_4m'] >= 30
        have_error = q1['tp_at_4m'] and q4['tp_at_4m']
        contrasts.append({
            'class': name, 'range_bin_m': distance,
            'visibility': visibility,
            'q1_gt': q1['gt_count'], 'q4_gt': q4['gt_count'],
            'q1_tp_at_4m': q1['tp_at_4m'], 'q4_tp_at_4m': q4['tp_at_4m'],
            'q1_mean_center_error_m': q1['mean_center_error_m'],
            'q4_mean_center_error_m': q4['mean_center_error_m'],
            'q4_minus_q1_center_error_m': (
                q4['mean_center_error_m'] - q1['mean_center_error_m']
                if have_error else ''),
            'scene_cluster_se_m': se, 'ci95_low_m': low,
            'ci95_high_m': high, 'scene_count': scenes,
            'ci95_excludes_zero': int(enough and scenes >= 10 and
                                     low != '' and (low > 0 or high < 0)),
            'q4_minus_q1_abs_radial_m': (
                q4['mean_abs_radial_m'] - q1['mean_abs_radial_m']
                if have_error else ''),
            'q4_minus_q1_abs_tangential_m': (
                q4['mean_abs_tangential_m'] - q1['mean_abs_tangential_m']
                if have_error else ''),
            'q1_coverage_at_4m_pct': q1['coverage_at_4m_pct'],
            'q4_coverage_at_4m_pct': q4['coverage_at_4m_pct'],
            'q1_mean_gt_range_m': q1['mean_gt_range_m'],
            'q4_mean_gt_range_m': q4['mean_gt_range_m'],
            'q1_mean_gt_length_m': q1['mean_gt_length_m'],
            'q4_mean_gt_length_m': q4['mean_gt_length_m'],
            'at_least_30_tp_each': int(enough),
        })
    return summary, contrasts


def invert_matrix(matrix):
    """Gauss-Jordan inverse of a small, full-rank symmetric matrix."""
    n = len(matrix)
    aug = [list(row) + [float(i == j) for j in range(n)]
           for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-9:
            raise ValueError('Adjusted regression has collinear features')
        aug[col], aug[pivot] = aug[pivot], aug[col]
        scale = aug[col][col]
        aug[col] = [value / scale for value in aug[col]]
        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            aug[row] = [a - factor * b for a, b in zip(aug[row], aug[col])]
    return [row[n:] for row in aug]


def adjusted_features(row):
    """Observed covariates; area coefficient is metres per area doubling."""
    distance = row['gt_range_m'] / 10
    camera = row['best_camera']
    values = [
        1.0,
        math.log2(row['apparent_area_px2']),
        distance, distance * distance,
        math.log2(row['gt_length_m']),
        math.log2(row['gt_width_m']),
        math.log2(row['gt_height_m']),
        row['relative_orientation_deg'] / 90,
        row['off_axis_angle_deg'] / 45,
        float(row['visibility_token'] == '2'),
        float(row['visibility_token'] == '3'),
        float(row['visibility_token'] == '4'),
    ]
    values.extend(float(camera == name) for name in
                  ('CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT',
                   'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT'))
    return values


def adjusted_analysis(rows):
    """Per-class OLS association, with scene-cluster sandwich intervals."""
    names = ('intercept', 'log2_apparent_area', 'exact_range', 'range_squared',
             'log2_gt_length', 'log2_gt_width', 'log2_gt_height',
             'relative_orientation', 'off_axis_angle',
             'visibility_2', 'visibility_3', 'visibility_4',
             'camera_back', 'camera_back_left', 'camera_back_right',
             'camera_front_left', 'camera_front_right')
    grouped = defaultdict(list)
    for row in rows:
        if row['matched_at_4m']:
            grouped[row['class']].append(row)
    output = []
    for name in CLASSES:
        members = grouped[name]
        if len(members) < 50:
            continue
        full_x = [adjusted_features(row) for row in members]
        active = [0, 1] + [j for j in range(2, len(names))
                           if max(x[j] for x in full_x) -
                           min(x[j] for x in full_x) > 1e-8]
        k = len(active)
        xtx = [[0.0] * k for _ in range(k)]
        xty = [0.0] * k
        design = []
        for row, x_all in zip(members, full_x):
            x = [x_all[j] for j in active]
            design.append(x)
            y = float(row['center_error_m'])
            for i in range(k):
                xty[i] += x[i] * y
                for j in range(i + 1):
                    xtx[i][j] += x[i] * x[j]
        for i in range(k):
            for j in range(i):
                xtx[j][i] = xtx[i][j]
        try:
            inverse = invert_matrix(xtx)
        except ValueError:
            continue
        beta = [sum(inverse[i][j] * xty[j] for j in range(k))
                for i in range(k)]
        scores = defaultdict(lambda: [0.0] * k)
        for row, x in zip(members, design):
            residual = float(row['center_error_m']) - sum(
                b * value for b, value in zip(beta, x))
            score = scores[row['scene_token']]
            for i in range(k):
                score[i] += x[i] * residual
        g = len(scores)
        if g < 2:
            continue
        # Variance of the area coefficient: a' (sum U_g U_g') a,
        # where a is the corresponding row of (X'X)^-1.
        area_row = inverse[1]
        variance = sum(sum(a * u for a, u in zip(area_row, score)) ** 2
                       for score in scores.values())
        variance *= g / (g - 1) * (len(members) - 1) / (len(members) - k)
        se = math.sqrt(max(0.0, variance))
        coefficient = beta[1]
        output.append({
            'class': name, 'matched_4m_count': len(members),
            'scene_count': g,
            'center_error_change_per_area_doubling_m': coefficient,
            'scene_cluster_se_m': se,
            'ci95_low_m': coefficient - 1.96 * se,
            'ci95_high_m': coefficient + 1.96 * se,
            'adjusted_for': ','.join(names[j] for j in active[2:]),
        })
    return output


def write_csv(path, fields, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def y_axis(value, top, bottom, maximum):
    return bottom - (bottom - top) * value / maximum


def draw_plot(path, summary, class_name, visibility, width):
    """Q1-Q4 line plot with distance groups and per-point 4 m TP counts."""
    ranges = ['{}-{}'.format(start, start + width)
              for start in range(0, 50, width)]
    plot = [r for r in summary if r['class'] == class_name and
            r['visibility'] == visibility]
    lookup = {(r['range_bin_m'], r['area_quartile']): r for r in plot}
    maximum = max([float(r['mean_center_error_m']) for r in plot
                   if r['mean_center_error_m'] != ''] or [1])
    maximum = max(1.0, math.ceil(maximum * 2) / 2)
    svg_width, svg_height = 980, 590
    left, right, top, bottom = 105, 780, 125, 475
    items = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="980" height="590" viewBox="0 0 980 590">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#263238}.title{font-size:21px;font-weight:bold}'
        '.sub{font-size:13px;fill:#59666f}.tick{font-size:12px}.legend{font-size:13px}</style>',
        '<text x="490" y="36" text-anchor="middle" class="title">{}: center error versus apparent area</text>'.format(
            html.escape(class_name.replace('_', ' '))),
        '<text x="490" y="61" text-anchor="middle" class="sub">Visibility {} · quartiles defined within each class, range, and visibility group</text>'.format(
            html.escape(visibility)),
        '<text x="490" y="82" text-anchor="middle" class="sub">Mean BEV center error among original 4 m TPs; n labels show TP counts</text>',
    ]
    for tick in range(0, 6):
        value = maximum * tick / 5
        y = y_axis(value, top, bottom, maximum)
        items.append('<line x1="{}" y1="{:.1f}" x2="{}" y2="{:.1f}" stroke="#e1e5e8"/>'.format(
            left, y, right, y))
        items.append('<text x="{}" y="{:.1f}" text-anchor="end" class="tick">{:.1f}</text>'.format(
            left - 9, y + 4, value))
    xs = [left + (right - left) * (i + 0.5) / 4 for i in range(4)]
    for i, x in enumerate(xs):
        items.append('<text x="{:.1f}" y="502" text-anchor="middle" class="tick">Q{}</text>'.format(x, i + 1))
    items.append('<text x="445" y="532" text-anchor="middle" class="sub">Apparent-area quartile (Q1 smallest, Q4 largest)</text>')
    items.append('<text x="24" y="300" transform="rotate(-90 24 300)" text-anchor="middle" class="sub">Mean BEV center error (m)</text>')
    for idx, distance in enumerate(ranges):
        color = COLORS[idx % len(COLORS)]
        points = []
        for q, x in enumerate(xs, 1):
            record = lookup.get((distance, 'Q{}'.format(q)))
            if not record or record['tp_at_4m'] < 10:
                continue
            y = y_axis(float(record['mean_center_error_m']), top, bottom, maximum)
            points.append((x, y, record))
        if len(points) >= 2:
            items.append('<polyline points="{}" fill="none" stroke="{}" stroke-width="2.6"/>'.format(
                ' '.join('{:.1f},{:.1f}'.format(x, y) for x, y, _ in points), color))
        for x, y, record in points:
            items.append('<circle cx="{:.1f}" cy="{:.1f}" r="5" fill="{}"><title>{}: Q{}, n={} TPs, mean={:.3f} m, GT={}</title></circle>'.format(
                x, y, color, distance, record['area_quartile'][1:],
                record['tp_at_4m'], record['mean_center_error_m'], record['gt_count']))
            items.append('<text x="{:.1f}" y="{:.1f}" text-anchor="middle" style="font-size:10px;fill:{}">n={}</text>'.format(
                x, y - 9, color, record['tp_at_4m']))
        legend_y = 145 + idx * 30
        items.append('<line x1="810" y1="{}" x2="835" y2="{}" stroke="{}" stroke-width="3"/>'.format(
            legend_y, legend_y, color))
        items.append('<text x="842" y="{}" class="legend">{} m</text>'.format(
            legend_y + 4, distance))
    items.append('<text x="490" y="568" text-anchor="middle" class="sub">Interpret with 4 m GT coverage and physical dimensions in the CSV; lines are descriptive.</text>')
    items.append('</svg>')
    path.write_text('\n'.join(items) + '\n')


def write_report(path, model, summary, contrasts, adjusted, outside, width):
    lines = [
        '# Experiment 2: apparent area and localization for {}'.format(model), '',
        'GT boxes are ranked by projected pixel area **within each class, '
        '{} m GT-range bin, and annotation visibility category**. Separate '
        '`all` visibility rows pool visibility categories. Quartiles use all '
        'eligible GT with a camera footprint; center errors use the original '
        '4 m matches only. {} GT outside the camera images are excluded.'.format(
            width, outside), '',
        'The outcome is mean BEV center error in metres. Each row also reports '
        '4 m GT coverage, mean radial/tangential error, exact mean GT range, '
        'viewing angle, and physical GT length/width/height. The Q4−Q1 '
        'interval uses scene-cluster standard errors and is shown only as '
        'an uncertainty measure for this association; repeated scenes, '
        'occlusion, physical size, orientation, and selection into 4 m TPs '
        'still limit causal interpretation. Projected GT box area is a '
        'geometric footprint, not a count of truly visible object pixels. '
        'Intervals are exploratory and are not corrected for the many '
        'class/range/visibility comparisons.', '',
        '## All visibility: Q4 versus Q1', '',
        '| Class | Range | Q1 / Q4 TPs | Q1 error | Q4 error | Q4−Q1 | 95% scene CI |',
        '|---|---|---:|---:|---:|---:|---:|',
    ]
    ordered = sorted((r for r in contrasts if r['visibility'] == 'all'
                      and r['at_least_30_tp_each']),
                     key=lambda r: (CLASSES.index(r['class']),
                                    int(r['range_bin_m'].split('-')[0])))
    for row in ordered:
        ci = ('[{:+.2f}, {:+.2f}]'.format(row['ci95_low_m'], row['ci95_high_m'])
              if row['ci95_low_m'] != '' and row['scene_count'] >= 10 else '—')
        lines.append('| {} | {} m | {:,} / {:,} | {:.2f} m | {:.2f} m | {:+.2f} m | {} |'.format(
            row['class'].replace('_', ' '), row['range_bin_m'],
            row['q1_tp_at_4m'], row['q4_tp_at_4m'],
            row['q1_mean_center_error_m'], row['q4_mean_center_error_m'],
            row['q4_minus_q1_center_error_m'], ci))
    lines += ['', '## Association after measured covariate adjustment', '',
              'This per-class linear model uses matched 4 m TPs and adjusts '
              'for exact GT range (linear and squared), physical dimensions, '
              'relative orientation, off-axis angle, visibility category, '
              'and camera. The coefficient is the change in mean center error '
              'for a doubling of apparent area. A negative value means '
              'larger projected area is associated with lower error after '
              'these adjustments. Scene-cluster 95% intervals quantify '
              'sampling uncertainty, not causal effects.', '',
              '| Class | 4 m TPs | Change per area doubling | 95% scene CI |',
              '|---|---:|---:|---:|']
    for row in adjusted:
        lines.append('| {} | {:,} | {:+.3f} m | [{:+.3f}, {:+.3f}] m |'.format(
            row['class'].replace('_', ' '), row['matched_4m_count'],
            row['center_error_change_per_area_doubling_m'],
            row['ci95_low_m'], row['ci95_high_m']))
    lines += ['', 'See `area_quartiles.csv` and `q4_vs_q1.csv` for visibility '
              'strata and covariate details. `plots/` contains one line chart '
              'per class for pooled visibility and each annotation visibility '
              'category. A lower Q4 error means the larger-image group has '
              'better center localization within that stratum.', '']
    path.write_text('\n'.join(lines))


def analyze_csv(input_path, output_dir, model, data_root=None, range_width=10):
    if 50 % range_width:
        raise ValueError('range_width must divide 50')
    rows, outside = read_objects(input_path, range_width, data_root)
    summary, contrasts = summarize(rows)
    adjusted = adjusted_analysis(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / 'objects.csv', OBJECT_FIELDS, rows)
    write_csv(output_dir / 'area_quartiles.csv', SUMMARY_FIELDS, summary)
    write_csv(output_dir / 'q4_vs_q1.csv', CONTRAST_FIELDS, contrasts)
    write_csv(output_dir / 'adjusted_area_association.csv', ADJUSTED_FIELDS,
              adjusted)
    write_report(output_dir / 'summary.md', model, summary, contrasts, adjusted,
                 outside, range_width)
    plots = output_dir / 'plots'
    plots.mkdir(exist_ok=True)
    for name in CLASSES:
        for visibility in ('all', '1', '2', '3', '4'):
            draw_plot(plots / '{}_visibility_{}.svg'.format(name, visibility),
                      summary, name, visibility, range_width)
    print('Experiment 2: {:,} objects, {} quartile rows, {} Q4/Q1 contrasts'.format(
        len(rows), len(summary), len(contrasts)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path,
                        help='Existing gt_geometry.csv from image geometry analysis')
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--model', required=True)
    parser.add_argument('--data-root', type=Path,
                        help='Needed if input lacks scene IDs or physical dimensions')
    parser.add_argument('--range-width', type=int, default=10)
    args = parser.parse_args()
    analyze_csv(args.input, args.output_dir, args.model,
                args.data_root, args.range_width)


if __name__ == '__main__':
    main()
