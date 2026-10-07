#!/usr/bin/env python3
"""Question-first SVG scorecard from full-validation query-level measurements."""

import argparse
import csv
import html
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / 'experiments/cross_rig_attention/output/r1f_val/query_scorecard'
COLORS = {'r1': '#c9574c', 'r1f': '#168176'}


def label(x, y, value, size=20, color='#183047', weight=400, anchor='start'):
    return ('<text x="{}" y="{}" fill="{}" font-size="{}" '
            'font-family="Arial,Helvetica,sans-serif" font-weight="{}" '
            'text-anchor="{}">{}</text>'.format(
                x, y, color, size, weight, anchor, html.escape(str(value))))


def rectangle(x, y, w, h, color='white', radius=16):
    return '<rect x="{}" y="{}" width="{}" height="{}" rx="{}" fill="{}"/>'.format(
        x, y, w, h, radius, color)


def number(value):
    return None if value in (None, '') else float(value)


def get_rows(path):
    with path.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    paired = {}
    for row in rows:
        key = (row['sample_token'], row['gt_index'])
        paired.setdefault(key, {})[row['model']] = row
    return {key: pair for key, pair in paired.items()
            if set(pair) == {'r1', 'r1f'}}


def metrics(paired, keys, model, score_cutoff, attention_cutoff):
    rows = [paired[key][model] for key in keys]
    n = len(rows)
    if not n:
        return {'n': 0, 'attention_auc_median': None,
                'attention_valid': 0, 'attention_pass': 0.0,
                'class_pass': 0.0, 'center_pass': 0.0,
                'class_and_center': 0.0, 'all_three': 0.0,
                'class_pass_count': 0,
                'class_pass_center_fail_count': 0,
                'class_fail_count': 0, 'center_pass_count': 0}
    attention = [number(row['attention_auc_l6']) for row in rows]
    valid = [value for value in attention if value is not None]
    cls = [row['top_class'] == 'car' and
           float(row['car_score']) >= score_cutoff for row in rows]
    loc = [float(row['center_error_m']) < 1 for row in rows]
    attended = [value is not None and value >= attention_cutoff
                for value in attention]
    return {'n': n, 'attention_auc_median': statistics.median(valid)
            if valid else None, 'attention_valid': len(valid),
            'attention_pass': sum(attended)/n,
            'class_pass': sum(cls)/n,
            'center_pass': sum(loc)/n,
            'class_and_center': sum(a and b for a, b in zip(cls, loc))/n,
            'all_three': sum(a and b and c for a, b, c in
                             zip(attended, cls, loc))/n,
            'class_pass_count': sum(cls),
            'class_pass_center_fail_count': sum(a and not b for a, b in
                                                zip(cls, loc)),
            'class_fail_count': sum(not a for a in cls),
            'center_pass_count': sum(loc)}


def make_figure(output, summary, score_cutoff, attention_cutoff):
    width, height = 1540, 1040
    elements = ['<svg xmlns="http://www.w3.org/2000/svg" width="{}" '
                'height="{}" viewBox="0 0 {} {}">'.format(
                    width, height, width, height),
                rectangle(0, 0, width, height, '#f3f6f9', 0),
                label(52, 69, 'Do R1 queries find the car but place it wrongly?',
                      33, weight=700),
                label(52, 105, 'Fresh inference · same R1-f frames/calibration · '
                      'GT-associated query in each frozen checkpoint', 18, '#61758a')]
    for section_index, (section, title) in enumerate([
            ('reference', 'Cars R1-f recognizes and localizes within 4 m'),
            ('all', 'All paired GT cars')]):
        top = 145+section_index*420
        n = summary[section]['r1']['n']
        elements.extend([rectangle(52, top, 1436, 390),
                         label(79, top+42, title, 25, weight=700),
                         label(1458, top+42, 'n={:,}'.format(n), 20,
                               '#61758a', 700, 'end'),
                         label(80, top+92, 'CHECKPOINT', 14, '#61758a', 700)])
        headings = [('attention_auc_median', 'Car-ray attention AUROC', 415),
                    ('class_pass', 'Car class + score', 740),
                    ('class_and_center', 'Class + center <1 m', 1020),
                    ('all_three', 'All three checks', 1308)]
        for _, title_text, x in headings:
            elements.append(label(x, top+92, title_text, 15, '#61758a', 700,
                                  'middle'))
        for index, model in enumerate(('r1', 'r1f')):
            y = top+153+index*98
            values = summary[section][model]
            elements.extend([rectangle(77, y-45, 1383, 80, '#f3f6f9', 9),
                             rectangle(77, y-45, 8, 80, COLORS[model], 3),
                             label(99, y+5, 'R1 checkpoint' if model == 'r1'
                                   else 'R1-f checkpoint', 20, COLORS[model], 700)])
            for field, _, x in headings:
                value = values[field]
                shown = ('n/a' if value is None else
                         '{:.3f}'.format(value) if field == 'attention_auc_median'
                         else '{:.1f}%'.format(100*value))
                elements.append(label(x, y+5, shown, 24, COLORS[model], 700,
                                      'middle'))
        ref_r1 = summary[section]['r1']
        elements.append(label(80, top+365,
                              'R1: {:,} car-class passes but center fails 1 m; '
                              '{:,} fail the car-class/score check.'.format(
                                  ref_r1['class_pass_center_fail_count'],
                                  ref_r1['class_fail_count']),
                              18, '#334b62'))
    elements.extend([
        label(54, 1004, 'Attention = L6 AUROC of GT-box-intersecting rays vs '
              'same-camera background (0.5 = chance). Class = top class car '
              'and score ≥ {:.2f}.'.format(score_cutoff), 15, '#61758a'),
        label(54, 1028, 'All three = attention AUROC ≥ {:.2f} + class pass + '
              'center <1 m. GT selects a query for diagnosis; it is not fed to '
              'the detector.'.format(attention_cutoff), 15, '#61758a'),
        '</svg>'])
    output.write_text('\n'.join(elements))


def make_camera_figure(input_dir, paired, reference):
    camera_path = input_dir / 'per_car_camera_attention.csv'
    if not camera_path.exists():
        return
    allowed = set(reference)
    groups = {}
    with camera_path.open(newline='') as handle:
        for row in csv.DictReader(handle):
            key = (row['sample_token'], row['gt_index'])
            if key not in allowed:
                continue
            group = (int(row['camera_index']), row['camera_name'], row['model'])
            groups.setdefault(group, []).append(row)
    camera_names = sorted({(index, name) for index, name, _ in groups})
    height = 250+len(camera_names)*112
    elements = ['<svg xmlns="http://www.w3.org/2000/svg" width="1540" '
                'height="{}" viewBox="0 0 1540 {}">'.format(height, height),
                rectangle(0, 0, 1540, height, '#f3f6f9', 0),
                label(52, 67, 'Where does the car query attend across cameras?',
                      32, weight=700),
                label(52, 103, 'Same fixed R1-f-detected GT-car cohort · '
                      'L6 cross-attention · red = R1, teal = R1-f',
                      18, '#61758a'),
                rectangle(52, 133, 1436, 54, '#e5ebf1', 8),
                label(75, 168, 'Camera', 16, weight=700),
                label(560, 168, 'All-query attention mass', 16,
                      weight=700, anchor='middle'),
                label(1090, 168, 'Car-ray attention AUROC', 16,
                      weight=700, anchor='middle'),
                label(1440, 168, 'Cars visible', 16,
                      weight=700, anchor='end')]
    for row_index, (camera_index, camera_name) in enumerate(camera_names):
        y = 225+row_index*112
        elements.extend([rectangle(52, y-29, 1436, 97),
                         label(75, y+9, camera_name, 19, weight=700),
                         label(75, y+38, 'camera {}'.format(camera_index),
                               14, '#61758a')])
        for model, offset in (('r1', 0), ('r1f', 36)):
            rows = groups[(camera_index, camera_name, model)]
            mass = statistics.mean(float(row['camera_attention_mass_l6'])
                                   for row in rows)
            aucs = [float(row['attention_auc_l6']) for row in rows
                    if row['attention_auc_l6'] not in ('', None)]
            auc = statistics.median(aucs) if aucs else None
            color = COLORS[model]
            yy = y-11+offset
            elements.extend([
                label(390, yy+14, 'R1' if model == 'r1' else 'R1-f',
                      15, color, 700),
                rectangle(448, yy, 205, 17, '#e8edf1', 5),
                rectangle(448, yy, min(205, 205*mass), 17, color, 5),
                label(674, yy+15, '{:.1f}%'.format(100*mass), 17, color, 700),
                label(1080, yy+15, 'n/a' if auc is None else '{:.3f}'.format(auc),
                      20, color, 700, 'middle'),
                label(1438, yy+15, '{:,}'.format(len(aucs)),
                      17, color, 700, 'end')])
    elements.extend([label(52, height-24,
                           'Attention mass includes all cameras. AUROC is only '
                           'defined where GT-car rays intersect that camera; '
                           'a non-visible camera is not a failed localization.',
                           15, '#61758a'), '</svg>'])
    (input_dir / 'camera_attention.svg').write_text('\n'.join(elements))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=DEFAULT)
    parser.add_argument('--score-threshold', type=float, default=.35)
    parser.add_argument('--attention-auc-threshold', type=float, default=.6)
    args = parser.parse_args()
    paired = get_rows(args.input_dir / 'per_car_query.csv')
    reference = [key for key, pair in paired.items()
                 if pair['r1f']['top_class'] == 'car' and
                 float(pair['r1f']['car_score']) >= args.score_threshold and
                 float(pair['r1f']['center_error_m']) < 4]
    summary = {'paired_gt_cars': len(paired), 'reference_gt_cars': len(reference),
               'score_threshold': args.score_threshold,
               'attention_auc_threshold': args.attention_auc_threshold,
               'reference': {}, 'all': {}}
    metadata = args.input_dir / 'run_metadata.json'
    if metadata.exists():
        with metadata.open() as handle:
            summary['run_metadata'] = json.load(handle)
    for section, keys in (('reference', reference), ('all', list(paired))):
        for model in ('r1', 'r1f'):
            summary[section][model] = metrics(
                paired, keys, model, args.score_threshold,
                args.attention_auc_threshold)
    with (args.input_dir / 'query_scorecard_summary.json').open('w') as handle:
        json.dump(summary, handle, indent=2)
    output = args.input_dir / 'query_scorecard.svg'
    make_figure(output, summary, args.score_threshold,
                args.attention_auc_threshold)
    make_camera_figure(args.input_dir, paired, reference)
    print('Saved:', output)


if __name__ == '__main__':
    main()
