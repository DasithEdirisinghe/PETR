#!/usr/bin/env python3
"""A fixed-cohort, question-first visualization of saved car outcomes."""

import argparse
import csv
import html
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / 'experiments/cross_rig_attention/output/r1f_val/failure_modes'
NAVY, MUTED, BG = '#183047', '#63778b', '#f3f6f9'
TEAL, RED, GOLD, PURPLE, GREY = ('#168176', '#c9574c', '#e7a946',
                                   '#9a70a6', '#a9b8c7')
COLOR = {'car_correct': TEAL, 'car_mislocalized': RED,
         'car_low_score': GOLD, 'car_joint': PURPLE,
         'other_class_near': GREY, 'no_candidate': GREY}
THRESHOLDS = (0.5, 1.0, 2.0, 4.0)


def txt(x, y, value, size=20, color=NAVY, weight=400, anchor='start'):
    return ('<text x="{:.1f}" y="{:.1f}" fill="{}" font-size="{}" '
            'font-family="Arial, Helvetica, sans-serif" font-weight="{}" '
            'text-anchor="{}">{}</text>'.format(
                x, y, color, size, weight, anchor, html.escape(str(value))))


def box(x, y, w, h, fill='white', radius=18):
    return '<rect x="{}" y="{}" width="{}" height="{}" rx="{}" fill="{}"/>'.format(
        x, y, w, h, radius, fill)


def line(x1, y1, x2, y2, stroke='#d7e0e8', width=1):
    return '<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="{}"/>'.format(
        x1, y1, x2, y2, stroke, width)


def load(csv_path):
    by_threshold = {threshold: {} for threshold in THRESHOLDS}
    with csv_path.open(newline='') as handle:
        for row in csv.DictReader(handle):
            threshold = float(row['threshold_m'])
            key = (row['sample_token'], int(row['gt_index']))
            if key in by_threshold[threshold]:
                raise ValueError('Duplicate GT car and threshold: {}'.format(key))
            by_threshold[threshold][key] = row
    keys = set(by_threshold[4.0])
    if any(set(by_threshold[t]) != keys for t in THRESHOLDS):
        raise ValueError('The four thresholds do not contain the same GT cars')
    cohort = {key for key, row in by_threshold[4.0].items()
              if int(row['r1f_diagnostic_hit'])}
    if not cohort:
        raise ValueError('No R1-f-detected cars in saved outcomes')
    return by_threshold, cohort


def counts(by_threshold, cohort, threshold, score_threshold):
    rows = [by_threshold[threshold][key] for key in cohort]
    result = Counter()
    for row in rows:
        distance = row['r1_nearest_car_distance_m']
        score = row['r1_nearest_car_score']
        if not distance:
            result['no_candidate'] += 1
        else:
            high = float(score) >= score_threshold
            near = float(distance) < threshold
            category = ('car_correct' if high and near else
                        'car_mislocalized' if high else
                        'car_low_score' if near else 'car_joint')
            result[category] += 1
    return result, sum(
        int(row['r1_diagnostic_hit']) for row in rows), sum(
            int(row['r1f_diagnostic_hit']) for row in rows)


def card(parts, x, y, title, count, n, color, subtitle, width=345):
    parts.extend([box(x, y, width, 173), box(x, y, 8, 173, color, 4),
                  txt(x+24, y+40, title, 19, NAVY, 700),
                  txt(x+24, y+102, '{:,}'.format(count), 45, color, 700),
                  txt(x+width-24, y+100, '{:.1f}%'.format(100*count/n),
                      23, color, 700, 'end'),
                  txt(x+24, y+143, subtitle, 15, MUTED)])


def make_svg(by_threshold, cohort, summary, output):
    n = len(cohort)
    at_one, r1_hit, r1f_hit = counts(
        by_threshold, cohort, 1.0, summary['score_threshold'])
    no_car = at_one['no_candidate']
    no_car_rows = [by_threshold[1.0][key] for key in cohort
                   if not by_threshold[1.0][key]['r1_nearest_car_distance_m']]
    other_class = sum(row['r1_category'] == 'other_class_near'
                      for row in no_car_rows)
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1600" '
             'height="1220" viewBox="0 0 1600 1220">',
             box(0, 0, 1600, 1220, BG, 0),
             txt(54, 67, 'When R1-f detects a car, what does R1 output?',
                 33, NAVY, 700),
             txt(54, 104, 'Same R1-f validation frames and calibration · '
                 'fixed physical-car cohort · saved predictions only', 18, MUTED),
             box(54, 136, 1492, 111),
             txt(82, 180, 'REFERENCE COHORT', 18, MUTED, 700),
             txt(82, 220, '{:,} GT cars'.format(n), 31, TEAL, 700),
             txt(410, 181, 'R1-f car score ≥ {:.2f} and center error < 4 m'.format(
                 summary['score_threshold']), 22, NAVY, 700),
             txt(410, 218, 'The denominator stays the same at every threshold below.',
                 17, MUTED),
             txt(54, 294, 'At the strict 1 m threshold', 27, NAVY, 700),
             txt(54, 323, 'One nearest R1 car-labelled candidate per GT car: '
                 'score ≥ {:.2f}? center error < 1 m?'.format(
                     summary['score_threshold']), 17, MUTED),
             txt(62, 368, 'R1 CAR SCORE', 16, MUTED, 700),
             txt(425, 368, 'CENTER < 1 m', 16, MUTED, 700),
             txt(821, 368, 'CENTER ≥ 1 m, WITHIN {:.0f} m'.format(
                 summary['nearby_radius_m']), 16, MUTED, 700)]
    card(parts, 54, 386, 'Score passes + center passes',
         at_one['car_correct'], n, TEAL, 'Car-labelled candidate within 1 m', 710)
    card(parts, 790, 386, 'Score passes + center fails',
         at_one['car_mislocalized'], n, RED,
         'Car-labelled candidate is 1–{:.0f} m away'.format(
             summary['nearby_radius_m']), 756)
    card(parts, 54, 578, 'Score low + center passes',
         at_one['car_low_score'], n, GOLD, 'Car-labelled, but below score cutoff', 710)
    card(parts, 790, 578, 'Score low + center fails',
         at_one['car_joint'], n, PURPLE,
         'Both diagnostic criteria fail', 756)
    parts.extend([box(54, 770, 1492, 93),
                  box(54, 770, 8, 93, GREY, 4),
                  txt(78, 808, 'No qualifying car-labelled candidate within {:.0f} m'.format(
                      summary['nearby_radius_m']), 21, NAVY, 700),
                  txt(78, 839, '{} have a nearby other-class output; '
                      '{} have no qualifying nearby output.'.format(
                          other_class, no_car-other_class),
                      15, MUTED),
                  txt(1512, 828, '{:,}  |  {:.1f}%'.format(
                      no_car, 100*no_car/n), 29, GREY, 700, 'end'),
                  txt(54, 920, 'What changes when the center tolerance changes?',
                      25, NAVY, 700),
                  txt(54, 951, 'Same {:.0f} reference cars throughout; each R1 category '
                      'uses its nearest car-labelled output.'.format(n), 16, MUTED)])
    columns = [82, 260, 470, 688, 905, 1128, 1367]
    headings = ['Threshold', 'Score + center pass', 'Score pass, center fail',
                'Score low, center pass', 'Score + center fail',
                'No car candidate', 'R1-f detected']
    parts.append(box(54, 972, 1492, 39, '#e4ebf1', 7))
    for x, heading in zip(columns, headings):
        parts.append(txt(x, 999, heading, 14, NAVY, 700))
    for row_idx, threshold in enumerate(THRESHOLDS):
        y = 1045+row_idx*37
        c, _, r1f = counts(by_threshold, cohort, threshold,
                           summary['score_threshold'])
        numbers = [c['car_correct'], c['car_mislocalized'], c['car_low_score'],
                   c['car_joint'], c['no_candidate'], r1f]
        parts.append(line(58, y+10, 1540, y+10))
        parts.append(txt(columns[0], y, '{:.1f} m'.format(threshold), 17,
                         NAVY, 700))
        for index, (x, value) in enumerate(zip(columns[1:], numbers)):
            color = (TEAL, RED, GOLD, PURPLE, GREY, TEAL)[index]
            parts.append(txt(x, y, '{} ({:.1f}%)'.format(
                value, 100*value/n), 16, color, 700))
    parts += [txt(54, 1202, 'Nearest-output association is diagnostic, not query identity. '
                  'A high-score misplaced car supports localization loss; absence '
                  'does not prove misclassification.', 15, MUTED),
              '</svg>']
    output.write_text('\n'.join(parts))
    return {'fixed_cohort_n': n,
            'r1f_diagnostic_hit_1m': r1f_hit,
            'r1_diagnostic_hit_1m': r1_hit,
            'r1_1m_categories': dict(at_one)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=DEFAULT)
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()
    with (args.input_dir / 'summary.json').open() as handle:
        summary = json.load(handle)
    by_threshold, cohort = load(args.input_dir / 'per_car_threshold_outcomes.csv')
    output = args.output or args.input_dir / 'r1f_car_question.svg'
    output.parent.mkdir(parents=True, exist_ok=True)
    facts = make_svg(by_threshold, cohort, summary, output)
    with output.with_suffix('.json').open('w') as handle:
        json.dump(facts, handle, indent=2)
    print('Saved:', output)


if __name__ == '__main__':
    main()
