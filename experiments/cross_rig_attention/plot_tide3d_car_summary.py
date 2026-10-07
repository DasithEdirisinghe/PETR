#!/usr/bin/env python3
"""Plot the paired R1-f-frame car AP failure audit as one SVG dashboard."""

import argparse
import html
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AUDIT = (ROOT / 'experiments/cross_rig_attention/output/r1f_val/'
                 'car_ap_rank_audit/tide3d_car_full')


def esc(value):
    return html.escape(str(value), quote=True)


def rect(parts, x, y, width, height, fill, radius=0, stroke=None):
    attrs = ' stroke="{}"'.format(esc(stroke)) if stroke else ''
    parts.append('<rect x="{}" y="{}" width="{}" height="{}" rx="{}" '
                 'fill="{}"{}/>'.format(x, y, width, height, radius,
                                      esc(fill), attrs))


def label(parts, x, y, value, size=18, color='#183047', weight='normal',
          anchor='start'):
    parts.append('<text x="{}" y="{}" text-anchor="{}" font-family="Arial" '
                 'font-size="{}" font-weight="{}" fill="{}">{}</text>'.format(
                     x, y, anchor, size, weight, esc(color), esc(value)))


def bar_list(parts, x, y, rows, max_count, bar_x, bar_width, colors):
    for index, (key, title, count) in enumerate(rows):
        line = y + 76*index
        label(parts, x, line, title, 17)
        rect(parts, bar_x, line-18, bar_width, 20, '#e7edf2', 4)
        width = bar_width * count / max_count
        rect(parts, bar_x, line-18, max(width, 2 if count else 0), 20,
             colors[key], 4)
        label(parts, bar_x+bar_width+13, line, format(count, ','), 18,
              weight='bold')


def make_svg(summary):
    native = summary['models']['R1-f']
    cross = summary['models']['R1']
    transitions = summary['paired_gt_transitions']
    native_only = summary['paired_gt_transition_categories']['native_only__R1']
    fp = cross['car_fp_early_categories']
    assert sum(transitions.values()) == summary['gt_cars']
    assert sum(native_only.values()) == transitions['native_only']
    assert sum(fp.values()) == cross['car_fp_before_25pct_recall']
    native_ap = 100*native['official_car_ap_4m']
    cross_ap = 100*cross['official_car_ap_4m']
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1500" height="1000" '
             'viewBox="0 0 1500 1000">']
    rect(parts, 0, 0, 1500, 1000, '#f2f6fa')
    label(parts, 52, 56, 'Why does R1 lose car AP on R1-f frames?', 32,
          weight='bold')
    label(parts, 52, 88, 'Paired GT cars and score-ranked car false positives '
          '· official BEV-center match <4 m', 17, '#597083')

    cards = ((52, 'R1-f native car AP@4 m', native_ap, '#168176'),
             (540, 'R1 cross-rig car AP@4 m', cross_ap, '#c9574c'),
             (1028, 'AP gap', native_ap-cross_ap, '#375e89'))
    for x, title, value, color in cards:
        rect(parts, x, 117, 420, 121, 'white', 16)
        rect(parts, x, 117, 8, 121, color, 4)
        label(parts, x+26, 154, title, 17, '#526779')
        label(parts, x+26, 212, '{:.2f}{}'.format(value,
              ' pp' if title == 'AP gap' else '%'), 49, color, 'bold')

    label(parts, 52, 284, 'Same 3,591 GT cars: paired detection outcomes',
          22, weight='bold')
    rect(parts, 52, 303, 1396, 57, 'white', 12)
    segments = (
        ('both_tp', 'Detected by both', '#2b8f86'),
        ('native_only', 'Native only', '#d9a02d'),
        ('cross_only', 'Cross only', '#647fb7'),
        ('neither', 'Missed by both', '#9caaba'),
    )
    cursor = 52
    for key, title, color in segments:
        width = 1396*transitions[key]/summary['gt_cars']
        rect(parts, round(cursor, 2), 303, round(width, 2), 57, color)
        if width >= 100:
            label(parts, round(cursor+width/2, 2), 339,
                  format(transitions[key], ','), 19, 'white', 'bold', 'middle')
        cursor += width
    legend_x = 60
    for key, title, color in segments:
        rect(parts, legend_x, 376, 17, 17, color, 3)
        label(parts, legend_x+25, 391,
              '{}: {:,}'.format(title, transitions[key]), 15)
        legend_x += 340

    rect(parts, 52, 418, 682, 480, 'white', 16)
    rect(parts, 766, 418, 682, 480, 'white', 16)
    label(parts, 79, 459, '{:,} native-detected cars missed by R1'.format(
        transitions['native_only']), 22, weight='bold')
    label(parts, 79, 486, 'Closest free final-box evidence for each GT car',
          15, '#61788b')
    left_rows = (
        ('localization_candidate', 'Car box 4–8 m away',
         native_only.get('localization_candidate', 0)),
        ('class_only_candidate', 'Non-car box within 4 m',
         native_only.get('class_only_candidate', 0)),
        ('class_and_localization_candidate', 'Non-car box 4–8 m away',
         native_only.get('class_and_localization_candidate', 0)),
        ('no_free_final_prediction_within_near_radius', 'No free box within 8 m',
         native_only.get('no_free_final_prediction_within_near_radius', 0)),
        ('matching_competition', 'Matching competition',
         native_only.get('matching_competition', 0)),
    )
    left_colors = {'localization_candidate': '#d59b2c',
                   'class_only_candidate': '#7564af',
                   'class_and_localization_candidate': '#537fb4',
                   'no_free_final_prediction_within_near_radius': '#8797a7',
                   'matching_competition': '#506779'}
    bar_list(parts, 79, 543, left_rows, 340, 358, 285, left_colors)

    label(parts, 794, 459, 'High-ranked R1 car false positives', 22,
          weight='bold')
    label(parts, 794, 486, 'Before 25% recall: R1 {:,} vs R1-f {:,}'.format(
        cross['car_fp_before_25pct_recall'],
        native['car_fp_before_25pct_recall']), 15, '#61788b')
    right_rows = (
        ('no_evaluator_gt_within_near_radius', 'No GT within 8 m',
         fp.get('no_evaluator_gt_within_near_radius', 0)),
        ('car_label_near_car_outside_4m', 'Near car, outside 4 m',
         fp.get('car_label_near_car_outside_4m', 0)),
        ('car_label_near_other_outside_4m', 'Near other-class GT, 4–8 m',
         fp.get('car_label_near_other_outside_4m', 0)),
        ('car_label_on_other_class_gt_candidate', 'On other-class GT, <4 m',
         fp.get('car_label_on_other_class_gt_candidate', 0)),
        ('duplicate_or_competing_car', 'Duplicate / competing',
         fp.get('duplicate_or_competing_car', 0)),
    )
    right_colors = {'no_evaluator_gt_within_near_radius': '#8797a7',
                    'car_label_near_car_outside_4m': '#d59b2c',
                    'car_label_near_other_outside_4m': '#537fb4',
                    'car_label_on_other_class_gt_candidate': '#7564af',
                    'duplicate_or_competing_car': '#506779'}
    bar_list(parts, 794, 543, right_rows, 420, 1077, 280, right_colors)

    label(parts, 53, 935,
          'The 8 m radius is diagnostic, not an AP threshold. Proximity suggests '
          'failure candidates; it does not prove object identity or cause.',
          16, '#536b80')
    label(parts, 53, 963, 'Both models’ untouched AP@4 m was replayed exactly '
          'for all nine classes. Oracle scenarios are separate post-hoc tests.',
          15, '#536b80')
    parts.append('</svg>')
    return '\n'.join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, default=DEFAULT_AUDIT / 'summary.json')
    parser.add_argument('--output', type=Path,
                        default=DEFAULT_AUDIT / 'paired_car_failure_summary.svg')
    args = parser.parse_args()
    with args.summary.open() as handle:
        summary = json.load(handle)
    if summary['diagnostic_near_radius_m'] != 8.0:
        raise ValueError('This figure labels the default 8 m diagnostic radius')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(make_svg(summary))
    print('Saved', args.output)


if __name__ == '__main__':
    main()
