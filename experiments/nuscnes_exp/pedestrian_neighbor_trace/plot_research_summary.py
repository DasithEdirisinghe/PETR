#!/usr/bin/env python3
"""Relate official pedestrian AP to the crowded groups shown in index.html."""

import csv
import html
import json
import math
import shutil
import subprocess
from pathlib import Path


THRESHOLDS = ('0.5', '1.0', '2.0', '4.0')
BLUE = '#2563eb'
TEAL = '#0f766e'
ORANGE = '#ea580c'
INK = '#17212b'
MUTED = '#52606d'


def svg_text(x, y, value, size=18, weight='normal', color=INK, anchor='start'):
    return '<text x="{}" y="{}" font-family="Arial,sans-serif" font-size="{}" font-weight="{}" fill="{}" text-anchor="{}">{}</text>'.format(
        x, y, size, weight, color, anchor, html.escape(str(value)))


def main():
    base = Path(__file__).resolve().parent
    out = base / 'outputs' / 'visualizations'
    manifest = list(csv.DictReader((out / 'manifest.csv').open()))
    selected = {row['sample_token'] for row in manifest}
    frame_number = {row['sample_token']: int(row['index']) for row in manifest}
    metrics_path = base.parent / 'localization_oracle' / 'outputs' / 'baseline' / 'baseline' / 'metrics_summary.json'
    metrics = json.loads(metrics_path.read_text())
    official_ap = [100 * metrics['label_aps']['pedestrian'][t] for t in THRESHOLDS]

    all_gt = 0
    selected_gt = 0
    selected_tp = [0] * 4
    selected_fp = [0] * 4
    selected_outside = [0] * 4
    all_tp_by_threshold = [0] * 4
    nonnearest = [0] * 4
    claimed_earlier = [0] * 4
    review = []
    group_count = 0
    for group in map(json.loads, (base / 'outputs' / 'groups.jsonl').open()):
        group_count += 1
        n_gt = len(group['ground_truth'])
        all_gt += n_gt
        is_selected = group['sample_token'] in selected
        if is_selected:
            selected_gt += n_gt
        gt_ids = {gt['gt_index'] for gt in group['ground_truth']}
        pairs = {(pair['prediction_index'], pair['gt_index']): pair
                 for pair in group['pair_matrix']} if is_selected else {}
        ranks = {pred['prediction_index']: pred['local_score_rank']
                 for pred in group['predictions']} if is_selected else {}
        for ti, threshold in enumerate(THRESHOLDS):
            outcome = group['thresholds'][threshold]
            summary = outcome['summary']
            all_tp_by_threshold[ti] += summary['official_tp_gt_count']
            if not is_selected:
                continue
            selected_tp[ti] += summary['official_tp_gt_count']
            selected_fp[ti] += summary['local_prediction_fp']
            selected_outside[ti] += summary['local_prediction_matched_elsewhere']
            assignment = {int(pi): int(gi) for pi, gi in
                          outcome['official_prediction_to_gt'].items()
                          if gi is not None}
            owner = {gi: pi for pi, gi in assignment.items()}
            for pred in group['predictions']:
                pi = pred['prediction_index']
                assigned = assignment.get(pi)
                if assigned not in gt_ids:
                    continue
                nearest = min(gt_ids, key=lambda gi: pairs[pi, gi]['center_distance_m'])
                if nearest == assigned:
                    continue
                nonnearest[ti] += 1
                earlier = nearest in owner and ranks.get(owner[nearest], math.inf) < ranks[pi]
                claimed_earlier[ti] += earlier
                review.append({
                    'frame': frame_number[group['sample_token']],
                    'sample_token': group['sample_token'],
                    'threshold_m': threshold,
                    'prediction': 'P{}'.format(pi),
                    'confidence_rank_in_group': ranks[pi],
                    'score': pred['score'],
                    'assigned_gt': 'G{}'.format(assigned),
                    'assigned_center_distance_m': pairs[pi, assigned]['center_distance_m'],
                    'nearest_group_gt': 'G{}'.format(nearest),
                    'nearest_center_distance_m': pairs[pi, nearest]['center_distance_m'],
                    'nearest_gt_claimed_by_higher_rank_prediction': earlier,
                })
    if len(selected) != len(manifest) or group_count < len(selected):
        raise ValueError('Manifest or group data is incomplete')
    if any(tp > selected_gt for tp in selected_tp):
        raise ValueError('Selected TP count exceeds selected GT count')
    if not all(a == b for a, b in zip(nonnearest, claimed_earlier)):
        raise ValueError('Unexpected non-nearest assignment without an earlier owner')

    table_path = out / 'pedestrian_ap_scene_summary.csv'
    with table_path.open('w', newline='') as handle:
        fields = ['threshold_m', 'official_pedestrian_ap_percent', 'selected_gt',
                  'selected_group_tp', 'selected_group_fn', 'selected_group_tp_rate_percent',
                  'all_crowded_gt', 'all_crowded_tp', 'all_crowded_tp_rate_percent',
                  'selected_local_fp', 'selected_local_tp_outside_group',
                  'selected_tp_assigned_non_nearest_group_gt']
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i, threshold in enumerate(THRESHOLDS):
            writer.writerow({
                'threshold_m': threshold,
                'official_pedestrian_ap_percent': official_ap[i],
                'selected_gt': selected_gt,
                'selected_group_tp': selected_tp[i],
                'selected_group_fn': selected_gt - selected_tp[i],
                'selected_group_tp_rate_percent': 100 * selected_tp[i] / selected_gt,
                'all_crowded_gt': all_gt,
                'all_crowded_tp': all_tp_by_threshold[i],
                'all_crowded_tp_rate_percent': 100 * all_tp_by_threshold[i] / all_gt,
                'selected_local_fp': selected_fp[i],
                'selected_local_tp_outside_group': selected_outside[i],
                'selected_tp_assigned_non_nearest_group_gt': nonnearest[i],
            })
    review_path = out / 'non_nearest_tp_review.csv'
    with review_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(review[0]))
        writer.writeheader()
        writer.writerows(sorted(review, key=lambda row: (row['frame'], float(row['threshold_m']), row['confidence_rank_in_group'])))

    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1500" height="1010" viewBox="0 0 1500 1010">',
           '<rect width="1500" height="1010" fill="white"/>',
           svg_text(48, 56, 'Pedestrian AP and crowded-scene matching', 32, 'bold'),
           svg_text(48, 89, 'Official full-validation AP above; selected visible close-range scenes are diagnostic examples.', 18, color=MUTED)]
    panels = [(48, 125, 690, 360), (762, 125, 690, 360),
              (48, 515, 690, 340), (762, 515, 690, 340)]
    for x, y, w, h in panels:
        svg.append('<rect x="{}" y="{}" width="{}" height="{}" rx="10" fill="#f8fafc" stroke="#cbd5e1"/>'.format(x, y, w, h))

    # Panel A: official AP, which includes confidence ordering and false positives.
    svg.append(svg_text(74, 162, 'A. Official pedestrian AP', 22, 'bold'))
    svg.append(svg_text(74, 187, 'Complete validation set · center-distance criterion', 15, color=MUTED))
    for i, value in enumerate(official_ap):
        x = 118 + i * 145
        height = 218 * value / 100
        svg.append('<rect x="{}" y="{:.1f}" width="78" height="{:.1f}" rx="4" fill="{}"/>'.format(x, 421-height, height, TEAL))
        svg.append(svg_text(x+39, 413-height, '{:.1f}%'.format(value), 18, 'bold', anchor='middle'))
        svg.append(svg_text(x+39, 451, '{} m'.format(THRESHOLDS[i].rstrip('0').rstrip('.')), 17, anchor='middle'))
    svg.append('<line x1="91" y1="421" x2="700" y2="421" stroke="#94a3b8"/>')

    # Panel B: match coverage in selected and all analyzed crowded groups.
    svg.append(svg_text(788, 162, 'B. GT matched at each threshold', 22, 'bold'))
    svg.append(svg_text(788, 187, 'GT match fraction, not AP', 15, color=MUTED))
    svg.append('<line x1="808" y1="421" x2="1414" y2="421" stroke="#94a3b8"/>')
    for tick in (0, 50, 100):
        y = 421 - 2.14 * tick
        svg.append('<line x1="808" y1="{:.1f}" x2="1414" y2="{:.1f}" stroke="#e2e8f0"/>'.format(y, y))
        svg.append(svg_text(800, y+5, '{}%'.format(tick), 14, color=MUTED, anchor='end'))
    xs = [847 + i*182 for i in range(4)]
    series = [(selected_tp, selected_gt, BLUE), (all_tp_by_threshold, all_gt, ORANGE)]
    for counts, denominator, stroke in series:
        points = [(x, 421-214*count/denominator) for x, count in zip(xs, counts)]
        svg.append('<polyline points="{}" fill="none" stroke="{}" stroke-width="4"/>'.format(
            ' '.join('{:.1f},{:.1f}'.format(x, y) for x, y in points), stroke))
        for x, y in points:
            svg.append('<circle cx="{:.1f}" cy="{:.1f}" r="6" fill="{}" stroke="white" stroke-width="2"/>'.format(x, y, stroke))
    for x, threshold in zip(xs, THRESHOLDS):
        svg.append(svg_text(x, 451, '{} m'.format(threshold.rstrip('0').rstrip('.')), 17, anchor='middle'))
    svg.append('<line x1="822" y1="474" x2="852" y2="474" stroke="{}" stroke-width="4"/>'.format(BLUE))
    svg.append(svg_text(860, 479, '60 displayed groups ({} GT)'.format(selected_gt), 15))
    svg.append('<line x1="1132" y1="474" x2="1162" y2="474" stroke="{}" stroke-width="4"/>'.format(ORANGE))
    svg.append(svg_text(1170, 479, 'all {} groups'.format(group_count), 15))

    # Panel C: association ambiguity in the visible selected groups.
    svg.append(svg_text(74, 552, 'C. Assigned TP is not nearest group GT', 21, 'bold'))
    svg.append(svg_text(74, 578, 'Same-class center geometry in the 60 displayed groups', 15, color=MUTED))
    for i, (count, tp) in enumerate(zip(nonnearest, selected_tp)):
        x = 112 + i*146
        height = 196*count/max(selected_tp)
        svg.append('<rect x="{}" y="{:.1f}" width="78" height="{:.1f}" rx="4" fill="{}"/>'.format(x, 791-height, height, ORANGE))
        svg.append(svg_text(x+39, 781-height, '{} / {}'.format(count, tp), 16, 'bold', anchor='middle'))
        svg.append(svg_text(x+39, 817, '{} m'.format(THRESHOLDS[i].rstrip('0').rstrip('.')), 17, anchor='middle'))
    svg.append('<line x1="91" y1="791" x2="700" y2="791" stroke="#94a3b8"/>')

    # Panel D: interpretation limits and the next image-level check.
    svg.append(svg_text(788, 552, 'D. Research interpretation', 22, 'bold'))
    observations = [
        ('1', 'AP rises {:.1f}% → {:.1f}% as tolerance grows.'.format(official_ap[0], official_ap[-1])),
        ('', 'This is consistent with strong center-error sensitivity.'),
        ('2', 'All {} displayed GTs match at 4 m; {} match at 0.5 m.'.format(selected_gt, selected_tp[0])),
        ('', 'These visible ≤20 m scenes are a selected sample.'),
        ('3', '{} / {} TPs at 4 m have a nearer group GT.'.format(nonnearest[-1], selected_tp[-1])),
        ('', 'Each nearer GT was claimed by a higher-rank prediction.'),
        ('4', 'Image-level identity still needs visual or mask review.'),
    ]
    y = 590
    for number, sentence in observations:
        if number:
            svg.append(svg_text(790, y, number + '.', 18, 'bold', TEAL))
        svg.append(svg_text(820, y, sentence, 17, 'bold' if number else 'normal', INK if number else MUTED))
        y += 33
    svg.append(svg_text(48, 915, 'Scope: official AP uses the full validation set and score-ranked TP/FP precision–recall; panels B–C show selected GT/prediction counts.', 16, color=MUTED))
    svg.append(svg_text(48, 941, 'A non-nearest official assignment is a geometric warning flag, not proof that the image detection belongs to another person.', 16, color=MUTED))
    svg.append(svg_text(48, 972, 'Sources: baseline official metrics_summary.json and pedestrian_neighbor_trace groups.jsonl / visualizations/manifest.csv.', 14, color=MUTED))
    svg.append('</svg>')
    svg_path = out / 'pedestrian_ap_scene_summary.svg'
    svg_path.write_text('\n'.join(svg) + '\n')
    converter = shutil.which('rsvg-convert')
    if converter:
        with (out / 'pedestrian_ap_scene_summary.png').open('wb') as handle:
            subprocess.run([converter, '-f', 'png', str(svg_path)], check=True, stdout=handle)
    print('Selected groups: {}, GT: {}, all crowded GT: {}'.format(len(selected), selected_gt, all_gt))
    print('Official AP:', [round(x, 2) for x in official_ap])
    print('Selected TP:', selected_tp, 'non-nearest:', nonnearest)
    print(svg_path)


if __name__ == '__main__':
    main()
