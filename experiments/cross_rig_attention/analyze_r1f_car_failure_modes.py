#!/usr/bin/env python3
"""GT-centric R1/R1-f car outcome decomposition from saved detections only."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from compare_r1_r1f_cars import gt_frames


ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / 'experiments/cross_rig_attention/output/r1f_val'
THRESHOLDS = (0.5, 1.0, 2.0, 4.0)
ORDER = ('car_correct', 'car_mislocalized', 'car_low_score',
         'car_joint', 'other_class_near', 'no_candidate')
LABELS = {
    'car_correct': 'Car score passes; center passes',
    'car_mislocalized': 'Car score passes; center fails',
    'car_low_score': 'Center passes; car score low',
    'car_joint': 'Car score low; center fails',
         'other_class_near': 'Other-class output ≤2 m only',
    'no_candidate': 'No qualifying nearby output',
}
COLORS = {'car_correct': '#168176', 'car_mislocalized': '#c9574c',
          'car_low_score': '#e6a94b', 'car_joint': '#a56b9d',
          'other_class_near': '#7488bd', 'no_candidate': '#b9c4d1'}
R1, R1F = '#c9574c', '#168176'


def options():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ann-file', type=Path,
                        default=ROOT / 'data/pccr/R1-f/R1-f_infos_val.pkl')
    parser.add_argument('--r1-results', type=Path,
                        default=DEFAULT / 'R1/formatted/results_pccr.json')
    parser.add_argument('--r1f-results', type=Path,
                        default=DEFAULT / 'R1-f/formatted/results_pccr.json')
    parser.add_argument('--r1-metrics', type=Path,
                        default=DEFAULT / 'R1/formatted/metrics_summary.json')
    parser.add_argument('--r1f-metrics', type=Path,
                        default=DEFAULT / 'R1-f/formatted/metrics_summary.json')
    parser.add_argument('--output-dir', type=Path,
                        default=DEFAULT / 'failure_modes')
    parser.add_argument('--score-threshold', type=float, default=.35)
    parser.add_argument('--candidate-score-floor', type=float, default=.05,
                        help='Ignore lower-score outputs in the diagnostic categories.')
    parser.add_argument('--nearby-radius', type=float, default=8.0)
    return parser.parse_args()


def load_predictions(path):
    with path.open() as handle:
        content = json.load(handle)['results']
    return {token: [(np.asarray(box['translation'][:2], dtype=float),
                     float(box['detection_score']), box['detection_name'])
                    for box in boxes] for token, boxes in content.items()}


def ap(path):
    with path.open() as handle:
        values = json.load(handle)['label_aps']['car']
    return {str(key): float(value) for key, value in values.items()}


def owned_candidates(centers, predictions, radius, score_floor):
    """Assign each saved output to its nearest GT car, never two GT cars."""
    owned = [[] for _ in centers]
    if not len(centers):
        return owned
    for pred_index, (center, score, cls) in enumerate(predictions):
        if score < score_floor:
            continue
        distances = np.linalg.norm(centers-center[None, :], axis=1)
        index = int(np.argmin(distances))
        distance = float(distances[index])
        if distance < radius:
            owned[index].append((distance, score, cls, pred_index))
    return owned


def classify(candidates, threshold, score_threshold):
    """Exclusive diagnostic categories, ordered by available car evidence."""
    high_near = [c for c in candidates if c[2] == 'car' and
                 c[1] >= score_threshold and c[0] < threshold]
    high_far = [c for c in candidates if c[2] == 'car' and
                c[1] >= score_threshold and c[0] >= threshold]
    low_near = [c for c in candidates if c[2] == 'car' and
                c[1] < score_threshold and c[0] < threshold]
    low_far = [c for c in candidates if c[2] == 'car' and
               c[1] < score_threshold and c[0] >= threshold]
    if high_near:
        return 'car_correct', max(high_near, key=lambda c: c[1])
    if high_far:
        return 'car_mislocalized', min(high_far, key=lambda c: c[0])
    if low_near:
        return 'car_low_score', max(low_near, key=lambda c: c[1])
    if low_far:
        return 'car_joint', min(low_far, key=lambda c: c[0])
    # A loose 8 m candidate radius would attribute unrelated signs/people to
    # cars. Only a tightly colocated other-class output is informative here.
    others = [c for c in candidates if c[2] != 'car' and c[0] < 2.0]
    if others:
        return 'other_class_near', min(others, key=lambda c: c[0])
    return 'no_candidate', None


def diagnostic_hits(centers, predictions, threshold, score_threshold):
    """Confidence-ordered, one-to-one car matching; NOT official AP."""
    hits = set()
    ordered = sorted((p for p in predictions if p[2] == 'car' and
                      p[1] >= score_threshold), key=lambda p: -p[1])
    for center, _, _ in ordered:
        available = [i for i in range(len(centers)) if i not in hits]
        if not available:
            break
        distances = np.linalg.norm(centers[available]-center[None], axis=1)
        nearest = int(np.argmin(distances))
        if distances[nearest] < threshold:
            hits.add(available[nearest])
    return hits


def validate_inputs(frames, r1, r1f, args):
    expected = set(frames)
    for label, predictions in (('R1', r1), ('R1-f', r1f)):
        actual = set(predictions)
        if actual != expected:
            raise ValueError('{} results/GT token mismatch: {} GT-only, {} result-only'
                             .format(label, len(expected-actual), len(actual-expected)))
    if args.nearby_radius <= max(THRESHOLDS):
        raise ValueError('--nearby-radius must exceed the largest threshold (4 m)')
    if not 0 <= args.candidate_score_floor < args.score_threshold <= 1:
        raise ValueError('Require 0 <= candidate score floor < score threshold <= 1')


def analyze(frames, r1, r1f, args):
    rows = []
    for frame_number, (token, frame) in enumerate(frames.items(), 1):
        centers = frame['centers']
        r1_preds, r1f_preds = r1[token], r1f[token]
        r1_owned = owned_candidates(centers, r1_preds, args.nearby_radius,
                                    args.candidate_score_floor)
        r1f_owned = owned_candidates(centers, r1f_preds, args.nearby_radius,
                                     args.candidate_score_floor)
        origin = frame['ego_origin']
        for threshold in THRESHOLDS:
            r1_hits = diagnostic_hits(centers, r1_preds, threshold,
                                      args.score_threshold)
            r1f_hits = diagnostic_hits(centers, r1f_preds, threshold,
                                       args.score_threshold)
            for i, center in enumerate(centers):
                category, chosen = classify(r1_owned[i], threshold,
                                            args.score_threshold)
                r1f_category, _ = classify(r1f_owned[i], threshold,
                                           args.score_threshold)
                best_car = min((item for item in r1_owned[i] if item[2] == 'car'),
                               key=lambda item: item[0], default=None)
                rows.append({
                    'sample_token': token, 'gt_index': frame['gt_indices'][i],
                    'threshold_m': threshold,
                    'gt_ego_distance_m': float(np.linalg.norm(center-origin)),
                    'r1f_diagnostic_hit': int(i in r1f_hits),
                    'r1_diagnostic_hit': int(i in r1_hits),
                    'r1_category': category, 'r1f_category': r1f_category,
                    'r1_candidate_distance_m': chosen[0] if chosen else None,
                    'r1_candidate_score': chosen[1] if chosen else None,
                    'r1_candidate_class': chosen[2] if chosen else None,
                    'r1_nearest_car_distance_m': best_car[0] if best_car else None,
                    'r1_nearest_car_score': best_car[1] if best_car else None,
                    'r1_num_owned_candidates': len(r1_owned[i]),
                    'r1_num_owned_car_candidates': sum(c[2] == 'car'
                                                       for c in r1_owned[i]),
                })
        if frame_number % 200 == 0:
            print('Processed {}/{} frames'.format(frame_number, len(frames)),
                  flush=True)
    return rows


def count_rows(rows, threshold, cohort):
    group = [row for row in rows if row['threshold_m'] == threshold and
             (cohort == 'all' or row['r1f_diagnostic_hit'])]
    return len(group), Counter(row['r1_category'] for row in group)


def write_outputs(rows, args, r1_ap, r1f_ap, num_frames):
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'per_car_threshold_outcomes.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {'num_frames': num_frames,
               'num_frames_with_gt_cars': len(set(r['sample_token'] for r in rows)),
               'num_gt_cars': sum(r['threshold_m'] == .5 for r in rows),
               'score_threshold': args.score_threshold,
               'candidate_score_floor': args.candidate_score_floor,
               'nearby_radius_m': args.nearby_radius,
               'official_car_ap': {'r1': r1_ap, 'r1f': r1f_ap},
               'by_threshold': {}}
    for threshold in THRESHOLDS:
        section = {}
        for cohort in ('all', 'r1f_detected'):
            n, counts = count_rows(rows, threshold, cohort)
            section[cohort] = {'n': n, 'counts': {key: counts[key] for key in ORDER},
                               'percent': {key: 100*counts[key]/n if n else 0
                                           for key in ORDER}}
        subset = [r for r in rows if r['threshold_m'] == threshold]
        section['r1_diagnostic_hits'] = sum(r['r1_diagnostic_hit'] for r in subset)
        section['r1f_diagnostic_hits'] = sum(r['r1f_diagnostic_hit'] for r in subset)
        summary['by_threshold'][str(threshold)] = section
    with (output / 'summary.json').open('w') as handle:
        json.dump(summary, handle, indent=2)
    return summary


def plot_overview(summary, output):
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    y = np.arange(len(THRESHOLDS))
    for axis, cohort, title in zip(
            axes, ('all', 'r1f_detected'),
            ('All GT cars · what R1 outputs',
             'Cars R1-f detects · what R1 outputs for the same GT cars')):
        left = np.zeros(len(y))
        for key in ORDER:
            values = [summary['by_threshold'][str(t)][cohort]['percent'][key]
                      for t in THRESHOLDS]
            bars = axis.barh(y, values, left=left, height=.62,
                             color=COLORS[key], label=LABELS[key],
                             edgecolor='white', linewidth=1)
            for i, (bar, value) in enumerate(zip(bars, values)):
                if value >= 6:
                    count = summary['by_threshold'][str(THRESHOLDS[i])][cohort][
                        'counts'][key]
                    axis.text(left[i]+value/2, i, '{}\n{:.1f}%'.format(count, value),
                              ha='center', va='center', fontsize=9,
                              color='white' if key not in ('car_low_score',
                                                           'no_candidate') else '#182c3f',
                              fontweight='bold')
            left += values
        for i, t in enumerate(THRESHOLDS):
            n = summary['by_threshold'][str(t)][cohort]['n']
            axis.text(101, i, 'n={:,}'.format(n), va='center', fontsize=10,
                      color='#304459')
        axis.set_yticks(y)
        axis.set_yticklabels(['{:.1f} m'.format(t) for t in THRESHOLDS])
        axis.set_xlim(0, 115)
        axis.set_title(title, loc='left', fontsize=14, fontweight='bold')
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
        axis.spines['bottom'].set_visible(False)
        axis.spines['left'].set_visible(False)
        axis.tick_params(axis='y', length=0)
        axis.set_xticks(range(0, 101, 20))
        axis.grid(axis='x', alpha=.15)
        axis.set_axisbelow(True)
    axes[1].set_xlabel('Percentage of GT cars in cohort')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False,
               bbox_to_anchor=(.5, .005), fontsize=10)
    fig.suptitle('R1-f validation: is R1 losing car class, center location, or both?',
                 x=.07, y=.98, ha='left', fontsize=18, fontweight='bold')
    fig.text(.07, .925,
             'Saved detections · car pass ≥ {:.2f} · candidate floor {:.2f} · radius {:.1f} m · '
             'categories are diagnostic, not an AP decomposition'.format(
                 summary['score_threshold'], summary['candidate_score_floor'],
                 summary['nearby_radius_m']),
             fontsize=10, color='#5d6e80')
    fig.subplots_adjust(top=.88, bottom=.15, left=.1, right=.94, hspace=.4)
    for ext in ('png', 'svg'):
        fig.savefig(output / ('outcome_overview.'+ext), dpi=220)
    plt.close(fig)


def plot_ap(summary, output):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.3))
    for label, color in (('r1', R1), ('r1f', R1F)):
        ap_values = [100*summary['official_car_ap'][label][str(t)]
                     for t in THRESHOLDS]
        axes[0].plot(THRESHOLDS, ap_values, '-o', lw=2.8, markersize=7,
                     color=color, label='{} checkpoint'.format(label.upper()))
        for x, value in zip(THRESHOLDS, ap_values):
            axes[0].annotate('{:.1f}%'.format(value), (x, value),
                             xytext=(0, 9 if label == 'r1f' else -17),
                             textcoords='offset points', ha='center', color=color,
                             fontsize=10, fontweight='bold')
    axes[0].set_xticks(THRESHOLDS)
    axes[0].set_xlabel('Matching center-distance threshold (m)')
    axes[0].set_ylabel('Official car AP (%)')
    axes[0].set_title('Official AP: confidence-ranked metric')
    axes[0].legend(frameon=False)
    for label, color in (('r1', R1), ('r1f', R1F)):
        key = '{}_diagnostic_hits'.format(label)
        values = [100*summary['by_threshold'][str(t)][key]/
                  summary['num_gt_cars'] for t in THRESHOLDS]
        axes[1].plot(THRESHOLDS, values, '-o', lw=2.8, markersize=7,
                     color=color, label='{} checkpoint'.format(label.upper()))
        for x, value in zip(THRESHOLDS, values):
            axes[1].annotate('{:.1f}%'.format(value), (x, value),
                             xytext=(0, 9 if label == 'r1f' else -17),
                             textcoords='offset points', ha='center', color=color,
                             fontsize=10, fontweight='bold')
    axes[1].set_xticks(THRESHOLDS)
    axes[1].set_xlabel('Matching center-distance threshold (m)')
    axes[1].set_ylabel('Diagnostic one-to-one car coverage (%)')
    axes[1].set_title('Coverage at fixed car-score cutoff')
    for axis in axes:
        axis.grid(alpha=.18)
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
    fig.suptitle('AP loss and GT-car coverage are related, but not identical',
                 x=.045, ha='left', fontsize=16, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, .92))
    for ext in ('png', 'svg'):
        fig.savefig(output / ('ap_and_coverage.'+ext), dpi=220)
    plt.close(fig)


def plot_distance(rows, summary, output):
    bins = ((0, 10), (10, 20), (20, 30), (30, 50))
    selected = [r for r in rows if r['threshold_m'] == 1.0 and
                r['r1f_diagnostic_hit']]
    fig, axis = plt.subplots(figsize=(13, 6))
    labels = []
    left = np.zeros(len(bins))
    for low, high in bins:
        n = sum(low <= r['gt_ego_distance_m'] < high for r in selected)
        labels.append('{}–{} m\nn={:,}'.format(low, high, n))
    for key in ORDER:
        values = []
        for low, high in bins:
            group = [r for r in selected if low <= r['gt_ego_distance_m'] < high]
            values.append(100*sum(r['r1_category'] == key for r in group)/
                          len(group) if group else 0)
        bars = axis.barh(range(len(bins)), values, left=left,
                         height=.63, color=COLORS[key], label=LABELS[key],
                         edgecolor='white')
        for bar, value, start in zip(bars, values, left):
            if value >= 9:
                axis.text(start+value/2, bar.get_y()+bar.get_height()/2,
                          '{:.0f}%'.format(value), ha='center', va='center',
                          fontsize=10, fontweight='bold',
                          color='white' if key not in ('car_low_score',
                                                       'no_candidate') else '#182c3f')
        left += values
    axis.set_yticks(range(len(bins)))
    axis.set_yticklabels(labels)
    axis.set_xlim(0, 100)
    axis.set_xlabel('Percentage of R1-f-detected GT cars')
    axis.set_title('R1 outcomes at 1 m, by car distance', loc='left',
                   fontsize=16, fontweight='bold')
    axis.grid(axis='x', alpha=.18)
    axis.set_axisbelow(True)
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    axis.legend(loc='upper center', bbox_to_anchor=(.5, -.14), ncol=3,
                frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, 1))
    for ext in ('png', 'svg'):
        fig.savefig(output / ('distance_breakdown_1m.'+ext), dpi=220)
    plt.close(fig)


def report(summary, output):
    lines = [
        '# Full-validation R1-f car failure modes', '',
        'Both checkpoints ran on the same {} R1-f frames. There are {} GT cars. '
        'Diagnostic car-score cutoff is {:.2f}; candidate floor is {:.2f}; '
        'candidate radius is {:.1f} m (other-class evidence requires <2 m).'
        .format(summary['num_frames'], summary['num_gt_cars'],
                summary['score_threshold'], summary['candidate_score_floor'],
                summary['nearby_radius_m']),
        '', 'Each output box is assigned to its nearest GT car within the radius. '
        'Categories are exclusive and use this priority: passing-score car '
        'inside the center threshold, passing-score car outside it, low-score '
        'car inside it, low-score car outside it, other-class box within 2 m, '
        'none. '
        'The reference cohort is GT cars with a confidence-ordered, one-to-one '
        'R1-f car match at that same threshold. This is not official AP.',
        '', '| Threshold | R1-f-detected GT cars | R1 correct | R1 car-score passes, '
        'center fails | R1 car center passes, score low | R1 both fail | '
        'Nearby other-class only | No qualifying nearby output | R1 AP | R1-f AP |',
        '| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for t in THRESHOLDS:
        section = summary['by_threshold'][str(t)]['r1f_detected']
        numbers = [section['counts'][key] for key in ORDER]
        lines.append('| {:.1f} m | {} | {} | {} | {} | {} | {} | {} | '
                     '{:.2f}% | {:.2f}% |'.format(
                         t, section['n'], *numbers,
                         100*summary['official_car_ap']['r1'][str(t)],
                         100*summary['official_car_ap']['r1f'][str(t)]))
    lines += ['', '## Important limits', '',
              'The saved JSON contains post-processed class-labelled boxes, not '
              'every query\'s raw class logits. A nearby other-class box is '
              '**not proof** that the car query was misclassified. A distant '
              'car box is evidence of a car-labelled localization candidate, '
              'but in a crowded frame may belong to another object despite '
              'nearest-GT ownership. The chosen score cutoff and candidate '
              'radius are diagnostic choices; vary both for sensitivity. '
              'Official AP uses confidence ranking and false positives, and '
              'cannot be decomposed exactly by these per-GT categories.', '']
    (output / 'README_RESULTS.md').write_text('\n'.join(lines))


def main():
    args = options()
    for path in (args.ann_file, args.r1_results, args.r1f_results,
                 args.r1_metrics, args.r1f_metrics):
        if not path.is_file():
            raise FileNotFoundError(str(path))
    frames = gt_frames(args.ann_file)
    r1 = load_predictions(args.r1_results)
    r1f = load_predictions(args.r1f_results)
    validate_inputs(frames, r1, r1f, args)
    rows = analyze(frames, r1, r1f, args)
    summary = write_outputs(rows, args, ap(args.r1_metrics), ap(args.r1f_metrics),
                            len(frames))
    plot_overview(summary, args.output_dir)
    plot_ap(summary, args.output_dir)
    plot_distance(rows, summary, args.output_dir)
    report(summary, args.output_dir)
    print('Saved full-validation car failure analysis:', args.output_dir)


if __name__ == '__main__':
    main()
