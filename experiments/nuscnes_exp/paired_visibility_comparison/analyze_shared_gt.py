#!/usr/bin/env python3
"""Analyze visibility, localization and confidence for GT matched by both models."""

import argparse
import csv
import math
import shutil
import statistics
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from xml.sax.saxutils import escape


HERE = Path(__file__).resolve().parent
VISIBILITY = ('0-40%', '40-60%', '60-80%', '80-100%')
BANDS = ('0-0.5', '0.5-1', '1-2', '2-4')
COLORS = {'PETR': '#d44b43', 'StreamPETR': '#2563a5'}


def band(error):
    for index, upper in enumerate((.5, 1, 2, 4)):
        if error < upper:
            return index
    raise ValueError('Shared GT has a match outside 4 m')


def quantile(values, fraction):
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def write_csv(path, rows):
    if not rows:
        raise ValueError('No data to write: {}'.format(path))
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_shared(path):
    rows = []
    with path.open(newline='') as handle:
        for source in csv.DictReader(handle):
            if not (source['petr_error_at_4m_m'] and
                    source['stream_error_at_4m_m']):
                continue
            visibility = source['visibility_band']
            if visibility not in VISIBILITY:
                raise ValueError('Unexpected GT visibility {}'.format(visibility))
            row = {
                'class': source['class'],
                'visibility_band': visibility,
                'petr_error_m': float(source['petr_error_at_4m_m']),
                'stream_error_m': float(source['stream_error_at_4m_m']),
                'petr_score': float(source['petr_score_at_4m']),
                'stream_score': float(source['stream_score_at_4m']),
            }
            if not all(math.isfinite(value) for key, value in row.items()
                       if key not in ('class', 'visibility_band')):
                raise ValueError('Nonfinite error or score in {}'.format(path))
            if not (0 <= row['petr_score'] <= 1 and
                    0 <= row['stream_score'] <= 1):
                raise ValueError('Score outside [0,1]')
            row['petr_band'] = band(row['petr_error_m'])
            row['stream_band'] = band(row['stream_error_m'])
            rows.append(row)
    if not rows:
        raise ValueError('No GT matched by both models')
    return rows


def add_within_class_ranks(rows):
    """Percentile within the same class and shared-4m-GT cohort, per model."""
    by_class = defaultdict(list)
    for index, row in enumerate(rows):
        by_class[row['class']].append(index)
    for indices in by_class.values():
        count = len(indices)
        for model in ('petr', 'stream'):
            ranked = sorted(indices, key=lambda i: rows[i][model + '_score'])
            start = 0
            while start < count:
                end = start + 1
                score = rows[ranked[start]][model + '_score']
                while end < count and rows[ranked[end]][model + '_score'] == score:
                    end += 1
                percentile = (100 * (start + end - 1) / 2 / (count - 1)
                              if count > 1 else 50)
                for position in range(start, end):
                    rows[ranked[position]][model + '_rank_percentile'] = percentile
                start = end


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups['all', row['visibility_band']].append(row)
        groups[row['class'], row['visibility_band']].append(row)
    output = []
    for (name, visibility), group in sorted(groups.items()):
        n = len(group)
        delta = [r['stream_error_m'] - r['petr_error_m'] for r in group]
        rank_delta = [r['stream_rank_percentile'] - r['petr_rank_percentile']
                      for r in group]
        result = {'class': name, 'visibility_band': visibility, 'gt_count': n}
        for model in ('petr', 'stream'):
            for measure, suffix in (('error_m', 'error_m'),
                                    ('score', 'raw_score'),
                                    ('rank_percentile', 'rank_percentile')):
                values = [r[model + '_' + measure] for r in group]
                result[model + '_mean_' + suffix] = statistics.mean(values)
                result[model + '_median_' + suffix] = statistics.median(values)
            for threshold in (.5, 1, 2):
                result[model + '_lt_' + str(threshold).replace('.', '_') + 'm_percent'] = (
                    100 * sum(r[model + '_error_m'] < threshold
                              for r in group) / n)
        result['mean_stream_minus_petr_error_m'] = statistics.mean(delta)
        result['median_stream_minus_petr_error_m'] = statistics.median(delta)
        result['stream_lower_error_count'] = sum(value < 0 for value in delta)
        result['stream_lower_error_percent'] = 100 * result['stream_lower_error_count'] / n
        result['stream_better_band_count'] = sum(
            r['stream_band'] < r['petr_band'] for r in group)
        result['stream_same_band_count'] = sum(
            r['stream_band'] == r['petr_band'] for r in group)
        result['stream_worse_band_count'] = sum(
            r['stream_band'] > r['petr_band'] for r in group)
        result['median_stream_minus_petr_rank_percentile'] = statistics.median(rank_delta)
        result['stream_higher_rank_percentile_percent'] = (
            100 * sum(value > 0 for value in rank_delta) / n)
        output.append(result)
    return output


def transitions(rows):
    counts = Counter((r['visibility_band'], r['petr_band'], r['stream_band'])
                     for r in rows)
    result = []
    for visibility in VISIBILITY:
        for p in range(4):
            total = sum(counts[visibility, p, s] for s in range(4))
            for s in range(4):
                count = counts[visibility, p, s]
                result.append({'visibility_band': visibility,
                               'petr_error_band': BANDS[p],
                               'stream_error_band': BANDS[s],
                               'gt_count': count,
                               'petr_band_gt_count': total,
                               'row_percent': 100 * count / total if total else 0})
    return result


def svg_start(width, height):
    return ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {} {}" '
            'width="{}" height="{}">'.format(width, height, width, height),
            '<rect width="100%" height="100%" fill="#ffffff"/>',
            '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#263444}'
            '.title{font-size:27px;font-weight:bold}'
            '.note{font-size:14px;fill:#596879}'
            '.head{font-size:18px;font-weight:bold}'
            '.label{font-size:13px}.count{font-size:15px;font-weight:bold}'
            '.pct{font-size:12px}</style>']


def txt(parts, x, y, value, css='label', anchor=None, color=None):
    extra = ' text-anchor="{}"'.format(anchor) if anchor else ''
    if color:
        extra += ' style="fill:{}"'.format(color)
    parts.append('<text x="{:.1f}" y="{:.1f}" class="{}"{}>{}</text>'.format(
        x, y, css, extra, escape(str(value))))


def rect(parts, x, y, w, h, fill, stroke=None, width=1):
    extra = (' stroke="{}" stroke-width="{}"'.format(stroke, width)
             if stroke else '')
    parts.append('<rect x="{:.1f}" y="{:.1f}" width="{:.1f}" '
                 'height="{:.1f}" fill="{}"{}/>'.format(x, y, w, h, fill, extra))


def line(parts, x1, y1, x2, y2, color, width=1):
    parts.append('<line x1="{:.1f}" y1="{:.1f}" x2="{:.1f}" y2="{:.1f}" '
                 'stroke="{}" stroke-width="{}"/>'.format(
                     x1, y1, x2, y2, color, width))


def heat_color(percent):
    t = min(1, percent / 75)
    start, end = (238, 246, 251), (28, 99, 151)
    rgb = [round(a + (b - a) * t) for a, b in zip(start, end)]
    return '#{:02x}{:02x}{:02x}'.format(*rgb), t > .58


def draw_transitions(path, data, summary):
    width, height = 1230, 845
    parts = svg_start(width, height)
    txt(parts, 28, 44, 'Same GT: PETR to StreamPETR center-error bands', 'title')
    txt(parts, 28, 70, 'Each panel uses only GT matched by both models within 4 m. '
        'Cells give count and percentage of their PETR row.', 'note')
    lookup = {(r['visibility_band'], r['petr_error_band'],
               r['stream_error_band']): r for r in data}
    totals = {(r['visibility_band']): r for r in summary if r['class'] == 'all'}
    for panel, visibility in enumerate(VISIBILITY):
        col, row = panel % 2, panel // 2
        px, py = 28 + col * 605, 108 + row * 345
        rect(parts, px, py, 585, 322, '#ffffff', '#d7e0e8')
        total = totals[visibility]
        txt(parts, px + 18, py + 29,
            '{} visible | n={:,} | Stream lower error {:.1f}%'.format(
                visibility, total['gt_count'], total['stream_lower_error_percent']),
            'head')
        gx, gy, cw, ch = px + 118, py + 88, 110, 48
        txt(parts, gx + 2 * cw, py + 53, 'StreamPETR error', 'label', 'middle')
        for index, label in enumerate(BANDS):
            txt(parts, gx + (index + .5) * cw, gy - 11,
                label + 'm', 'label', 'middle')
            txt(parts, gx - 9, gy + index * ch + 29,
                label + 'm', 'label', 'end')
        for p in range(4):
            for s in range(4):
                cell = lookup[visibility, BANDS[p], BANDS[s]]
                x, y = gx + s * cw, gy + p * ch
                fill, light = heat_color(cell['row_percent'])
                rect(parts, x + 1.5, y + 1.5, cw - 3, ch - 3, fill,
                     '#8699ab' if p == s else None, 1)
                ink = '#ffffff' if light else '#263444'
                txt(parts, x + cw / 2, y + 20,
                    '{:,}'.format(cell['gt_count']), 'count', 'middle', ink)
                txt(parts, x + cw / 2, y + 38,
                    '{:.1f}%'.format(cell['row_percent']), 'pct', 'middle', ink)
        txt(parts, px + 18, py + 309,
            'Left of diagonal: better band   |   right: worse band', 'note')
    txt(parts, 28, height - 30,
        'Localization bands use the models\' separately assigned <4 m predictions. '
        'A lower band means a smaller BEV center error.', 'note')
    parts.append('</svg>')
    path.write_text('\n'.join(parts) + '\n')


def boxplot(parts, x0, y, width, values, color):
    q05, q25, med, q75, q95 = [quantile(values, q)
                               for q in (.05, .25, .5, .75, .95)]
    scale = lambda value: x0 + value * width
    line(parts, scale(q05), y, scale(q95), y, color, 2)
    line(parts, scale(q05), y - 8, scale(q05), y + 8, color, 2)
    line(parts, scale(q95), y - 8, scale(q95), y + 8, color, 2)
    rect(parts, scale(q25), y - 10, max(1, scale(q75) - scale(q25)),
         20, color)
    line(parts, scale(med), y - 12, scale(med), y + 12, '#ffffff', 2)


def draw_scores(path, rows, summary):
    width, height = 1230, 710
    parts = svg_start(width, height)
    txt(parts, 28, 45, 'Matched GT confidence: raw scores and relative ranks',
        'title')
    txt(parts, 28, 72, 'Box: 25th–75th percentile; white line: median; whiskers: '
        '5th–95th percentile. Same {:,} shared GT.'.format(len(rows)), 'note')
    txt(parts, 28, 99, 'Raw scores can have different calibration across models. '
        'Relative rank is within each class and model among shared 4 m TPs.', 'note')
    line(parts, 30, 124, 68, 124, COLORS['PETR'], 3)
    txt(parts, 76, 129, 'PETR', 'label')
    line(parts, 142, 124, 180, 124, COLORS['StreamPETR'], 3)
    txt(parts, 188, 129, 'StreamPETR', 'label')
    by_vis = defaultdict(list)
    for row in rows:
        by_vis[row['visibility_band']].append(row)
    x0, x1, axis_w = 178, 698, 420
    txt(parts, x0 + axis_w / 2, 145, 'Raw detection score (0–1)', 'head', 'middle')
    txt(parts, x1 + axis_w / 2, 145,
        'Within-class score-rank percentile', 'head', 'middle')
    for start in (x0, x1):
        for tick in (0, .25, .5, .75, 1):
            x = start + tick * axis_w
            line(parts, x, 166, x, 610, '#e0e8ef')
            txt(parts, x, 634, '{:.0f}'.format(tick * (100 if start == x1 else 1))
                if start == x1 else '{:.2g}'.format(tick),
                'label', 'middle')
    for index, visibility in enumerate(VISIBILITY):
        group = by_vis[visibility]
        y0 = 200 + index * 104
        txt(parts, 28, y0 + 23, '{} visible'.format(visibility), 'head')
        txt(parts, 28, y0 + 45, 'n={:,}'.format(len(group)), 'label')
        for model, offset in (('petr', 0), ('stream', 30)):
            shade = COLORS['PETR' if model == 'petr' else 'StreamPETR']
            y = y0 + offset
            boxplot(parts, x0, y, axis_w,
                    [r[model + '_score'] for r in group], shade)
            boxplot(parts, x1, y, axis_w,
                    [r[model + '_rank_percentile'] / 100 for r in group], shade)
        line(parts, 28, y0 + 66, 1147, y0 + 66, '#e0e8ef')
    txt(parts, 28, height - 42,
        'These score plots include matched true positives only. They cannot '
        'measure false-positive ranking or explain AP on their own.', 'note')
    parts.append('</svg>')
    path.write_text('\n'.join(parts) + '\n')


def png(svg):
    converter = shutil.which('rsvg-convert')
    if converter:
        subprocess.run([converter, '-o', str(svg.with_suffix('.png')),
                        str(svg)], check=True)
        return
    try:
        import cairosvg
    except ImportError:
        return
    cairosvg.svg2png(url=str(svg), write_to=str(svg.with_suffix('.png')))


def report(path, summary):
    lookup = {r['visibility_band']: r for r in summary if r['class'] == 'all'}
    total = sum(lookup[visibility]['gt_count'] for visibility in VISIBILITY)
    lines = ['# PETR and StreamPETR on the same <4 m matched GT', '',
             'This analysis keeps **only GT annotations matched by both models '
             'within 4 m**. The denominator is {:,} GT annotations across '
             'validation frames. Each model\'s score-ordered 4 m match is used; '
             'predictions are not directly paired by query identity.'.format(total), '',
             '![Error-band transitions by GT visibility](shared_gt_error_transitions.svg)', '',
             '![Matched-detection confidence distributions](shared_gt_scores.svg)', '',
             '| GT visibility | Shared GT | Mean center error PETR → StreamPETR '
             '| Stream lower error | Median raw score PETR → StreamPETR '
             '| Median within-class rank PETR → StreamPETR |',
             '|---|---:|---:|---:|---:|---:|']
    for visibility in VISIBILITY:
        row = lookup[visibility]
        lines.append('| {} | {:,} | {:.3f} → {:.3f} m | {:.1f}% | '
                     '{:.3f} → {:.3f} | {:.1f} → {:.1f} percentile |'.format(
                         visibility, row['gt_count'], row['petr_mean_error_m'],
                         row['stream_mean_error_m'], row['stream_lower_error_percent'],
                         row['petr_median_raw_score'],
                         row['stream_median_raw_score'],
                         row['petr_median_rank_percentile'],
                         row['stream_median_rank_percentile']))
    lines += ['', 'Raw score magnitudes are **not directly calibrated across '
              'models**. The within-class percentiles compare each detection\'s '
              'relative priority among **shared matched TPs only**; neither '
              'plot accounts for false positives. The localization comparison '
              'is conditional on both models matching a GT within 4 m and '
              'does not describe GT missed by either model.', '',
              'See `shared_gt_visibility_summary.csv` for class × visibility '
              'results and `shared_gt_error_transitions.csv` for the exact '
              '4×4 matrices.', '']
    path.write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paired-gt', type=Path,
                        default=HERE / 'outputs/paired_gt.csv')
    parser.add_argument('--output-dir', type=Path,
                        default=HERE / 'outputs/shared_gt_analysis')
    args = parser.parse_args()
    if not args.paired_gt.is_file():
        parser.error('Missing paired GT file: {}'.format(args.paired_gt))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_shared(args.paired_gt)
    add_within_class_ranks(rows)
    summary = summarize(rows)
    matrix = transitions(rows)
    write_csv(args.output_dir / 'shared_gt_visibility_summary.csv', summary)
    write_csv(args.output_dir / 'shared_gt_error_transitions.csv', matrix)
    figures = ((args.output_dir / 'shared_gt_error_transitions.svg',
                lambda path: draw_transitions(path, matrix, summary)),
               (args.output_dir / 'shared_gt_scores.svg',
                lambda path: draw_scores(path, rows, summary)))
    for path, draw in figures:
        draw(path)
        png(path)
    report(args.output_dir / 'RESULTS.md', summary)
    print('Analyzed {:,} shared <4 m GT; wrote {}'.format(len(rows),
          args.output_dir), flush=True)


if __name__ == '__main__':
    main()
