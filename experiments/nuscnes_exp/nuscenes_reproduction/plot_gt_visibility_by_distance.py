#!/usr/bin/env python3
"""Plot nuScenes GT visibility by class and range from saved CSV counts."""

import argparse
import csv
import html
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path


CLASSES = (
    'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
)
DISTANCES = ('0-10m', '10-20m', '20-30m', '30-40m', '40-50m')
RAW_DISTANCES = DISTANCES + ('50m+',)
BANDS = (
    ('1', '0–40%', '#8B1E3F'),
    ('2', '40–60%', '#E76F51'),
    ('3', '60–80%', '#E9C46A'),
    ('4', '80–100%', '#E8F1F2'),
)
HEIGHT = 1875
LEFT, TOP = 220, 205
CELL_W, CELL_H = 380, 160


def text(x, y, value, size=15, color='#263445', weight='normal', anchor=None):
    anchor_attr = ' text-anchor="{}"'.format(anchor) if anchor else ''
    return ('<text x="{:.1f}" y="{:.1f}" font-size="{}" fill="{}" '
            'font-weight="{}"{}>{}</text>').format(
                x, y, size, color, weight, anchor_attr, html.escape(str(value)))


def rect(x, y, width, height, fill, stroke=None, radius=0):
    border = ' stroke="{}"'.format(stroke) if stroke else ''
    return ('<rect x="{:.2f}" y="{:.2f}" width="{:.2f}" height="{:.2f}" '
            'rx="{}" fill="{}"{}/>').format(
                x, y, width, height, radius, fill, border)


def count_gt(path):
    counts = defaultdict(Counter)
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        required = {'class', 'distance_bin', 'visibility_token'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError('Missing required GT columns in {}'.format(path))
        for row in reader:
            name, distance, visibility = (row['class'], row['distance_bin'],
                                          row['visibility_token'])
            if name not in CLASSES or distance not in DISTANCES or visibility not in ('1', '2', '3', '4'):
                raise ValueError('Unexpected class, distance, or visibility: {}'.format(row))
            counts[name, distance][visibility] += 1
    return counts


def count_raw_gt(path, split):
    counts = defaultdict(Counter)
    columns = ('visibility_0_40_count', 'visibility_40_60_count',
               'visibility_60_80_count', 'visibility_80_100_count')
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        required = {'split', 'class', 'distance_bin', 'gt_count', *columns}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError('Missing raw GT summary columns in {}'.format(path))
        for row in reader:
            if row['split'] != split or row['class'] not in CLASSES:
                continue
            distance = row['distance_bin'].replace('–', '-') + (
                '' if row['distance_bin'] == '50+' else 'm')
            if distance == '50+':
                distance = '50m+'
            if distance not in RAW_DISTANCES:
                raise ValueError('Unexpected distance bin: {}'.format(row['distance_bin']))
            cell = counts[row['class'], distance]
            for index, column in enumerate(columns, 1):
                cell[str(index)] += int(row[column])
            if sum(int(row[column]) for column in columns) != int(row['gt_count']):
                raise ValueError('Visibility counts do not sum to raw GT count: {}'.format(row))
    return counts


def render(counts, distances, population):
    width = LEFT + len(distances) * CELL_W + 30
    raw = population.startswith('raw-')
    split = 'training' if population == 'raw-train' else 'validation'
    cohort = ('Raw annotated GTs' if raw else 'Evaluator-eligible GTs used for PETR AP')
    note = ('All annotated distance bins are shown, including 50 m+.' if raw else
            'Blank gray cells are outside class evaluation range.')
    parts = [rect(0, 0, width, HEIGHT, '#FFFFFF'),
             text(32, 45, 'nuScenes {} GT visibility by class and distance'.format(split),
                  29, weight='bold'),
             text(32, 77, '{}; every cell is one class × ego-distance bin.'.format(cohort),
                  18, '#536374'),
             text(32, 105, 'Bar = visibility share within the cell. Labels = GT count and share. {}'.format(note),
                  16, '#536374')]
    for index, (token, label, color) in enumerate(BANDS):
        x = 32 + index * 325
        parts.extend((rect(x, 127, 22, 22, color, '#758392', 3),
                      text(x + 31, 144, 'V{}: {}'.format(token, label), 17)))
    total = sum(sum(cell.values()) for cell in counts.values())
    parts.append(text(width - 37, 144, 'N = {:,} {} GTs'.format(total, 'raw' if raw else 'eligible'),
                      18, weight='bold', anchor='end'))
    for col, distance in enumerate(distances):
        x = LEFT + col * CELL_W + CELL_W / 2
        parts.append(text(x, TOP - 16, distance.replace('-', '–'),
                          18, weight='bold', anchor='middle'))
    for row, name in enumerate(CLASSES):
        y = TOP + row * CELL_H
        class_total = sum(sum(counts[name, d].values()) for d in distances)
        parts.extend((text(LEFT - 17, y + 59, name.replace('_', ' ').title(),
                           17, weight='bold', anchor='end'),
                      text(LEFT - 17, y + 82, 'N = {:,}'.format(class_total),
                           15, '#536374', anchor='end')))
        for col, distance in enumerate(distances):
            x = LEFT + col * CELL_W
            cell = counts[name, distance]
            n = sum(cell.values())
            parts.append(rect(x + 3, y + 3, CELL_W - 6, CELL_H - 6,
                              '#FFFFFF' if n else '#F0F2F4', '#CCD3D9', 5))
            if not n:
                parts.append(text(x + CELL_W / 2, y + 78,
                                  'Outside evaluation range' if not raw else 'No GTs', 16,
                                  '#728091', anchor='middle'))
                continue
            parts.append(text(x + 17, y + 30, 'N = {:,}'.format(n),
                              18, weight='bold'))
            bar_x, bar_y, bar_w, bar_h = x + 17, y + 43, CELL_W - 34, 23
            cursor = bar_x
            for token, _, color in BANDS:
                segment_w = bar_w * cell[token] / n
                if segment_w:
                    parts.append(rect(cursor, bar_y, segment_w, bar_h, color))
                cursor += segment_w
            parts.append(rect(bar_x, bar_y, bar_w, bar_h, 'none', '#7B8793', 2))
            for index, (token, _, color) in enumerate(BANDS):
                lx = x + 17 + (index % 2) * 174
                ly = y + 96 + (index // 2) * 33
                parts.extend((rect(lx, ly - 12, 13, 13, color, '#87929D', 2),
                              text(lx + 20, ly,
                                   'V{} {:,} ({:.1f}%)'.format(
                                       token, cell[token], 100 * cell[token] / n),
                                   14)))
    parts.append(text(LEFT, HEIGHT - 28,
                      'GT annotations are counted per keyframe. Visibility token summarizes the object across the six cameras.',
                      15, '#536374'))
    return ('<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" '
            'viewBox="0 0 {} {}" font-family="Arial, Helvetica, sans-serif">{}</svg>').format(
                width, HEIGHT, width, HEIGHT, ''.join(parts))


def write_csv(counts, distances, path):
    fields = ['class', 'distance_bin', 'gt_count']
    for token, _, _ in BANDS:
        fields.extend(('visibility_{}_count'.format(token),
                       'visibility_{}_percent'.format(token)))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name in CLASSES:
            for distance in distances:
                cell = counts[name, distance]
                n = sum(cell.values())
                row = {'class': name, 'distance_bin': distance, 'gt_count': n}
                for token, _, _ in BANDS:
                    row['visibility_{}_count'.format(token)] = cell[token]
                    row['visibility_{}_percent'.format(token)] = (
                        round(100 * cell[token] / n, 2) if n else '')
                writer.writerow(row)


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gt-transitions', type=Path,
                        default=here.parent / 'petr_ap_gap_audit/outputs/gt_transitions.csv')
    parser.add_argument('--raw-distribution', type=Path,
                        default=here.parent / 'gt_distance_distribution/GT_VISIBILITY_BY_DISTANCE.csv')
    parser.add_argument('--population', choices=('eligible-val', 'raw-train', 'raw-val'),
                        default='eligible-val')
    parser.add_argument('--output-dir', type=Path, default=here / 'outputs/plots')
    args = parser.parse_args()
    if args.population == 'eligible-val':
        counts, distances = count_gt(args.gt_transitions), DISTANCES
    else:
        split = 'train' if args.population == 'raw-train' else 'val'
        counts, distances = count_raw_gt(args.raw_distribution, split), RAW_DISTANCES
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / ('{}_gt_visibility_by_class_distance'.format(
        args.population.replace('-', '_')))
    if args.population == 'eligible-val':
        stem = args.output_dir / 'eligible_gt_visibility_by_class_distance'
    stem.with_suffix('.svg').write_text(render(counts, distances, args.population))
    write_csv(counts, distances, stem.with_suffix('.csv'))
    converter = shutil.which('rsvg-convert')
    if converter:
        subprocess.run([converter, '-o', str(stem.with_suffix('.png')),
                        str(stem.with_suffix('.svg'))], check=True)
    print('Wrote {} (N={:,} GTs)'.format(stem, sum(sum(c.values()) for c in counts.values())))


if __name__ == '__main__':
    main()
