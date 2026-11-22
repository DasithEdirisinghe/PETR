#!/usr/bin/env python3
"""Draw train/validation GT visibility composition from the summary CSV."""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent /
                       'fixed_4m_tp_retention'))
from plot_tp_visibility_bands import (BANDS, format_count, svg_rect,
                                      svg_text, write_svg_and_png)

DISTANCE_LABELS = ('0–10', '10–20', '20–30', '30–40', '40–50', '50+')
SPLITS = ('train', 'val')
CELL_W, CELL_H = 340, 180
LEFT, TOP = 130, 150
PANEL_W, PANEL_H = LEFT + 6 * CELL_W + 20, TOP + 2 * CELL_H + 55
UNKNOWN_BAND = ('?', 'unlabeled', 'visibility_unknown_count', '#8997A5')


def panel_svg(name, lookup, show_unknown):
    bands = BANDS + (UNKNOWN_BAND,) if show_unknown else BANDS
    parts = [svg_rect(0, 0, PANEL_W, PANEL_H, '#FFFFFF'),
             svg_text(28, 43, name.replace('_', ' ').title(), 27,
                      weight='bold'),
             svg_text(28, 71, 'nuScenes keyframe GT · bar width = '
                      'visibility share within each distance bin',
                      16, '#536374')]
    legend_step = 395 if not show_unknown else 340
    for index, (token, interval, _, color) in enumerate(bands):
        x = 28 + index * legend_step
        parts.append(svg_rect(x, 89, 20, 20, color, '#758392', 3))
        parts.append(svg_text(x + 28, 105,
                              'V{} {}{}'.format(token, interval,
                                                ' · less visible' if index == 0
                                                else ' · more visible' if index == 3
                                                else ''), 15))
    for column, distance_label in enumerate(DISTANCE_LABELS):
        x = LEFT + column * CELL_W + CELL_W / 2
        parts.append(svg_text(x, TOP - 12, distance_label + ' m', 17,
                              weight='bold', anchor='middle'))
    for row_index, split in enumerate(SPLITS):
        y = TOP + row_index * CELL_H
        parts.append(svg_text(LEFT - 13, y + CELL_H / 2,
                              'Training' if split == 'train' else 'Validation',
                              17, weight='bold', anchor='end'))
        for column, distance_label in enumerate(DISTANCE_LABELS):
            x = LEFT + column * CELL_W
            cell = lookup.get((split, name, distance_label))
            if cell is None:
                raise ValueError('Missing GT cell: {} / {} / {}'.format(
                    split, name, distance_label))
            total = int(cell['gt_count'])
            counts = [int(cell[field]) for _, _, field, _ in bands]
            if sum(counts) != total:
                raise ValueError('GT visibility counts do not sum to total: '
                                 '{} / {} / {}'.format(split, name,
                                                        distance_label))
            parts.append(svg_rect(x + 2, y + 2, CELL_W - 4, CELL_H - 4,
                                  '#FFFFFF' if total else '#F2F4F6',
                                  '#C8D0D8', 5))
            if not total:
                parts.append(svg_text(x + CELL_W / 2, y + CELL_H / 2,
                                      'No GT', 17, '#667586', anchor='middle'))
                continue
            parts.append(svg_text(x + 16, y + 29,
                                  'N = {}'.format(format_count(total)),
                                  18, weight='bold'))
            bar_x, bar_y, bar_w, bar_h = x + 16, y + 43, CELL_W - 32, 25
            left = bar_x
            for count, (_, _, _, color) in zip(counts, bands):
                width = bar_w * count / total
                if width:
                    parts.append(svg_rect(left, bar_y, width, bar_h, color))
                left += width
            parts.append(svg_rect(bar_x, bar_y, bar_w, bar_h, 'none',
                                  '#667586', 2))
            for index, (count, (token, _, _, color)) in enumerate(zip(counts, bands)):
                label_x = x + 16 + (index % 2) * 161
                label_y = y + 103 + (index // 2) * 32
                parts.append(svg_rect(label_x, label_y - 11, 12, 12,
                                      color, '#667586', 2))
                parts.append(svg_text(label_x + 20, label_y,
                                      'V{} {} ({:.1f}%)'.format(
                                          token, format_count(count),
                                          100 * count / total), 13))
    parts.append(svg_text(LEFT, PANEL_H - 24,
                          'N counts raw GT annotations; each category count '
                          'and percentage uses that cell’s N.',
                          15, '#536374'))
    return ''.join(parts)


def render_gt_visibility(rows, output_dir, names=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if names is None:
        names = list(dict.fromkeys(row['class'] for row in rows))
    lookup = {(row['split'], row['class'], row['distance_bin']): row
              for row in rows}
    show_unknown = any(int(row.get('visibility_unknown_count', 0))
                       for row in rows)
    for name in names:
        write_svg_and_png(output_dir / ('gt_visibility_bands_' +
                                        name.replace(' ', '_')),
                          PANEL_W, PANEL_H,
                          panel_svg(name, lookup, show_unknown))
    columns, gap = 2, 20
    rows_of_panels = (len(names) + columns - 1) // columns
    width = columns * PANEL_W + (columns + 1) * gap
    height = rows_of_panels * PANEL_H + (rows_of_panels + 1) * gap
    overview = [svg_rect(0, 0, width, height, '#E8EDF2')]
    for index, name in enumerate(names):
        x = gap + (index % columns) * (PANEL_W + gap)
        y = gap + (index // columns) * (PANEL_H + gap)
        overview.append('<g transform="translate({}, {})">{}</g>'.format(
            x, y, panel_svg(name, lookup, show_unknown)))
    write_svg_and_png(output_dir / 'gt_visibility_bands_overview',
                      width, height, ''.join(overview))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary-csv', type=Path, default=Path(__file__).parent /
                        'GT_VISIBILITY_BY_DISTANCE.csv')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).parent /
                        'visibility_plots')
    args = parser.parse_args()
    with args.summary_csv.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    render_gt_visibility(rows, args.output_dir)
    print('Wrote GT visibility plots to {}'.format(args.output_dir))


if __name__ == '__main__':
    main()
