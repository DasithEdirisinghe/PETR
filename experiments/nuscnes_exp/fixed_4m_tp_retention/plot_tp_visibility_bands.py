#!/usr/bin/env python3
"""Render four-band GT visibility composition from the existing TP summary CSV."""
import argparse
import csv
import html
import shutil
import subprocess
from pathlib import Path

DISTANCE_LABELS = ('0-10', '10-20', '20-30', '30-40', '40-50')
ERROR_LABELS = ('0-0.5', '0.5-1', '1-2', '2-4')
BANDS = (
    ('1', '0–40%', 'visibility_0_40_count', '#8B1E3F'),
    ('2', '40–60%', 'visibility_40_60_count', '#E76F51'),
    ('3', '60–80%', 'visibility_60_80_count', '#E9C46A'),
    ('4', '80–100%', 'visibility_80_100_count', '#E8F1F2'),
)
PANEL_W, PANEL_H = 1500, 1040
LEFT, TOP = 122, 156
CELL_W, CELL_H = 330, 165


def svg_text(x, y, value, size=16, color='#263445', weight='normal', anchor=None):
    anchor_attr = ' text-anchor="{}"'.format(anchor) if anchor else ''
    return ('<text x="{}" y="{}" fill="{}" font-size="{}" '
            'font-weight="{}"{}>{}</text>').format(
                x, y, color, size, weight, anchor_attr, html.escape(str(value)))


def svg_rect(x, y, width, height, fill, stroke=None, radius=0):
    stroke_attr = ' stroke="{}"'.format(stroke) if stroke else ''
    return ('<rect x="{}" y="{}" width="{}" height="{}" rx="{}" '
            'fill="{}"{}/>').format(x, y, width, height, radius, fill,
                                     stroke_attr)


def format_count(count):
    return '{:,}'.format(count)


def panel_svg(name, lookup):
    parts = [svg_rect(0, 0, PANEL_W, PANEL_H, '#FFFFFF'),
             svg_text(36, 45, name.replace('_', ' ').title(), 28,
                      weight='bold'),
             svg_text(36, 75, 'GT visibility among PETR 4 m TPs · '
                      'bar width = within-cell share', 16, '#536374')]
    for index, (token, interval, _, color) in enumerate(BANDS):
        x = 36 + index * 345
        parts.append(svg_rect(x, 91, 21, 20, color, '#758392', 3))
        parts.append(svg_text(x + 29, 107,
                              'V{} {} {}'.format(token, interval,
                                                  'less visible' if index == 0
                                                  else 'more visible' if index == 3
                                                  else ''), 16))
    for column, error_label in enumerate(ERROR_LABELS):
        x = LEFT + column * CELL_W + CELL_W / 2
        parts.append(svg_text(x, TOP - 15, error_label + ' m error', 17,
                              weight='bold', anchor='middle'))
    for row_index, distance_label in enumerate(DISTANCE_LABELS):
        y = TOP + row_index * CELL_H
        parts.append(svg_text(LEFT - 14, y + CELL_H / 2 + 5,
                              distance_label + ' m', 17,
                              weight='bold', anchor='end'))
        for column, error_label in enumerate(ERROR_LABELS):
            x = LEFT + column * CELL_W
            cell = lookup.get((name, distance_label, error_label))
            if cell is None:
                raise ValueError('Missing summary cell: {} / {} / {}'.format(
                    name, distance_label, error_label))
            counts = [int(cell[field]) for _, _, field, _ in BANDS]
            total = int(cell['tp_count'])
            if sum(counts) != total:
                raise ValueError('Visibility counts do not sum to TP count: '
                                 '{} / {} / {}'.format(name, distance_label,
                                                        error_label))
            parts.append(svg_rect(x + 2, y + 2, CELL_W - 4, CELL_H - 4,
                                  '#FFFFFF' if total else '#F2F4F6',
                                  '#C8D0D8', 5))
            if not total:
                parts.append(svg_text(x + CELL_W / 2, y + CELL_H / 2,
                                      'No 4 m TPs', 17, '#667586',
                                      anchor='middle'))
                continue
            parts.append(svg_text(x + 16, y + 29,
                                  'N = {}'.format(format_count(total)),
                                  18, weight='bold'))
            bar_x, bar_y, bar_w, bar_h = x + 16, y + 42, CELL_W - 32, 25
            left = bar_x
            for count, (_, _, _, color) in zip(counts, BANDS):
                width = bar_w * count / total
                if width:
                    parts.append(svg_rect(left, bar_y, width, bar_h, color))
                left += width
            parts.append(svg_rect(bar_x, bar_y, bar_w, bar_h, 'none',
                                  '#667586', 2))
            for index, (count, (token, _, _, color)) in enumerate(zip(counts, BANDS)):
                label_x = x + 16 + (index % 2) * 153
                label_y = y + 100 + (index // 2) * 34
                parts.append(svg_rect(label_x, label_y - 11, 12, 12,
                                      color, '#667586', 2))
                parts.append(svg_text(label_x + 20, label_y,
                                      'V{} {} ({:.1f}%)'.format(
                                          token, format_count(count),
                                          100 * count / total), 14))
    parts.append(svg_text(LEFT, PANEL_H - 25,
                          'Visibility is the nuScenes annotation category '
                          'across all cameras; counts sum to N in each cell.',
                          15, '#536374'))
    return ''.join(parts)


def write_svg_and_png(path, width, height, content):
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" '
           'width="{}" height="{}" viewBox="0 0 {} {}" '
           'font-family="Arial, Helvetica, sans-serif">{}</svg>').format(
               width, height, width, height, content)
    svg_path, png_path = path.with_suffix('.svg'), path.with_suffix('.png')
    svg_path.write_text(svg)
    converter = shutil.which('rsvg-convert')
    if converter:
        subprocess.run([converter, '-o', str(png_path), str(svg_path)],
                       check=True)
    else:
        try:
            import cairosvg
        except ImportError as error:
            raise RuntimeError('Install rsvg-convert or cairosvg to write PNG '
                               'from the generated SVG') from error
        cairosvg.svg2png(bytestring=svg.encode('utf-8'),
                         write_to=str(png_path))


def render_visibility_bands(rows, output_dir, names=None):
    """Write class figures and a large overview at the original figure path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if names is None:
        names = list(dict.fromkeys(row['class'] for row in rows))
    lookup = {(row['class'], row['distance_bin'], row['error_bin']): row
              for row in rows}
    for name in names:
        write_svg_and_png(output_dir / ('tp_visibility_bands_' +
                                        name.replace(' ', '_')),
                          PANEL_W, PANEL_H, panel_svg(name, lookup))
    columns = 2
    rows_of_panels = (len(names) + columns - 1) // columns
    gap = 20
    width = columns * PANEL_W + (columns + 1) * gap
    height = rows_of_panels * PANEL_H + (rows_of_panels + 1) * gap
    overview = [svg_rect(0, 0, width, height, '#E8EDF2')]
    for index, name in enumerate(names):
        x = gap + (index % columns) * (PANEL_W + gap)
        y = gap + (index // columns) * (PANEL_H + gap)
        overview.append('<g transform="translate({}, {})">{}</g>'.format(
            x, y, panel_svg(name, lookup)))
    write_svg_and_png(output_dir / 'tp_mean_visibility_heatmap',
                      width, height, ''.join(overview))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary-csv', type=Path, default=Path(__file__).parent /
                        'outputs/baseline/tp_context_by_class_range_error.csv')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).parent /
                        'outputs/baseline')
    args = parser.parse_args()
    with args.summary_csv.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    render_visibility_bands(rows, args.output_dir)
    print('Wrote visibility composition figures to {}'.format(args.output_dir))


if __name__ == '__main__':
    main()
