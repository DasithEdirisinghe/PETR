#!/usr/bin/env python3
"""Compare saved R1 car-TP and car-FP attention-region mass summaries."""

import argparse
import html
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'experiments/cross_rig_attention/output/r1f_val'
REGIONS = (
    ('car', 'Car GT rays'),
    ('other_object', 'Other-object GT rays'),
    ('no_annotated_gt', 'No annotated GT rays'),
)
TP_COLOR = '#087e83'
FP_COLOR = '#d75b43'


def element(tag, **attrs):
    return '<{} {} />'.format(tag, ' '.join(
        '{}="{}"'.format(key.replace('_', '-'), html.escape(str(value), quote=True))
        for key, value in attrs.items()))


def label(x, y, value, size=19, color='#172b41', weight=400, anchor=None):
    attrs = {'x': x, 'y': y, 'font-size': size, 'fill': color,
             'font-weight': weight}
    if anchor:
        attrs['text-anchor'] = anchor
    return '<text {}>{}</text>'.format(' '.join(
        '{}="{}"'.format(k, html.escape(str(v), quote=True))
        for k, v in attrs.items()), html.escape(str(value)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tp-summary', type=Path, default=BASE /
                        'tp_attention_regions/r1/summary.json')
    parser.add_argument('--fp-summary', type=Path, default=BASE /
                        'r1_fp_attention_regions/summary.json')
    parser.add_argument('--output', type=Path, default=BASE /
                        'r1_tp_fp_attention_mass_comparison.svg')
    args = parser.parse_args()
    with args.tp_summary.open() as handle:
        tp = json.load(handle)
    with args.fp_summary.open() as handle:
        fp = json.load(handle)
    assert tp['model'] == 'R1' and tp['test_set'] == 'r1f_val'
    assert 'R1 car FPs on R1-f validation' in fp['cohort']
    assert tp['score_threshold'] == fp['score_threshold'] == 0.6
    assert tp['box_expansion_m'] == fp['box_expansion_m'] == 1.0
    assert tp['is_complete'] and fp['is_complete']

    n_tp = tp['traced_tp_count']
    n_fp = fp['traced_fp_count']
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1500" height="1050" viewBox="0 0 1500 1050">',
        element('rect', x=0, y=0, width=1500, height=1050, fill='#f3f6f9'),
        element('rect', x=34, y=32, width=1432, height=986, rx=22, fill='white'),
        label(76, 91, 'Where does R1 car-query attention fall?', 34, weight=700),
        label(76, 129, 'R1 model on R1-f validation frames | final car score >= 0.60 | official car AP @ 4 m', 18, '#4d6074'),
        element('circle', cx=89, cy=171, r=8, fill=TP_COLOR),
        label(108, 177, 'True positives: {} matched car predictions'.format(n_tp), 18),
        element('circle', cx=577, cy=171, r=8, fill=FP_COLOR),
        label(596, 177, 'False positives: {} unmatched car predictions'.format(n_fp), 18),
        label(76, 227, 'Each number is the median fraction of a query\'s final-layer attention weight', 18, weight=600),
        label(76, 253, 'assigned to that region across all cameras (heads averaged; GT boxes expanded by 1 m).', 18),
        element('line', x1=77, y1=279, x2=1423, y2=279, stroke='#dce4ec'),
        label(76, 317, 'Image-ray region', 19, weight=700),
        label(420, 317, 'Attention mass: 0-100%', 19, weight=700),
        label(1210, 317, 'TP - FP', 19, weight=700, anchor='middle'),
        label(1390, 317, 'Region size', 19, weight=700, anchor='end'),
    ]
    bar_x, bar_width = 420, 610
    for tick in (0, 25, 50, 75, 100):
        x = bar_x + bar_width * tick / 100
        parts.append(element('line', x1=x, y1=340, x2=x, y2=690,
                             stroke='#e6edf2', stroke_width=1))
        parts.append(label(x, 714, '{}%'.format(tick), 16, '#66798b', anchor='middle'))
    for index, (key, name) in enumerate(REGIONS):
        y = 376 + index * 121
        tp_mass = 100 * tp['regions'][key]['median_attention_mass']
        fp_mass = 100 * fp['all_r1_fps']['regions'][key]['median_attention_mass']
        tp_cells = 100 * tp['regions'][key]['median_cell_fraction']
        fp_cells = 100 * fp['all_r1_fps']['regions'][key]['median_cell_fraction']
        parts.extend((
            label(76, y + 23, name, 21, weight=700),
            element('rect', x=bar_x, y=y, width=bar_width * tp_mass / 100,
                    height=23, rx=5, fill=TP_COLOR),
            element('rect', x=bar_x, y=y + 34, width=bar_width * fp_mass / 100,
                    height=23, rx=5, fill=FP_COLOR),
            label(1053, y + 20, 'TP {:.1f}%'.format(tp_mass), 18, TP_COLOR, 700),
            label(1053, y + 55, 'FP {:.1f}%'.format(fp_mass), 18, FP_COLOR, 700),
            label(1210, y + 40, '{:+.1f} pp'.format(tp_mass - fp_mass),
                  21, '#172b41', 700, 'middle'),
            label(1390, y + 20, 'TP {:.1f}%'.format(tp_cells), 17,
                  '#4d6074', anchor='end'),
            label(1390, y + 54, 'FP {:.1f}%'.format(fp_cells), 17,
                  '#4d6074', anchor='end'),
        ))
    parts.extend((
        element('rect', x=76, y=744, width=1348, height=116, rx=12, fill='#e8f3f2'),
        label(96, 775, 'How attention mass is calculated for one query q', 18, weight=700),
        label(96, 810, 'M_q(R) = [Σ_{i∈R∩V} a_{q,i}] / [Σ_{i∈V} a_{q,i}]', 25, '#087e83', 700),
        label(96, 839, 'a = last-decoder attention averaged over heads; V = valid cells in all cameras; R = one GT-ray region.', 17),
        element('rect', x=76, y=878, width=1348, height=108, rx=12, fill='#eaf0f5'),
        label(96, 908, 'Read with care', 18, weight=700),
        label(96, 936, 'TP and FP are different predictions/scenes; region-size differences make raw mass non-causal.', 17),
        label(96, 962, 'Plotted values are separate medians over queries; the three cohort medians need not sum to 100%.', 17),
        '</svg>',
    ))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text('\n'.join(parts) + '\n')
    print(args.output)


if __name__ == '__main__':
    main()
