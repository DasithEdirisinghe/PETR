#!/usr/bin/env python3
"""Compare R1 and R1-f query attention at identical R1-f feature cells."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / 'experiments/cross_rig_attention/output/r1f_val/query_scorecard'
RED, TEAL, NAVY = '#c9574c', '#168176', '#183047'


def median(values):
    values = [float(value) for value in values if value is not None and
              np.isfinite(float(value))]
    return float(np.median(values)) if values else None


def display(value):
    return 'n/a' if value is None else '{:.3f}'.format(value)


def read_query_rows(path):
    with path.open(newline='') as handle:
        entries = list(csv.DictReader(handle))
    by_gt = {}
    for row in entries:
        by_gt.setdefault((row['sample_token'], int(row['gt_index'])), {})[
            row['model']] = row
    return by_gt


def normalize(values, valid):
    values = values.astype(np.float64)
    values[~valid] = 0
    return values/max(float(values.sum()), 1e-12)


def overlap(first, second, region):
    mass_first, mass_second = first[region].sum(), second[region].sum()
    absolute = float(np.minimum(first[region], second[region]).sum())
    conditional = (float(np.minimum(first[region]/mass_first,
                                     second[region]/mass_second).sum())
                   if mass_first > 1e-9 and mass_second > 1e-9 else None)
    return float(mass_first), float(mass_second), absolute, conditional


def compare_frame(token, input_dir, query_rows, result, cameras):
    paths = [input_dir / 'attention_maps' / model / (token+'.npz')
             for model in ('r1', 'r1f')]
    if not all(path.is_file() for path in paths):
        return
    with np.load(str(paths[0])) as r1, np.load(str(paths[1])) as r1f:
        indexes = [{int(gt): i for i, gt in enumerate(source['gt_indices'])}
                   for source in (r1, r1f)]
        if not np.array_equal(r1['camera_ids'], r1f['camera_ids']):
            raise AssertionError('Camera-token layouts differ in {}'.format(token))
        camera_ids = r1['camera_ids']
        for gt in sorted(set(indexes[0]) & set(indexes[1])):
            pair = query_rows.get((token, gt))
            if pair is None or set(pair) != {'r1', 'r1f'}:
                continue
            i, j = indexes[0][gt], indexes[1][gt]
            mask = r1['car_masks'][i]
            if not np.array_equal(mask, r1f['car_masks'][j]):
                raise AssertionError('GT ray masks differ in {} GT {}'.format(token, gt))
            for layer in (1, 6):
                valid = r1['valid_l{}'.format(layer)] & r1f[
                    'valid_l{}'.format(layer)]
                first = normalize(r1['attention_l{}'.format(layer)][i], valid)
                second = normalize(r1f['attention_l{}'.format(layer)][j], valid)
                car = mask & valid
                m1, m2, joint, conditional = overlap(first, second, car)
                full = float(np.minimum(first, second).sum())
                result.append({
                    'sample_token': token, 'gt_index': gt, 'layer': layer,
                    'gt_ray_count': int(car.sum()),
                    'r1_car_attention_mass': m1,
                    'r1f_car_attention_mass': m2,
                    'car_attention_mass_gap': abs(m1-m2),
                    'full_spatial_overlap': full,
                    'joint_car_attention_mass': joint,
                    'within_car_spatial_overlap': conditional,
                    'r1_car_score': float(pair['r1']['car_score']),
                    'r1_top_class_car': int(pair['r1']['top_class'] == 'car'),
                    'r1_center_error_m': float(pair['r1']['center_error_m']),
                    'r1f_car_score': float(pair['r1f']['car_score']),
                    'r1f_top_class_car': int(pair['r1f']['top_class'] == 'car'),
                    'r1f_center_error_m': float(pair['r1f']['center_error_m']),
                })
                for camera in np.unique(camera_ids):
                    on_camera = (camera_ids == camera) & valid
                    cmask = car & on_camera
                    c1, c2, cjoint, cconditional = overlap(
                        first, second, cmask)
                    cameras.append({
                        'sample_token': token, 'gt_index': gt,
                        'layer': layer, 'camera_index': int(camera),
                        'gt_ray_visible': int(cmask.any()),
                        'gt_ray_count': int(cmask.sum()),
                        'r1_camera_attention_mass': float(first[on_camera].sum()),
                        'r1f_camera_attention_mass': float(second[on_camera].sum()),
                        'r1_car_attention_mass': c1,
                        'r1f_car_attention_mass': c2,
                        'joint_car_attention_mass': cjoint,
                        'within_car_spatial_overlap': cconditional,
                    })


def write_csv(path, rows):
    if not rows:
        raise ValueError('No paired attention maps found')
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(rows, summary, output):
    selected = [r for r in rows if r['layer'] == 6 and
                r['r1f_top_class_car'] and r['r1f_car_score'] >= .35 and
                r['r1f_center_error_m'] < 4 and r['gt_ray_count']]
    if not selected:
        raise ValueError('No selected reference cars have GT-intersecting rays')
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    x = np.asarray([r['r1_car_attention_mass'] for r in selected])
    y = np.asarray([r['r1f_car_attention_mass'] for r in selected])
    axes[0, 0].scatter(x, y, s=16, alpha=.32, c=NAVY, edgecolors='none')
    bound = max(float(max(x.max(), y.max())), .01)
    axes[0, 0].plot([0, bound], [0, bound], '--', color='#8b9aab')
    axes[0, 0].set_xlim(0, bound)
    axes[0, 0].set_ylim(0, bound)
    axes[0, 0].set_xlabel('R1 attention mass on this GT car\'s rays')
    axes[0, 0].set_ylabel('R1-f attention mass on same rays')
    axes[0, 0].set_title('Amount of attention on the car')
    axes[0, 0].text(.04, .96, 'medians: R1 {:.3f} · R1-f {:.3f}'.format(
        median(x), median(y)), transform=axes[0, 0].transAxes, va='top')
    full = [r['full_spatial_overlap'] for r in selected]
    within = [r['within_car_spatial_overlap'] for r in selected
              if r['within_car_spatial_overlap'] is not None]
    bins = np.linspace(0, 1, 26)
    axes[0, 1].hist(full, bins=bins, color=NAVY, alpha=.65,
                    label='All image cells · median {:.2f}'.format(median(full)))
    axes[0, 1].hist(within, bins=bins, color=TEAL, alpha=.55,
                    label='Within GT-car rays · median {}'.format(
                        display(median(within))))
    axes[0, 1].set_xlabel('Spatial overlap: 1 = identical attention maps')
    axes[0, 1].set_ylabel('GT cars')
    axes[0, 1].set_title('Do they select the same image cells?')
    axes[0, 1].legend(frameon=False)
    qualified = [r for r in selected if r['r1_top_class_car'] and
                 r['r1_car_score'] >= .35 and
                 r['within_car_spatial_overlap'] is not None]
    axes[1, 0].scatter([r['within_car_spatial_overlap'] for r in qualified],
                       [min(r['r1_center_error_m'], 10) for r in qualified],
                       s=20, alpha=.4, color=RED, edgecolors='none')
    axes[1, 0].axhline(1, linestyle='--', color=NAVY, linewidth=1)
    axes[1, 0].set_xlim(0, 1)
    axes[1, 0].set_ylim(0, 10)
    axes[1, 0].set_xlabel('R1/R1-f overlap within GT-car rays')
    axes[1, 0].set_ylabel('R1 BEV center error (m; clipped at 10)')
    axes[1, 0].set_title('When R1 says “car”, does overlap track location?')
    axes[1, 1].axis('off')
    lines = [
        'Fixed R1-f reference cohort: {:,} cars'.format(len(selected)),
        '',
        'R1 top class car, score ≥ 0.35: {:,}'.format(len(qualified)),
        'Of those, center error ≥ 1 m: {:,}'.format(sum(
            r['r1_center_error_m'] >= 1 for r in qualified)),
        '',
        'Median full-map overlap: {:.3f}'.format(median(full)),
        'Median within-car overlap: {}'.format(display(median(within))),
        '',
        'Equal GT-ray mass does not imply equal cell selection.',
        'Equal spatial selection does not imply equal feature values.'
    ]
    axes[1, 1].text(.02, .98, '\n'.join(lines), transform=axes[1, 1].transAxes,
                    va='top', fontsize=13, linespacing=1.7, color=NAVY)
    for axis in (axes[0, 0], axes[0, 1], axes[1, 0]):
        axis.grid(alpha=.16)
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
    fig.suptitle('Same R1-f frame, same GT car: paired query attention',
                 x=.045, ha='left', fontsize=19, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(output / 'paired_attention.png', dpi=220)
    fig.savefig(output / 'paired_attention.svg')
    plt.close(fig)
    summary['reference_l6'] = {
        'n': len(selected), 'median_r1_car_mass': median(x),
        'median_r1f_car_mass': median(y),
        'median_full_spatial_overlap': median(full),
        'median_within_car_spatial_overlap': median(within),
        'r1_class_pass_n': len(qualified),
        'r1_class_pass_center_fail_n': sum(
            r['r1_center_error_m'] >= 1 for r in qualified)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=DEFAULT)
    args = parser.parse_args()
    query_rows = read_query_rows(args.input_dir / 'per_car_query.csv')
    tokens = sorted(set(token for token, _ in query_rows))
    results, cameras = [], []
    for token in tokens:
        compare_frame(token, args.input_dir, query_rows, results, cameras)
    write_csv(args.input_dir / 'paired_attention.csv', results)
    write_csv(args.input_dir / 'paired_camera_attention.csv', cameras)
    summary = {'paired_cars_l1': sum(r['layer'] == 1 for r in results),
               'paired_cars_l6': sum(r['layer'] == 6 for r in results)}
    plot(results, summary, args.input_dir)
    with (args.input_dir / 'paired_attention_summary.json').open('w') as handle:
        json.dump(summary, handle, indent=2)
    print('Saved paired attention maps comparison:', args.input_dir)


if __name__ == '__main__':
    main()
