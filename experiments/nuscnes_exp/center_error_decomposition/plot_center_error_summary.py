#!/usr/bin/env python3
"""Plot the baseline PETR center-error summary with Matplotlib."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402


def main():
    experiment_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, default=experiment_dir /
                        'outputs/baseline/center_error_summary.json')
    parser.add_argument('--output-dir', type=Path, default=experiment_dir /
                        'outputs/baseline/plots')
    args = parser.parse_args()

    with args.summary.open() as handle:
        summary = json.load(handle)
    bins = summary['bins']
    labels = [row['center_error_bin_m'].replace('.0-', '–').replace('-', '–')
              for row in bins]
    positions = list(range(len(bins)))
    percentages = [row.get('percent_of_4m_tps',
                           100 * row.get('fraction_of_4m_tps', 0)) for row in bins]

    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.4), dpi=180)
    fig.suptitle('Baseline PETR: center errors among 4 m true positives',
                 fontsize=17, fontweight='bold')
    fig.text(0.5, 0.925,
             '{:,} matched predictions · ego-centered BEV radial/tangential axes'
             .format(summary['matches']), ha='center', color='#53616c')

    ax = axes[0, 0]
    width = 0.25
    series = (
        ('Center error', 'mean_center_error_m', '#34495e'),
        ('|Radial|', 'mean_abs_radial_m', '#2878b5'),
        ('|Tangential|', 'mean_abs_tangential_m', '#e29c45'),
    )
    for offset, (name, key, color) in enumerate(series):
        ax.bar([p + (offset - 1) * width for p in positions],
               [row[key] for row in bins], width=width,
               label=name, color=color, zorder=3)
    ax.set_title('Mean localization error')
    ax.set_ylabel('Metres')
    ax.legend(frameon=False, fontsize=9)

    ax = axes[0, 1]
    counts = [row['count'] for row in bins]
    bars = ax.bar(positions, counts, color='#638b7a', width=0.64, zorder=3)
    ax.set_ylim(0, max(counts) * 1.2)
    ax.set_title('4 m true positives by center-error band')
    ax.set_ylabel('Matched predictions')
    for bar, count, percentage in zip(bars, counts, percentages):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts) * 0.02,
                '{:,}\n{:.1f}%'.format(count, percentage),
                ha='center', va='bottom', fontsize=9)

    ax = axes[1, 0]
    ax.plot(positions, [row['mean_score'] for row in bins], 'o-',
            color='#8b5ca6', linewidth=2.4, markersize=7)
    ax.set_ylim(0, 1)
    ax.set_title('Mean detection confidence')
    ax.set_ylabel('Score')
    for p, row in zip(positions, bins):
        ax.annotate('{:.2f}'.format(row['mean_score']),
                    (p, row['mean_score']), xytext=(0, 8),
                    textcoords='offset points', ha='center', fontsize=9)

    ax = axes[1, 1]
    ax.plot(positions, [row['mean_gt_range_m'] for row in bins], 'o-',
            color='#d45f54', linewidth=2.4, markersize=7)
    ax.set_ylim(0, max(row['mean_gt_range_m'] for row in bins) * 1.2)
    ax.set_title('Mean GT range from ego')
    ax.set_ylabel('Metres')
    for p, row in zip(positions, bins):
        ax.annotate('{:.1f} m'.format(row['mean_gt_range_m']),
                    (p, row['mean_gt_range_m']), xytext=(0, 8),
                    textcoords='offset points', ha='center', fontsize=9)

    for ax in axes.flat:
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        ax.set_xlabel('Center error (m)')
        ax.grid(axis='y', alpha=0.18, zorder=0)
    fig.text(0.5, 0.015,
             'Counts and means are conditional on same-class matches with center error < 4 m.',
             ha='center', fontsize=9, color='#53616c')
    fig.tight_layout(rect=(0, 0.045, 1, 0.91), h_pad=2.5, w_pad=2.0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ('png', 'svg'):
        path = args.output_dir / ('center_error_summary.' + extension)
        fig.savefig(path, bbox_inches='tight')
        print(path)
    plt.close(fig)


if __name__ == '__main__':
    main()
