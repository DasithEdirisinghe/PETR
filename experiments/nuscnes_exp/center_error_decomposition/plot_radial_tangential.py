#!/usr/bin/env python3
"""Draw a radial/tangential geometry diagram beside measured baseline errors."""

import argparse
import json
from pathlib import Path


def main():
    experiment_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, default=experiment_dir /
                        'outputs/baseline/center_error_summary.json')
    parser.add_argument('--output', type=Path, default=experiment_dir /
                        'outputs/baseline/plots/radial_tangential.svg')
    args = parser.parse_args()
    with args.summary.open() as handle:
        bins = json.load(handle)['bins']

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="650" '
        'viewBox="0 0 1200 650">',
        '<defs><marker id="blue-arrow" viewBox="0 0 10 10" refX="9" refY="5" '
        'markerWidth="8" markerHeight="8" orient="auto-start-reverse">'
        '<path d="M 0 0 L 10 5 L 0 10 z" fill="#2878b5"/></marker>'
        '<marker id="orange-arrow" viewBox="0 0 10 10" refX="9" refY="5" '
        'markerWidth="8" markerHeight="8" orient="auto-start-reverse">'
        '<path d="M 0 0 L 10 5 L 0 10 z" fill="#de8b32"/></marker>'
        '</defs>',
        '<rect width="1200" height="650" fill="#f5f8fb"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#25313b}'
        '.title{font-size:27px;font-weight:700}.subtitle{font-size:15px;fill:#566570}'
        '.panel{font-size:19px;font-weight:700}.label{font-size:15px;font-weight:600}'
        '.small{font-size:13px;fill:#566570}.tick{font-size:12px;fill:#65737e}'
        '.value{font-size:13px;font-weight:700}</style>',
        '<text x="600" y="48" text-anchor="middle" class="title">'
        'Radial and tangential center error</text>',
        '<text x="600" y="76" text-anchor="middle" class="subtitle">'
        'Ego-centered BEV geometry and baseline PETR measurements</text>',
        '<rect x="36" y="108" width="550" height="478" rx="15" fill="white" '
        'stroke="#dce5ec"/>',
        '<rect x="614" y="108" width="550" height="478" rx="15" fill="white" '
        'stroke="#dce5ec"/>',
        '<text x="62" y="147" class="panel">One illustrative detection</text>',
        '<text x="640" y="147" class="panel">Measured mean absolute error</text>',
        # Geometry: ego (135,470), GT (355,310), radial projection (450,241),
        # prediction (403,176). The example coordinates are illustrative.
        '<line x1="135" y1="470" x2="355" y2="310" stroke="#a7b7c4" '
        'stroke-width="3" stroke-dasharray="7 6"/>',
        '<line x1="355" y1="310" x2="450" y2="241" stroke="#2878b5" '
        'stroke-width="5" marker-end="url(#blue-arrow)"/>',
        '<line x1="450" y1="241" x2="403" y2="176" stroke="#de8b32" '
        'stroke-width="5" marker-end="url(#orange-arrow)"/>',
        '<line x1="355" y1="310" x2="403" y2="176" stroke="#46515a" '
        'stroke-width="2" stroke-dasharray="6 5"/>',
        '<text x="215" y="213" class="label">BEV center distance</text>',
        '<line x1="352" y1="217" x2="380" y2="244" stroke="#46515a" '
        'stroke-width="1.5"/>',
        '<circle cx="135" cy="470" r="9" fill="#536573"/>',
        '<circle cx="355" cy="310" r="9" fill="#1e946c"/>',
        '<circle cx="403" cy="176" r="9" fill="#a45b9e"/>',
        '<circle cx="450" cy="241" r="5" fill="#74838f"/>',
        '<text x="76" y="498" class="label">Ego</text>',
        '<text x="298" y="345" class="label">GT center</text>',
        '<text x="418" y="170" class="label">Predicted center</text>',
        '<text x="397" y="295" class="label" fill="#2878b5">Radial</text>',
        '<text x="461" y="212" class="label" fill="#de8b32">Tangential</text>',
        '<text x="69" y="513" class="small">'
        'Distance = sqrt((pred x - GT x)^2 + (pred y - GT y)^2)</text>',
        '<text x="69" y="538" class="small">Radial: toward or away from ego</text>',
        '<text x="69" y="558" class="small">'
        'Tangential: sideways, perpendicular to radial; height is excluded</text>',
        '<rect x="642" y="168" width="15" height="15" rx="2" fill="#2878b5"/>',
        '<text x="664" y="181" class="small">|Radial|</text>',
        '<rect x="759" y="168" width="15" height="15" rx="2" fill="#de8b32"/>',
        '<text x="781" y="181" class="small">|Tangential|</text>',
    ]

    x0, plot_width = 755, 310
    for tick in (0, 1, 2, 3):
        x = x0 + plot_width * tick / 3
        parts.append('<line x1="{:.1f}" y1="209" x2="{:.1f}" y2="518" '
                     'stroke="#e7edf2"/>'.format(x, x))
        parts.append('<text x="{:.1f}" y="540" text-anchor="middle" '
                     'class="tick">{}</text>'.format(x, tick))
    parts.append('<text x="910" y="563" text-anchor="middle" class="small">Metres</text>')
    for index, row in enumerate(bins):
        y = 227 + index * 79
        label = row['center_error_bin_m'].replace('-', '–')
        radial = row['mean_abs_radial_m']
        tangential = row['mean_abs_tangential_m']
        rw = plot_width * radial / 3
        tw = plot_width * tangential / 3
        parts.extend((
            '<text x="735" y="{}" text-anchor="end" class="label">{} m</text>'
            .format(y + 20, label),
            '<rect x="{}" y="{}" width="{:.1f}" height="16" rx="3" '
            'fill="#2878b5"/>'.format(x0, y, rw),
            '<rect x="{}" y="{}" width="{:.1f}" height="16" rx="3" '
            'fill="#de8b32"/>'.format(x0, y + 22, tw),
            '<text x="{:.1f}" y="{}" class="value">{:.2f}</text>'
            .format(x0 + rw + 7, y + 13, radial),
            '<text x="{:.1f}" y="{}" class="value">{:.2f}</text>'
            .format(x0 + tw + 7, y + 35, tangential),
        ))
    parts.extend((
        '<text x="600" y="622" text-anchor="middle" class="small">'
        'Bars summarize predictions matched within 4 m; the left diagram is illustrative.</text>',
        '</svg>',
    ))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text('\n'.join(parts) + '\n')
    print(args.output)


if __name__ == '__main__':
    main()
