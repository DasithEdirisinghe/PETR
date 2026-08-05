#!/usr/bin/env python3
"""Create an SVG comparison of two standardized PCCR result files."""

import argparse
import html
import json
from pathlib import Path


RIG_ORDER = [
    "R1", "R1-c10", "R1-c6", "R1-f", "R1-r", "R1-t",
    "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("new", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--reference-label", default="PCCR-reported PETR")
    parser.add_argument("--new-label", default="Newly trained PETR")
    return parser.parse_args()


def load_map(path):
    payload = json.loads(path.read_text())
    results = payload["results"]
    missing = [rig for rig in RIG_ORDER if rig not in results]
    if missing:
        raise ValueError("Missing rigs in {}: {}".format(path, ", ".join(missing)))
    return [100.0 * results[rig]["metrics"]["mAP"] for rig in RIG_ORDER]


def esc(value):
    return html.escape(str(value), quote=True)


def main():
    args = parse_args()
    reference = load_map(args.reference)
    new = load_map(args.new)
    delta = [new_value - ref_value for ref_value, new_value in zip(reference, new)]

    width, height = 1600, 1040
    left, right = 115, 55
    plot_width = width - left - right
    top_y, top_height = 145, 520
    delta_y, delta_height = 770, 145
    max_map = 18.0
    delta_min, delta_max = -3.2, 1.2
    step = plot_width / (len(RIG_ORDER) - 1)

    def x_pos(index):
        return left + index * step

    def map_y(value):
        return top_y + top_height * (1.0 - value / max_map)

    def delta_pos(value):
        return delta_y + delta_height * (delta_max - value) / (delta_max - delta_min)

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">'.format(
            width, height, width, height
        ),
        "<style>",
        "text{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Arial,sans-serif;fill:#253047}",
        ".title{font-size:30px;font-weight:700}.subtitle{font-size:17px;fill:#657087}",
        ".axis{font-size:15px;fill:#596579}.tick{font-size:14px;fill:#657087}",
        ".legend{font-size:16px;font-weight:600}.value{font-size:13px;font-weight:600}",
        "</style>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="{}" y="48" class="title">PETR cross-rig performance after training on R1</text>'.format(left),
        '<text x="{}" y="78" class="subtitle">mAP across camera-rig configurations; lower panel is newly trained minus PCCR-reported</text>'.format(left),
    ]

    split_x = (x_pos(5) + x_pos(6)) / 2
    parts.extend([
        '<rect x="{}" y="{}" width="{}" height="{}" rx="10" fill="#f1f6ff"/>'.format(
            left - step * 0.35, top_y, split_x - (left - step * 0.35), top_height
        ),
        '<rect x="{}" y="{}" width="{}" height="{}" rx="10" fill="#faf7f2"/>'.format(
            split_x, top_y, width - right - split_x + step * 0.35, top_height
        ),
        '<text x="{}" y="{}" class="axis" font-weight="600">R1 and controlled variants</text>'.format(left + 14, top_y + 25),
        '<text x="{}" y="{}" class="axis" font-weight="600">Alternative rig layouts</text>'.format(split_x + 14, top_y + 25),
    ])

    for tick in range(0, 19, 3):
        y = map_y(tick)
        parts.append('<line x1="{}" y1="{:.2f}" x2="{}" y2="{:.2f}" stroke="#d9dee8" stroke-width="1"/>'.format(left, y, width - right, y))
        parts.append('<text x="{}" y="{:.2f}" text-anchor="end" dominant-baseline="middle" class="tick">{}</text>'.format(left - 14, y, tick))
    parts.append('<text x="32" y="{}" transform="rotate(-90 32 {})" text-anchor="middle" class="axis" font-weight="600">mAP (%)</text>'.format(top_y + top_height / 2, top_y + top_height / 2))

    colors = ["#2563eb", "#ef7d32"]
    labels = [args.reference_label, args.new_label]
    series = [reference, new]
    for series_index, values in enumerate(series):
        points = " ".join("{:.2f},{:.2f}".format(x_pos(i), map_y(value)) for i, value in enumerate(values))
        parts.append('<polyline points="{}" fill="none" stroke="{}" stroke-width="4" stroke-linejoin="round" stroke-linecap="round"/>'.format(points, colors[series_index]))
        for i, value in enumerate(values):
            x, y = x_pos(i), map_y(value)
            parts.append('<circle cx="{:.2f}" cy="{:.2f}" r="6.5" fill="#fff" stroke="{}" stroke-width="4"><title>{}: {} — {:.2f}% mAP</title></circle>'.format(x, y, colors[series_index], esc(RIG_ORDER[i]), esc(labels[series_index]), value))

    legend_x = width - right - 470
    for i, (label, color) in enumerate(zip(labels, colors)):
        y = 102
        x = legend_x + i * 250
        parts.extend([
            '<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="4"/>'.format(x, y, x + 35, y, color),
            '<circle cx="{}" cy="{}" r="5" fill="#fff" stroke="{}" stroke-width="3"/>'.format(x + 17.5, y, color),
            '<text x="{}" y="{}" dominant-baseline="middle" class="legend">{}</text>'.format(x + 46, y, esc(label)),
        ])

    zero_y = delta_pos(0)
    parts.extend([
        '<text x="{}" y="{}" class="axis" font-weight="600">Difference (percentage points)</text>'.format(left, delta_y - 24),
        '<line x1="{}" y1="{:.2f}" x2="{}" y2="{:.2f}" stroke="#7f8898" stroke-width="1.5"/>'.format(left, zero_y, width - right, zero_y),
    ])
    for tick in [-3, -2, -1, 0, 1]:
        y = delta_pos(tick)
        parts.append('<text x="{}" y="{:.2f}" text-anchor="end" dominant-baseline="middle" class="tick">{:+d}</text>'.format(left - 14, y, tick))

    bar_width = min(48, step * 0.55)
    for i, value in enumerate(delta):
        x = x_pos(i) - bar_width / 2
        y = min(zero_y, delta_pos(value))
        bar_height = max(1.5, abs(delta_pos(value) - zero_y))
        color = "#1b9e77" if value > 0 else "#d84a4a" if value < 0 else "#9aa2af"
        parts.append('<rect x="{:.2f}" y="{:.2f}" width="{:.2f}" height="{:.2f}" rx="3" fill="{}"><title>{}: {:+.2f} percentage points</title></rect>'.format(x, y, bar_width, bar_height, color, esc(RIG_ORDER[i]), value))
        text_y = y - 8 if value >= 0 else y + bar_height + 16
        parts.append('<text x="{:.2f}" y="{:.2f}" text-anchor="middle" class="value" fill="{}">{:+.2f}</text>'.format(x_pos(i), text_y, color, value))

    for i, rig in enumerate(RIG_ORDER):
        parts.append('<text x="{:.2f}" y="965" text-anchor="middle" class="axis" font-weight="600">{}</text>'.format(x_pos(i), esc(rig)))
    parts.append('<text x="{}" y="1012" class="subtitle">Positive difference means the newly trained model achieved higher mAP.</text>'.format(left))
    parts.append("</svg>")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(parts))
    print(args.output)


if __name__ == "__main__":
    main()
