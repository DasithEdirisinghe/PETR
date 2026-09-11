#!/usr/bin/env python3
"""Plot one standardized PCCR model result across all camera rigs."""

import argparse
import html
import json
import math
from pathlib import Path


RIG_ORDER = [
    "R1", "R1-c10", "R1-c6", "R1-f", "R1-r", "R1-t",
    "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot a standardized trained_on_<rig>.json result.")
    parser.add_argument("result", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--label", default="PCCR-reported PETR")
    parser.add_argument("--title")
    return parser.parse_args()


def load_result(path):
    payload = json.loads(path.read_text())
    results = payload["results"]
    missing = [rig for rig in RIG_ORDER if rig not in results]
    if missing:
        raise ValueError(
            "Missing rigs in {}: {}".format(path, ", ".join(missing)))
    values = [100.0 * results[rig]["metrics"]["mAP"] for rig in RIG_ORDER]
    return payload["trained_on"], values


def esc(value):
    return html.escape(str(value), quote=True)


def main():
    args = parse_args()
    train_rig, values = load_result(args.result)
    if train_rig not in RIG_ORDER:
        raise ValueError("Training rig is not in the plot order: {}".format(train_rig))

    width, height = 1600, 780
    left, right = 115, 55
    plot_width = width - left - right
    top_y, plot_height = 150, 480
    step = plot_width / (len(RIG_ORDER) - 1)
    max_map = max(15.0, 3.0 * math.ceil(max(values) / 3.0))
    color = "#2563eb"
    highlight = "#159570"
    title = args.title or (
        "PETR cross-rig performance after training on {}".format(train_rig))

    def x_pos(index):
        return left + index * step

    def map_y(value):
        return top_y + plot_height * (1.0 - value / max_map)

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">'.format(
            width, height, width, height),
        "<style>",
        "text{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Arial,sans-serif;fill:#253047}",
        ".title{font-size:30px;font-weight:700}.subtitle{font-size:17px;fill:#657087}",
        ".axis{font-size:15px;fill:#596579}.tick{font-size:14px;fill:#657087}",
        ".legend{font-size:16px;font-weight:600}.value{font-size:13px;font-weight:700}",
        "</style>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="{}" y="48" class="title">{}</text>'.format(left, esc(title)),
        '<text x="{}" y="78" class="subtitle">mAP across camera-rig configurations; the filled marker is the in-domain {} test set</text>'.format(
            left, esc(train_rig)),
    ]

    split_x = (x_pos(5) + x_pos(6)) / 2
    parts.extend([
        '<rect x="{}" y="{}" width="{}" height="{}" rx="10" fill="#f1f6ff"/>'.format(
            left - step * 0.35, top_y,
            split_x - (left - step * 0.35), plot_height),
        '<rect x="{}" y="{}" width="{}" height="{}" rx="10" fill="#faf7f2"/>'.format(
            split_x, top_y, width - right - split_x + step * 0.35,
            plot_height),
        '<text x="{}" y="{}" class="axis" font-weight="600">R1 and controlled variants</text>'.format(
            left + 14, top_y + 25),
        '<text x="{}" y="{}" class="axis" font-weight="600">Alternative rig layouts</text>'.format(
            split_x + 14, top_y + 25),
    ])

    for tick in range(0, int(max_map) + 1, 3):
        y = map_y(tick)
        parts.append(
            '<line x1="{}" y1="{:.2f}" x2="{}" y2="{:.2f}" stroke="#d9dee8" stroke-width="1"/>'.format(
                left, y, width - right, y))
        parts.append(
            '<text x="{}" y="{:.2f}" text-anchor="end" dominant-baseline="middle" class="tick">{}</text>'.format(
                left - 14, y, tick))
    parts.append(
        '<text x="32" y="{}" transform="rotate(-90 32 {})" text-anchor="middle" class="axis" font-weight="600">mAP (%)</text>'.format(
            top_y + plot_height / 2, top_y + plot_height / 2))

    points = " ".join(
        "{:.2f},{:.2f}".format(x_pos(i), map_y(value))
        for i, value in enumerate(values))
    parts.append(
        '<polyline points="{}" fill="none" stroke="{}" stroke-width="4" stroke-linejoin="round" stroke-linecap="round"/>'.format(
            points, color))

    for i, value in enumerate(values):
        rig = RIG_ORDER[i]
        x, y = x_pos(i), map_y(value)
        is_train_rig = rig == train_rig
        fill = highlight if is_train_rig else "#ffffff"
        stroke = highlight if is_train_rig else color
        radius = 8 if is_train_rig else 6.5
        parts.append(
            '<circle cx="{:.2f}" cy="{:.2f}" r="{}" fill="{}" stroke="{}" stroke-width="4"><title>{}: {:.2f}% mAP{}</title></circle>'.format(
                x, y, radius, fill, stroke, esc(rig), value,
                " (in-domain)" if is_train_rig else ""))
        label_y = y - 14 if value >= 0.8 else y - 12
        parts.append(
            '<text x="{:.2f}" y="{:.2f}" text-anchor="middle" class="value" fill="{}">{:.2f}</text>'.format(
                x, label_y, stroke, value))

    legend_x, legend_y = left, 106
    parts.extend([
        '<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="4"/>'.format(
            legend_x, legend_y, legend_x + 35, legend_y, color),
        '<circle cx="{}" cy="{}" r="5" fill="#fff" stroke="{}" stroke-width="3"/>'.format(
            legend_x + 17.5, legend_y, color),
        '<text x="{}" y="{}" dominant-baseline="middle" class="legend">{}</text>'.format(
            legend_x + 46, legend_y, esc(args.label)),
    ])

    for i, rig in enumerate(RIG_ORDER):
        parts.append(
            '<text x="{:.2f}" y="682" text-anchor="middle" class="axis" font-weight="600">{}</text>'.format(
                x_pos(i), esc(rig)))
    parts.append(
        '<text x="{}" y="740" class="subtitle">Source: standardized PCCR-reported result {}</text>'.format(
            left, esc(args.result.name)))
    parts.append("</svg>")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(parts))
    print(args.output)


if __name__ == "__main__":
    main()
