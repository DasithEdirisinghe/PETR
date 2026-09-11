#!/usr/bin/env python3
"""Create an SVG comparison of two to five standardized PCCR result files."""

import argparse
import html
import json
import math
from pathlib import Path


RIG_ORDER = [
    "R1", "R1-c10", "R1-c6", "R1-f", "R1-r", "R1-t",
    "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9",
]

PE_COMPONENTS = [
    ("petr3dpe", "PETR 3DPE"),
    ("multiview", "Multiview PE"),
    ("urope", "URoPE Q/K"),
    ("lidar_oracle", "LiDAR Oracle PE"),
    ("query3d", "Query 3D PE"),
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("new", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--reference-label", default="PCCR-reported PETR")
    parser.add_argument("--new-label", default="Newly trained PETR")
    parser.add_argument(
        "--reference-components",
        help="Comma-separated enabled PE components for the reference model.",
    )
    parser.add_argument(
        "--new-components",
        help="Comma-separated enabled PE components for the new model.",
    )
    parser.add_argument("--train-rig", default="R1")
    parser.add_argument(
        "--third",
        type=Path,
        help="Optional third standardized result file, such as PETR-URoPE.",
    )
    parser.add_argument("--third-label", default="PETR + URoPE")
    parser.add_argument("--third-components")
    parser.add_argument(
        "--fourth",
        type=Path,
        help="Optional fourth standardized result file, such as a no-3DPE ablation.",
    )
    parser.add_argument("--fourth-label", default="PETR without 3D PE")
    parser.add_argument("--fourth-components")
    parser.add_argument(
        "--fifth",
        type=Path,
        help="Optional fifth standardized result file, such as a hybrid model.",
    )
    parser.add_argument("--fifth-label", default="PETR 3DPE + URoPE")
    parser.add_argument("--fifth-components")
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


def parse_components(specification):
    """Parse and validate one model's enabled PE component names."""
    if specification is None:
        return None
    enabled = {
        component.strip()
        for component in specification.split(",")
        if component.strip()
    }
    valid = {name for name, _ in PE_COMPONENTS}
    unknown = sorted(enabled - valid)
    if unknown:
        raise ValueError(
            "Unknown PE component(s): {}. Valid names: {}".format(
                ", ".join(unknown), ", ".join(sorted(valid))
            )
        )
    return enabled


def nice_tick_step(span, target_intervals=6):
    """Choose a readable 1/2/2.5/5/10-based tick interval."""
    if span <= 0:
        return 1.0
    rough_step = span / target_intervals
    magnitude = 10 ** math.floor(math.log10(rough_step))
    normalized = rough_step / magnitude
    for candidate in (1.0, 2.0, 2.5, 5.0, 10.0):
        if normalized <= candidate:
            return candidate * magnitude
    return 10.0 * magnitude


def main():
    args = parse_args()
    reference = load_map(args.reference)
    new = load_map(args.new)
    series = [reference, new]
    labels = [args.reference_label, args.new_label]
    component_sets = [
        parse_components(args.reference_components),
        parse_components(args.new_components),
    ]
    if args.third:
        series.append(load_map(args.third))
        labels.append(args.third_label)
        component_sets.append(parse_components(args.third_components))
    if args.fourth:
        if not args.third:
            raise ValueError("--fourth requires --third")
        series.append(load_map(args.fourth))
        labels.append(args.fourth_label)
        component_sets.append(parse_components(args.fourth_components))
    if args.fifth:
        if not args.fourth:
            raise ValueError("--fifth requires --fourth")
        series.append(load_map(args.fifth))
        labels.append(args.fifth_label)
        component_sets.append(parse_components(args.fifth_components))

    has_component_legend = any(
        components is not None for components in component_sets
    )

    deltas = [
        [value - ref_value for ref_value, value in zip(reference, values)]
        for values in series[1:]
    ]

    width = 1600
    height = 1180 if has_component_legend else 1040
    left, right = 115, 55
    plot_width = width - left - right
    if has_component_legend:
        top_y, top_height = 325, 500
        delta_y, delta_height = 910, 140
        rig_label_y = 1100
        footer_y = 1150
    else:
        top_y, top_height = 145, 520
        delta_y, delta_height = 770, 145
        rig_label_y = 965
        footer_y = 1012
    peak_map = max(max(values) for values in series)
    max_map = max(18.0, 3.0 * math.ceil((peak_map + 0.75) / 3.0))
    all_delta_values = [value for values in deltas for value in values]
    raw_delta_min = min(0.0, min(all_delta_values))
    raw_delta_max = max(0.0, max(all_delta_values))
    delta_tick_step = nice_tick_step(raw_delta_max - raw_delta_min)
    delta_tick_min = (
        math.floor(raw_delta_min / delta_tick_step) * delta_tick_step
    )
    delta_tick_max = (
        math.ceil(raw_delta_max / delta_tick_step) * delta_tick_step
    )
    delta_min = delta_tick_min - 0.08 * delta_tick_step
    delta_max = delta_tick_max + 0.08 * delta_tick_step
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
        '<text x="{}" y="48" class="title">PETR cross-rig performance after training on {}</text>'.format(
            left, esc(args.train_rig)),
        '<text x="{}" y="78" class="subtitle">mAP across camera-rig configurations; lower panel shows each variant minus {}</text>'.format(
            left, esc(labels[0])
        ),
    ]

    colors = ["#2563eb", "#ef7d32", "#159570", "#8b5cf6", "#dc2626"]
    if has_component_legend:
        table_top = 101
        model_column_width = 470
        component_width = (plot_width - model_column_width) / len(PE_COMPONENTS)
        row_height = 29
        table_bottom = table_top + row_height * (len(labels) + 1) + 25
        parts.extend([
            '<rect x="{}" y="{}" width="{}" height="{}" rx="10" fill="#f8fafc" stroke="#d9dee8"/>'.format(
                left, table_top - 15, plot_width, table_bottom - table_top + 20
            ),
            '<text x="{}" y="{}" class="axis" font-weight="700">Model / positional components</text>'.format(
                left + 12, table_top + 7
            ),
        ])
        for component_index, (_, display_name) in enumerate(PE_COMPONENTS):
            component_x = (
                left + model_column_width +
                (component_index + 0.5) * component_width
            )
            parts.append(
                '<text x="{:.2f}" y="{}" text-anchor="middle" class="axis" font-weight="700">{}</text>'.format(
                    component_x, table_top + 7, esc(display_name)
                )
            )

        for model_index, (label, color, enabled) in enumerate(
                zip(labels, colors, component_sets)):
            row_y = table_top + (model_index + 1) * row_height
            if model_index:
                parts.append(
                    '<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="#e5e9f0"/>'.format(
                        left + 8, row_y - 17, width - right - 8, row_y - 17
                    )
                )
            parts.extend([
                '<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="4"/>'.format(
                    left + 12, row_y, left + 47, row_y, color
                ),
                '<circle cx="{}" cy="{}" r="5" fill="#fff" stroke="{}" stroke-width="3"/>'.format(
                    left + 29.5, row_y, color
                ),
                '<text x="{}" y="{}" dominant-baseline="middle" class="legend">{}</text>'.format(
                    left + 59, row_y, esc(label)
                ),
            ])
            for component_index, (component_name, _) in enumerate(PE_COMPONENTS):
                component_x = (
                    left + model_column_width +
                    (component_index + 0.5) * component_width
                )
                if enabled is None:
                    symbol, fill, symbol_color = "?", "#f1f3f6", "#7b8494"
                elif component_name in enabled:
                    symbol, fill, symbol_color = "&#10003;", "#dcf5e8", "#137a4c"
                else:
                    symbol, fill, symbol_color = "&#8212;", "#eef1f5", "#8992a2"
                parts.extend([
                    '<rect x="{:.2f}" y="{}" width="38" height="22" rx="6" fill="{}"/>'.format(
                        component_x - 19, row_y - 11, fill
                    ),
                    '<text x="{:.2f}" y="{}" text-anchor="middle" dominant-baseline="middle" font-size="16" font-weight="700" fill="{}">{}</text>'.format(
                        component_x, row_y, symbol_color, symbol
                    ),
                ])
        parts.append(
            '<text x="{}" y="{}" class="subtitle">&#10003; enabled   &#8212; disabled   ? unspecified; Multiview PE = camera + row + column</text>'.format(
                left + 12, table_bottom - 2
            )
        )

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

    for tick in range(0, int(max_map) + 1, 3):
        y = map_y(tick)
        parts.append('<line x1="{}" y1="{:.2f}" x2="{}" y2="{:.2f}" stroke="#d9dee8" stroke-width="1"/>'.format(left, y, width - right, y))
        parts.append('<text x="{}" y="{:.2f}" text-anchor="end" dominant-baseline="middle" class="tick">{}</text>'.format(left - 14, y, tick))
    parts.append('<text x="32" y="{}" transform="rotate(-90 32 {})" text-anchor="middle" class="axis" font-weight="600">mAP (%)</text>'.format(top_y + top_height / 2, top_y + top_height / 2))

    for series_index, values in enumerate(series):
        points = " ".join("{:.2f},{:.2f}".format(x_pos(i), map_y(value)) for i, value in enumerate(values))
        parts.append('<polyline points="{}" fill="none" stroke="{}" stroke-width="4" stroke-linejoin="round" stroke-linecap="round"/>'.format(points, colors[series_index]))
        for i, value in enumerate(values):
            x, y = x_pos(i), map_y(value)
            parts.append('<circle cx="{:.2f}" cy="{:.2f}" r="6.5" fill="#fff" stroke="{}" stroke-width="4"><title>{}: {} — {:.2f}% mAP</title></circle>'.format(x, y, colors[series_index], esc(RIG_ORDER[i]), esc(labels[series_index]), value))

    if not has_component_legend:
        legend_width = plot_width / len(labels)
        for i, (label, color) in enumerate(zip(labels, colors)):
            y = 102
            x = left + i * legend_width
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
    delta_tick_count = int(round(
        (delta_tick_max - delta_tick_min) / delta_tick_step
    ))
    for tick_index in range(delta_tick_count + 1):
        tick = delta_tick_min + tick_index * delta_tick_step
        y = delta_pos(tick)
        tick_label = "{:+g}".format(tick)
        parts.append('<text x="{}" y="{:.2f}" text-anchor="end" dominant-baseline="middle" class="tick">{}</text>'.format(left - 14, y, tick_label))

    group_width = min(90, step * 0.82)
    bar_width = group_width / len(deltas)
    for delta_index, values in enumerate(deltas):
        color = colors[delta_index + 1]
        for i, value in enumerate(values):
            x = x_pos(i) - group_width / 2 + delta_index * bar_width
            y = min(zero_y, delta_pos(value))
            bar_height = max(1.5, abs(delta_pos(value) - zero_y))
            parts.append('<rect x="{:.2f}" y="{:.2f}" width="{:.2f}" height="{:.2f}" rx="3" fill="{}"><title>{}: {} vs {}: {:+.2f} percentage points</title></rect>'.format(x, y, bar_width - 2, bar_height, color, esc(RIG_ORDER[i]), esc(labels[delta_index + 1]), esc(labels[0]), value))
            if len(deltas) <= 3:
                text_y = y - 7 if value >= 0 else y + bar_height + 14
                parts.append('<text x="{:.2f}" y="{:.2f}" text-anchor="middle" class="value" fill="{}">{:+.2f}</text>'.format(x + (bar_width - 2) / 2, text_y, color, value))

    for i, rig in enumerate(RIG_ORDER):
        parts.append('<text x="{:.2f}" y="{}" text-anchor="middle" class="axis" font-weight="600">{}</text>'.format(x_pos(i), rig_label_y, esc(rig)))
    parts.append('<text x="{}" y="{}" class="subtitle">Positive difference means the corresponding variant achieved higher mAP than {}.</text>'.format(
        left, footer_y, esc(labels[0])
    ))
    parts.append("</svg>")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(parts))
    print(args.output)


if __name__ == "__main__":
    main()
