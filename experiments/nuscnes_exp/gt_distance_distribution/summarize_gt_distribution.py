#!/usr/bin/env python3
"""Count nuScenes train/val keyframe GT by BEV distance from the ego vehicle.

Uses standard-library streaming JSON parsing, so the large nuScenes tables do
not need to fit in memory. The validation scene list is the 150-scene list
saved by compare_error_bands.py when it was run with --all-val.
"""
import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

CLASSES = ('car', 'truck', 'construction_vehicle', 'bus', 'trailer',
           'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone')
EDGES = (10, 20, 30, 40, 50)
BINS = ('0–10', '10–20', '20–30', '30–40', '40–50', '50+')
VISIBILITY_TOKENS = ('1', '2', '3', '4', 'unknown')
VISIBILITY_FIELDS = (
    'split', 'class', 'distance_bin', 'gt_count',
    'visibility_0_40_count', 'visibility_40_60_count',
    'visibility_60_80_count', 'visibility_80_100_count',
    'visibility_unknown_count',
)
CATEGORY_TO_CLASS = {
    'vehicle.car': 'car', 'vehicle.truck': 'truck',
    'vehicle.construction': 'construction_vehicle',
    'vehicle.bus.bendy': 'bus', 'vehicle.bus.rigid': 'bus',
    'vehicle.trailer': 'trailer', 'vehicle.motorcycle': 'motorcycle',
    'vehicle.bicycle': 'bicycle', 'movable_object.barrier': 'barrier',
    'movable_object.trafficcone': 'traffic_cone',
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'human.pedestrian.police_officer': 'pedestrian',
}


def stream_json_array(path):
    """Read a top-level JSON array one object at a time."""
    decoder = json.JSONDecoder()
    with path.open('r') as handle:
        buffer, position, ended = '', 0, False
        while True:
            if position >= len(buffer) and not ended:
                buffer, position = handle.read(1024 * 1024), 0
                ended = not buffer
            if position >= len(buffer):
                raise ValueError('Unexpected end of JSON: {}'.format(path))
            char = buffer[position]
            if char.isspace() or char in '[,':
                position += 1
                continue
            if char == ']':
                return
            try:
                value, position = decoder.raw_decode(buffer, position)
            except json.JSONDecodeError:
                if ended:
                    raise
                buffer = buffer[position:] + handle.read(1024 * 1024)
                position = 0
                ended = handle.tell() == path.stat().st_size
                continue
            yield value
            if position > 1024 * 1024:
                buffer = buffer[position:]
                position = 0


def make_table(counts, split):
    lines = ['| Class | ' + ' | '.join(BINS) + ' | Total |',
             '|:--|' + '|'.join(['--:'] * (len(BINS) + 1)) + '|']
    for name in CLASSES + ('All classes',):
        values = [counts[split, name, label] for label in BINS]
        total = sum(values)
        cells = ['{:,} ({:.1f}%)'.format(value, 100 * value / total)
                 if total else '0 (—)' for value in values]
        lines.append('| {} | {} | {:,} |'.format(name.replace('_', ' '),
                     ' | '.join(cells), total))
    return '\n'.join(lines)


def main():
    script = Path(__file__).resolve()
    workspace = script.parents[4]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=workspace /
                        'RoboBEV/data/sets/nuscenes/v1.0-trainval')
    parser.add_argument('--val-scenes-json', type=Path, default=script.parent.parent /
                        'petr_localization_visualization/outputs/error_band_comparisons/metadata/selected_scenes.json')
    parser.add_argument('--output', type=Path, default=script.parent / 'GT_DISTRIBUTION.md')
    args = parser.parse_args()

    val_records = json.loads(args.val_scenes_json.read_text())
    val_names = {record['name'] for record in val_records}
    if len(val_names) != 150:
        raise ValueError('Expected 150 official validation scenes')
    scenes = {row['token']: row['name'] for row in
              stream_json_array(args.data_root / 'scene.json')}
    if set(scenes.values()) & val_names != val_names or len(scenes) != 850:
        raise ValueError('Validation scene list does not match 850-scene trainval table')
    samples = {row['token']: ('val' if scenes[row['scene_token']] in val_names
                              else 'train')
               for row in stream_json_array(args.data_root / 'sample.json')}
    print('Loaded {:,} samples'.format(len(samples)), flush=True)
    sample_pose_tokens = {}
    for row in stream_json_array(args.data_root / 'sample_data.json'):
        if row['sample_token'] in samples and row['is_key_frame'] and \
                'LIDAR_TOP' in row['filename']:
            sample_pose_tokens[row['sample_token']] = row['ego_pose_token']
    if set(sample_pose_tokens) != set(samples):
        raise ValueError('Missing or duplicated keyframe LIDAR_TOP pose')
    print('Found {:,} LIDAR_TOP keyframe poses'.format(len(sample_pose_tokens)), flush=True)
    required_pose_tokens = set(sample_pose_tokens.values())
    poses = {row['token']: row['translation'][:2]
             for row in stream_json_array(args.data_root / 'ego_pose.json')
             if row['token'] in required_pose_tokens}
    if set(poses) != required_pose_tokens:
        raise ValueError('Missing ego pose records')
    sample_xy = {token: poses[pose_token]
                 for token, pose_token in sample_pose_tokens.items()}
    print('Loaded {:,} ego positions'.format(len(sample_xy)), flush=True)

    counts = Counter()
    visibility_counts = Counter()
    ignored = Counter()
    category_names = {row['token']: row['name'] for row in
                      stream_json_array(args.data_root / 'category.json')}
    instance_classes = {
        row['token']: CATEGORY_TO_CLASS.get(category_names[row['category_token']])
        for row in stream_json_array(args.data_root / 'instance.json')}
    for row in stream_json_array(args.data_root / 'sample_annotation.json'):
        token = row['sample_token']
        name = instance_classes[row['instance_token']]
        if name is None:
            ignored[row['instance_token']] += 1
            continue
        split = samples[token]
        ego_x, ego_y = sample_xy[token]
        gt_x, gt_y = row['translation'][:2]
        distance = math.hypot(gt_x - ego_x, gt_y - ego_y)
        label = next((BINS[i] for i, edge in enumerate(EDGES)
                      if distance < edge), BINS[-1])
        counts[split, name, label] += 1
        counts[split, 'All classes', label] += 1
        visibility = row.get('visibility_token')
        if visibility not in VISIBILITY_TOKENS:
            visibility = 'unknown'
        visibility_counts[split, name, label, visibility] += 1
        visibility_counts[split, 'All classes', label, visibility] += 1
    print('Counted {:,} detection-class GT annotations'.format(
        sum(counts['train', 'All classes', label] +
            counts['val', 'All classes', label] for label in BINS)), flush=True)

    summary_rows = []
    for split in ('train', 'val'):
        for name in CLASSES + ('All classes',):
            for label in BINS:
                row = dict(zip(VISIBILITY_FIELDS, (
                    split, name, label, counts[split, name, label],
                    *(visibility_counts[split, name, label, token]
                      for token in VISIBILITY_TOKENS))))
                if sum(row[field] for field in VISIBILITY_FIELDS[4:]) != row['gt_count']:
                    raise ValueError('GT visibility counts differ from GT total')
                summary_rows.append(row)

    sample_counts = Counter(samples.values())
    lines = [
        '# nuScenes GT distribution by ego distance', '',
        'Counts cover all annotated keyframes in the nuScenes v1.0-trainval '
        'tables: **700 training scenes** ({:,} keyframes) and '
        '**150 validation scenes** ({:,} keyframes).'.format(
            sample_counts['train'], sample_counts['val']), '',
        'Each cell is **GT count (share of that class within the split)**. '
        'Distance is the horizontal Euclidean distance between the GT box '
        'center and the LIDAR_TOP ego pose at that sample. Bins are left '
        'inclusive and right exclusive; 50+ includes all remaining GT.', '',
        '## Training set', '', make_table(counts, 'train'), '',
        '## Validation set', '', make_table(counts, 'val'), '',
        '## GT visibility by distance', '',
        'The [visibility overview](visibility_plots/gt_visibility_bands_overview.png) '
        'and [CSV counts](GT_VISIBILITY_BY_DISTANCE.csv) show the four nuScenes '
        'visibility categories within every split × detection class × '
        'distance bin. Each cell in the plot shows a 100% stacked bar, '
        'the exact category counts, and percentages of its GT total. '
        'There are also larger per-class plots in `visibility_plots/`. '
        'Dark red means less visible; light blue-gray means more visible.', '',
        '## Counting rules', '',
        '- Only the ten nuScenes detection classes are shown. Pedestrian '
        'subtypes and rigid/bendy buses are merged using the official '
        'detection class mapping. Other general categories are excluded.',
        '- These are raw annotated GT counts. No detection evaluation '
        'class-range, point-count, or bike-rack filters were applied. '
        'Counts can therefore differ from evaluator-eligible GT or 4 m TP '
        'counts in the PETR plots.',
        '- A GT annotation at a different keyframe counts again. '
        'This is an annotation distribution, not a unique-instance count.',
        '- The validation scene names come from the official split saved by '
        '`compare_error_bands.py --all-val`; the remaining scenes in '
        '`v1.0-trainval` are training scenes.',
        '- Regenerate with `python3 '
        'experiments/nuscnes_exp/gt_distance_distribution/summarize_gt_distribution.py` '
        'from the PETR repository root.', '',
        'Official class mapping: '
        '[nuScenes detection README](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/detection/README.md).',
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text('\n'.join(lines) + '\n')
    summary_path = args.output.parent / 'GT_VISIBILITY_BY_DISTANCE.csv'
    with summary_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=VISIBILITY_FIELDS)
        writer.writeheader()
        writer.writerows(summary_rows)
    from plot_gt_visibility_bands import render_gt_visibility
    render_gt_visibility(summary_rows, args.output.parent / 'visibility_plots',
                         CLASSES + ('All classes',))
    print('Wrote {}'.format(args.output), flush=True)


if __name__ == '__main__':
    main()
