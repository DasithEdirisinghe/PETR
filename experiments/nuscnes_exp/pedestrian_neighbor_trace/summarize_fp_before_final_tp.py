#!/usr/bin/env python3
"""Summarize already-projected FPs ranked before the final official TP."""

import argparse
import bisect
import csv
import json
import statistics
from collections import Counter
from pathlib import Path


CATEGORIES = ('interested_class_gt', 'other_class_gt', 'background')
STAGES = (
    ('0-25% of matched TPs', 0.0, 0.25),
    ('25-50% of matched TPs', 0.25, 0.50),
    ('50-75% of matched TPs', 0.50, 0.75),
    ('75-90% of matched TPs', 0.75, 0.90),
    ('90-100% of matched TPs', 0.90, 1.01),
)


def mean(values):
    values = [value for value in values if value is not None]
    return sum(values)/len(values) if values else None


def median(values):
    values = [value for value in values if value is not None]
    return statistics.median(values) if values else None


def number(row, field):
    value = row.get(field, '')
    return float(value) if value not in ('', None) else None


def summarize(rows, label):
    output = []
    for category in CATEGORIES:
        selected = [row for row in rows
                    if row['projection_category'] == category]
        center = [number(row, 'corresponding_gt_center_error_m')
                  for row in selected]
        output.append({
            'scope': label,
            'projection_category': category,
            'count': len(selected),
            'percent': 100*len(selected)/len(rows) if rows else 0,
            'mean_confidence': mean([number(row, 'score') for row in selected]),
            'median_confidence': median([
                number(row, 'score') for row in selected]),
            'mean_projected_iou2d': mean([
                number(row, 'max_projected_iou2d') for row in selected]),
            'mean_corresponding_gt_center_error_m': mean(center),
            'median_corresponding_gt_center_error_m': median(center),
            'mean_corresponding_gt_absolute_radial_error_m': mean([
                number(row, 'corresponding_gt_absolute_radial_error_m')
                for row in selected]),
            'mean_corresponding_gt_absolute_transverse_error_m': mean([
                number(row, 'corresponding_gt_absolute_transverse_error_m')
                for row in selected]),
            'corresponding_gt_within_4m_count': sum(
                value is not None and value < 4.0 for value in center),
            'corresponding_gt_within_4m_percent':
                100*sum(value is not None and value < 4.0 for value in center) /
                len(selected) if selected else 0,
        })
    return output


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--all-gt-csv', type=Path, required=True)
    parser.add_argument('--fp-records-csv', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    for path in (args.all_gt_csv, args.fp_records_csv):
        if not path.is_file():
            parser.error('Missing input: {}'.format(path))

    tp_ids = []
    gt_count = 0
    with args.all_gt_csv.open(newline='') as handle:
        for row in csv.DictReader(handle):
            gt_count += 1
            if row['official_match'] == '1':
                tp_ids.append(int(row['official_tp_prediction_id']))
    if not tp_ids:
        parser.error('No official TP prediction IDs found')
    tp_ids.sort()
    if len(tp_ids) != len(set(tp_ids)):
        raise ValueError('Official TP prediction IDs are not unique')
    last_tp_id = tp_ids[-1]

    before_final = []
    total_remaining_fp = 0
    with args.fp_records_csv.open(newline='') as handle:
        for row in csv.DictReader(handle):
            total_remaining_fp += 1
            prediction_id = int(row['prediction_id'])
            if prediction_id >= last_tp_id:
                continue
            tp_before = bisect.bisect_left(tp_ids, prediction_id)
            progress = tp_before/len(tp_ids)
            row['tp_count_before_prediction'] = tp_before
            row['fraction_of_matched_tps_before_prediction'] = progress
            row['absolute_recall_before_prediction'] = \
                tp_before/gt_count if gt_count else 0
            before_final.append(row)

    summary_rows = summarize(before_final, 'before_final_tp')
    stage_rows = []
    for label, lower, upper in STAGES:
        selected = [row for row in before_final
                    if lower <= float(
                        row['fraction_of_matched_tps_before_prediction']) < upper]
        stage_rows.extend(summarize(selected, label))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir/'before_final_tp_records.csv', before_final)
    write_csv(args.output_dir/'before_final_tp_summary.csv', summary_rows)
    write_csv(args.output_dir/'recall_stage_summary.csv', stage_rows)
    payload = {
        'gt_count': gt_count,
        'official_tp_count': len(tp_ids),
        'last_tp_prediction_id_zero_based': last_tp_id,
        'last_tp_rank_one_based': last_tp_id+1,
        'total_remaining_fp_count': total_remaining_fp,
        'remaining_fp_before_final_tp_count': len(before_final),
        'remaining_fp_after_final_tp_count':
            total_remaining_fp-len(before_final),
        'category_counts_before_final_tp': dict(Counter(
            row['projection_category'] for row in before_final)),
        'summary': summary_rows,
        'recall_stage_summary': stage_rows,
    }
    (args.output_dir/'before_final_tp_summary.json').write_text(
        json.dumps(payload, indent=2)+'\n')
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == '__main__':
    main()
