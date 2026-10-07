"""Resolve tracer GT indices in old and new paired-selection formats."""

from pathlib import Path

import numpy as np

from compare_r1_r1f_cars import gt_frames

ROOT = Path(__file__).resolve().parents[2]


def ensure_source_tracer_indices(pairs, split):
    """Return copies with R1-f filtered GT indices, validating car identity."""
    if all(pair.get('source_tracer_gt_index') is not None for pair in pairs):
        return pairs
    frames = gt_frames(ROOT / 'data/pccr/R1-f/R1-f_infos_{}.pkl'.format(split))
    resolved = []
    for pair in pairs:
        row = dict(pair)
        token = row['source_sample_token']
        frame = frames[token]
        raw_gt = int(row['source_gt_index'])
        matches = np.flatnonzero(frame['gt_indices'] == raw_gt)
        if len(matches) != 1:
            raise ValueError('R1-f raw GT index {} not found exactly once in {}'.format(
                raw_gt, token))
        local_index = int(matches[0])
        tracer_index = frame['tracer_indices'][local_index]
        if tracer_index is None:
            raise ValueError('R1-f car was filtered from PETR GT: {}'.format(token))
        world_center = frame['centers'][local_index]
        paired_distance = float(np.linalg.norm(
            world_center - np.asarray([row['gt_x'], row['gt_y']])))
        if paired_distance > .25:
            raise ValueError('Paired car center mismatch ({:.3f} m): {}'.format(
                paired_distance, token))
        if row.get('source_tracer_gt_index') is not None and int(
                row['source_tracer_gt_index']) != int(tracer_index):
            raise ValueError('Stale source tracer index for {}'.format(token))
        row['source_tracer_gt_index'] = int(tracer_index)
        resolved.append(row)
    return resolved
