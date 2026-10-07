"""Pure NumPy geometry and weighted/rank metrics for attention-region audits."""

import math

import numpy as np


REGIONS = ('car', 'other_object', 'no_annotated_gt')


def box_entry_distance(origins, directions, box, expansion):
    """First positive ray intersection with an expanded oriented LiDAR box.

    Returns infinity for a ray that misses. `box` uses gravity-centre xyz,
    width/length/height and yaw, as returned by PETR GT preprocessing.
    """
    box = np.asarray(box, dtype=np.float64)
    center = box[:3]
    half = np.maximum(box[3:6]*0.5+float(expansion), 1e-4)
    angle = float(box[6])
    cosine, sine = math.cos(angle), math.sin(angle)
    rotation = np.asarray([[cosine, sine, 0.0],
                           [-sine, cosine, 0.0], [0.0, 0.0, 1.0]])
    local_origins = (origins-center[None]).dot(rotation.T)
    local_directions = directions.dot(rotation.T)
    t_near = np.full(len(origins), -np.inf)
    t_far = np.full(len(origins), np.inf)
    hit = np.ones(len(origins), dtype=np.bool_)
    for axis in range(3):
        o, d = local_origins[:, axis], local_directions[:, axis]
        parallel = np.abs(d) < 1e-10
        hit &= ~(parallel & (np.abs(o) > half[axis]))
        safe_d = np.where(parallel, 1.0, d)
        a, b = (-half[axis]-o)/safe_d, (half[axis]-o)/safe_d
        near, far = np.minimum(a, b), np.maximum(a, b)
        near[parallel], far[parallel] = -np.inf, np.inf
        t_near, t_far = np.maximum(t_near, near), np.minimum(t_far, far)
    entry = np.maximum(t_near, 0.0)
    return np.where(hit & (t_far >= entry), entry, np.inf)


def region_labels(origins, directions, boxes, names, expansion):
    """Partition feature rays by their nearest expanded annotated box.

    Label 0=car, 1=other annotated object, 2=no annotated GT. Multiple
    intersected boxes are counted but assigned to the nearest entry depth.
    This is a geometric partition, not an occlusion/segmentation oracle.
    """
    if len(boxes) != len(names):
        raise ValueError('GT box/name count mismatch')
    closest = np.full(len(origins), np.inf)
    labels = np.full(len(origins), 2, dtype=np.int8)
    hit_count = np.zeros(len(origins), dtype=np.int16)
    for box, name in zip(boxes, names):
        distance = box_entry_distance(origins, directions, box, expansion)
        hit_count += np.isfinite(distance)
        nearer = distance < closest
        labels[nearer] = 0 if name == 'car' else 1
        closest[nearer] = distance[nearer]
    return labels, hit_count > 1


def average_ranks(values):
    """One-based average ranks with exact ties, for Mann-Whitney AUROC."""
    _, inverse, counts = np.unique(values, return_inverse=True,
                                   return_counts=True)
    midranks = np.cumsum(counts)-(counts-1)/2.0
    return midranks[inverse]


def score_regions(weights, valid, labels):
    """One-vs-rest AUROC plus mass and cell-area interpretation checks."""
    valid = np.asarray(valid, dtype=np.bool_)
    weights = np.asarray(weights, dtype=np.float64)[valid]
    labels = np.asarray(labels)[valid]
    if not len(weights) or len(weights) != len(labels) or not np.isfinite(weights).all():
        raise ValueError('Invalid attention/region arrays')
    total = weights.sum()
    if total <= 0:
        raise ValueError('No positive attention mass')
    weights = weights/total
    ranks = average_ranks(weights)
    result = {}
    for code, name in enumerate(REGIONS):
        mask = labels == code
        n_pos, n_neg = int(mask.sum()), int((~mask).sum())
        fraction = n_pos/len(weights)
        mass = float(weights[mask].sum())
        if n_pos and n_neg:
            auc = ((ranks[mask].sum()-n_pos*(n_pos+1)/2.0) /
                   (n_pos*n_neg))
        else:
            auc = None
        result[name] = {'cell_count': n_pos, 'cell_fraction': fraction,
                        'attention_mass': mass,
                        'attention_auroc': float(auc) if auc is not None else None}
    if abs(sum(x['attention_mass'] for x in result.values())-1.0) > 1e-6:
        raise ValueError('Attention region masses do not partition to one')
    return result
