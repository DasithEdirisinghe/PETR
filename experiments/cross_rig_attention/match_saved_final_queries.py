"""Uniquely associate saved PCCR final boxes with traced PETR query pairs.

This is a read-only identity check, not a new detector matching/evaluation rule.
"""

import math


def match_saved_queries(candidates, saved_boxes, score_tolerance=0.02,
                        center_tolerance_m=0.05, return_diagnostics=False):
    """Return one query ID per saved box, or fail on missing/ambiguous identity.

    Candidate and saved records require `class_name`, `score`, and `center_xy`;
    candidates also require `query_index`. Order is deliberately irrelevant.
    """
    by_class = {}
    for index, item in enumerate(candidates):
        by_class.setdefault(item['class_name'], []).append(index)
    def deltas(position, index):
        saved, candidate = saved_boxes[position], candidates[index]
        return (abs(candidate['score']-saved['score']),
                math.hypot(candidate['center_xy'][0]-saved['center_xy'][0],
                           candidate['center_xy'][1]-saved['center_xy'][1]))

    unused = set(range(len(candidates)))
    matched = [None]*len(saved_boxes)
    # Match *all* virtually identical outputs before allowing rerun drift.
    # Otherwise an earlier loosely matching saved box can consume another
    # saved box's exact query. A genuine exact duplicate remains ambiguous.
    for score_limit, center_limit in ((1e-6, 1e-4),
                                      (score_tolerance, center_tolerance_m)):
        while True:
            resolved = []
            for position, saved in enumerate(saved_boxes):
                if matched[position] is not None:
                    continue
                possible = [index for index in by_class.get(saved['class_name'], ())
                            if index in unused and
                            deltas(position, index)[0] <= score_limit and
                            deltas(position, index)[1] <= center_limit]
                if len(possible) == 1:
                    resolved.append((position, possible[0]))
                elif len(possible) > 1 and score_limit == 1e-6:
                    raise ValueError(
                        'Saved final prediction {} has {} unique traced candidates '
                        'even at exact tolerance'.format(position, len(possible)))
            if not resolved:
                break
            # Two saved boxes claiming the same query cannot be resolved by
            # greedy list order; leave them for the ambiguity report below.
            claims = {}
            for position, index in resolved:
                claims.setdefault(index, []).append(position)
            unique = [(positions[0], index) for index, positions in claims.items()
                      if len(positions) == 1]
            if not unique:
                break
            for position, index in unique:
                matched[position] = index
                unused.remove(index)

    for position, saved in enumerate(saved_boxes):
        if matched[position] is not None:
            continue
        possible = [(index, deltas(position, index))
                    for index in by_class.get(saved['class_name'], ())
                    if index in unused]
        possible = [(index, delta) for index, delta in possible
                    if delta[0] <= score_tolerance and
                    delta[1] <= center_tolerance_m]
        raise ValueError(
            'Saved final prediction {} ({}, score {:.6f}, xy {}) has {} '
            'unique traced candidates within tolerance: {}'.format(
                position, saved['class_name'], saved['score'],
                saved['center_xy'], len(possible),
                [(candidates[index]['query_index'], delta) for index, delta in possible[:8]]))
    queries = [candidates[index]['query_index'] for index in matched]
    diagnostics = [dict(zip(('score_delta', 'center_delta_m'),
                            deltas(position, index)))
                   for position, index in enumerate(matched)]
    return (queries, diagnostics) if return_diagnostics else queries
