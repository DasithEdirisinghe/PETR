"""Direct one-to-one 3D-IoU matching for selected-camera gallery boxes."""

import math
from types import SimpleNamespace

from analyze import overlaps


def minimum_cost_assignment(costs):
    """Assign every row to a distinct column, with rows <= columns."""
    n = len(costs)
    if not n:
        return []
    m = len(costs[0])
    if n > m or any(len(row) != m for row in costs):
        raise ValueError('Assignment matrix must be rectangular with rows <= columns')
    u, v = [0.0] * (n + 1), [0.0] * (m + 1)
    owner, previous = [0] * (m + 1), [0] * (m + 1)
    for row in range(1, n + 1):
        owner[0] = row
        column = 0
        best = [float('inf')] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[column] = True
            active = owner[column]
            delta, next_column = float('inf'), 0
            for candidate in range(1, m + 1):
                if used[candidate]:
                    continue
                value = costs[active - 1][candidate - 1] - u[active] - v[candidate]
                if value < best[candidate]:
                    best[candidate], previous[candidate] = value, column
                if best[candidate] < delta:
                    delta, next_column = best[candidate], candidate
            for candidate in range(m + 1):
                if used[candidate]:
                    u[owner[candidate]] += delta
                    v[candidate] -= delta
                else:
                    best[candidate] -= delta
            column = next_column
            if owner[column] == 0:
                break
        while True:
            candidate = previous[column]
            owner[column] = owner[candidate]
            column = candidate
            if column == 0:
                break
    assignment = [None] * n
    for column in range(1, m + 1):
        if owner[column]:
            assignment[owner[column] - 1] = column - 1
    return assignment


def oracle_orders(ground_truth, predictions, thresholds=(0.5, 1.0, 2.0, 4.0)):
    """Return prediction orders and direct assignments for every threshold."""
    baseline = [prediction['id'] for prediction in predictions]
    if not ground_truth or not predictions:
        return ({str(threshold): baseline for threshold in thresholds},
                {str(threshold): {} for threshold in thresholds})
    gt_boxes = [SimpleNamespace(translation=box['center'], size=box['size'],
                                rotation=box['rotation']) for box in ground_truth]
    pred_boxes = [SimpleNamespace(translation=box['center'], size=box['size'],
                                  rotation=box['rotation']) for box in predictions]
    pairs = {}
    for pi, prediction in enumerate(predictions):
        for gi, gt in enumerate(ground_truth):
            distance = math.hypot(prediction['center'][0] - gt['center'][0],
                                  prediction['center'][1] - gt['center'][1])
            if distance < max(thresholds):
                pairs[pi, gi] = (distance, overlaps(pred_boxes[pi], gt_boxes[gi])[1])
    result, assignments = {}, {}
    for threshold in thresholds:
        candidates = [pi for pi in range(len(predictions))
                      if any((pi, gi) in pairs and pairs[pi, gi][0] < threshold
                             for gi in range(len(ground_truth)))]
        selected = {}
        used_gt, used_pred = set(), set()
        positive = [pi for pi in candidates if any(
            pairs.get((pi, gi), (0, 0))[1] > 1e-9
            for gi in range(len(ground_truth)))]
        if positive:
            n_gt = len(ground_truth)
            penalty = n_gt + 1.0
            costs = [[1e9] * len(positive) + [penalty] * n_gt
                     for _ in range(n_gt)]
            for column, pi in enumerate(positive):
                for gi in range(n_gt):
                    pair = pairs.get((pi, gi))
                    if pair is not None and pair[0] < threshold and pair[1] > 1e-9:
                        costs[gi][column] = 1.0 - pair[1]
            for gi, column in enumerate(minimum_cost_assignment(costs)):
                if column < len(positive) and costs[gi][column] < penalty:
                    pi = positive[column]
                    distance, iou = pairs[pi, gi]
                    selected[pi] = (gi, distance, iou)
                    used_gt.add(gi); used_pred.add(pi)
        remaining_gt = [gi for gi in range(len(ground_truth)) if gi not in used_gt]
        remaining_pred = [pi for pi in candidates if pi not in used_pred and any(
            (pi, gi) in pairs and pairs[pi, gi][0] < threshold and
            pairs[pi, gi][1] <= 1e-9 for gi in remaining_gt)]
        if remaining_gt and remaining_pred:
            penalty = len(remaining_gt) * threshold + 1.0
            costs = [[1e9] * len(remaining_pred) + [penalty] * len(remaining_gt)
                     for _ in remaining_gt]
            for row, gi in enumerate(remaining_gt):
                for col, pi in enumerate(remaining_pred):
                    pair = pairs.get((pi, gi))
                    if pair is not None and pair[0] < threshold and pair[1] <= 1e-9:
                        costs[row][col] = pair[0]
            for row, col in enumerate(minimum_cost_assignment(costs)):
                if col < len(remaining_pred) and costs[row][col] < penalty:
                    pi, gi = remaining_pred[col], remaining_gt[row]
                    selected[pi] = (gi, pairs[pi, gi][0], 0.0)
        order = sorted(selected, key=lambda pi: (selected[pi][2] <= 1e-9,
                                                 -selected[pi][2], selected[pi][1], pi))
        order.extend(pi for pi in range(len(predictions)) if pi not in selected)
        result[str(threshold)] = [predictions[pi]['id'] for pi in order]
        assignments[str(threshold)] = {str(predictions[pi]['id']): match[0]
                                       for pi, match in selected.items()}
    return result, assignments
