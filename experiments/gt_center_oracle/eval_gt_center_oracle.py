#!/usr/bin/env python3
"""Evaluate PETR with nearest learned references moved to GT centers.

This is an oracle diagnostic: validation GT centers are deliberately exposed
at inference. For every sample, a one-to-one Hungarian assignment selects the
learned reference points closest to valid GT gravity centers. Selected points
are replaced by normalized GT centers; all other references remain unchanged.
"""

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet.datasets import replace_ImageToTensor
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from scipy.optimize import linear_sum_assignment


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--data-root', default=None)
    parser.add_argument('--ann-file', default=None)
    parser.add_argument('--distance', choices=('3d', 'bev'), default='3d')
    parser.add_argument('--device', default='cuda:0')
    return parser.parse_args()


def import_plugin(cfg):
    if cfg.get('plugin', False):
        module = os.path.dirname(cfg.get('plugin_dir', '')).replace('/', '.')
        importlib.import_module(module)


def valid_gt_centers(dataset, index, pc_range):
    annotation = dataset.get_ann_info(index)
    centers = annotation['gt_bboxes_3d'].gravity_center.float()
    labels = torch.as_tensor(annotation['gt_labels_3d'], dtype=torch.long)
    bounds = centers.new_tensor(pc_range)
    valid = labels >= 0
    valid &= torch.isfinite(centers).all(dim=1)
    valid &= ((centers >= bounds[:3]) & (centers <= bounds[3:6])).all(dim=1)
    indices = torch.nonzero(valid).flatten()
    return centers[indices], labels[indices], indices


def build_override(reference_points, gt_centers, pc_range, distance_mode):
    """Return [1,Q,3] normalized references and an assignment audit."""
    references = reference_points.detach().float().cpu()
    bounds = references.new_tensor(pc_range)
    metric_references = references * (bounds[3:6] - bounds[:3]) + bounds[:3]
    override = references.clone()
    if gt_centers.numel() == 0:
        return override.unsqueeze(0), []

    dimensions = 2 if distance_mode == 'bev' else 3
    cost = torch.cdist(
        gt_centers[:, :dimensions].cpu(),
        metric_references[:, :dimensions])
    gt_rows, query_columns = linear_sum_assignment(cost.numpy())
    normalized_gt = (
        gt_centers.cpu() - bounds[:3]) / (bounds[3:6] - bounds[:3])
    audit = []
    for gt_row, query_index in zip(gt_rows.tolist(), query_columns.tolist()):
        original = metric_references[query_index]
        target = gt_centers[gt_row].cpu()
        override[query_index] = normalized_gt[gt_row]
        audit.append({
            'valid_gt_row': int(gt_row),
            'query_index': int(query_index),
            'original_reference_center_m': original.tolist(),
            'gt_center_m': target.tolist(),
            'assignment_cost_m': float(cost[gt_row, query_index]),
            'movement_3d_m': float(torch.norm(original - target)),
            'movement_bev_m': float(torch.norm(original[:2] - target[:2])),
        })
    return override.unsqueeze(0), audit


def main():
    args = parse_args()
    os.chdir(str(REPO_ROOT))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config.fromfile(args.config)
    import_plugin(cfg)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.data.test.test_mode = True
    if args.data_root is not None:
        cfg.data.test.data_root = args.data_root
        if 'data_root' in cfg:
            cfg.data_root = args.data_root
    if args.ann_file is not None:
        cfg.data.test.ann_file = args.ann_file
        if 'test_ann_file' in cfg:
            cfg.test_ann_file = args.ann_file
    configured_samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
    if configured_samples_per_gpu > 1:
        cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset, samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False, shuffle=False)

    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    if cfg.get('fp16') is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.CLASSES = checkpoint.get('meta', {}).get('CLASSES', dataset.CLASSES)
    device_index = int(args.device.split(':', 1)[1]) if ':' in args.device else 0
    torch.cuda.set_device(device_index)
    model = MMDataParallel(model.cuda(device_index), device_ids=[device_index])
    model.eval()

    head = model.module.pts_bbox_head
    learned_references = head.reference_points.weight
    pc_range = list(head.pc_range)
    outputs, assignments = [], []
    progress = mmcv.ProgressBar(len(dataset))

    for index, data in enumerate(data_loader):
        gt_centers, gt_labels, original_gt_indices = valid_gt_centers(
            dataset, index, pc_range)
        override, audit = build_override(
            learned_references, gt_centers, pc_range, args.distance)
        override = override.to(
            device=learned_references.device,
            dtype=learned_references.dtype)
        with torch.no_grad():
            result = model(
                return_loss=False, rescale=True,
                reference_points_override=override, **data)
        outputs.extend(result)

        for row in audit:
            valid_row = row['valid_gt_row']
            row['annotation_gt_index'] = int(original_gt_indices[valid_row])
            row['gt_label'] = int(gt_labels[valid_row])
            row['gt_class'] = model.module.CLASSES[row['gt_label']]
        assignments.append({
            'dataset_index': index,
            'sample_token': dataset.data_infos[index]['token'],
            'num_valid_gt': int(len(gt_centers)),
            'assignments': audit,
        })
        for _ in result:
            progress.update()

    predictions_path = output_dir / 'predictions.pkl'
    mmcv.dump(outputs, str(predictions_path))
    with (output_dir / 'reference_assignments.json').open('w') as handle:
        json.dump({
            'oracle': True,
            'config': args.config,
            'checkpoint': args.checkpoint,
            'distance': args.distance,
            'point_cloud_range': pc_range,
            'num_queries': int(head.num_query),
            'samples': assignments,
        }, handle, indent=2)

    metrics = dataset.evaluate(
        outputs, metric='bbox',
        jsonfile_prefix=str(output_dir / 'formatted'))
    mmcv.dump(metrics, str(output_dir / 'metrics.json'))
    print(json.dumps({key: float(value) for key, value in metrics.items()},
                     indent=2))
    print('Predictions:', predictions_path)
    print('Assignments:', output_dir / 'reference_assignments.json')


if __name__ == '__main__':
    main()
