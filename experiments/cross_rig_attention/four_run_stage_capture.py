#!/usr/bin/env python3
"""Capture upstream PETR tensors for one rig/checkpoint/frame without GT input."""

import argparse
import importlib
import os
import sys
from pathlib import Path

import numpy as np
import torch
from mmcv import Config
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.query_trace.trace_query import (  # noqa: E402
    first_augmentation, prepare_one, relocate_dataset_paths,
    position_components)

CONFIG = ROOT / 'projects/configs/petr/petr_r50dcn_gridmask_p4_800x320_pccr.py'
CHECKPOINTS = {
    'r1': ROOT / 'results/petr_r50dcn_gridmask_p4_800x320_pccr/R1/latest.pth',
    'r1f': ROOT / 'results/petr_r50dcn_gridmask_p4_800x320_pccr/R1-f/latest.pth',
}


def tensor_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().half()
    if isinstance(value, (list, tuple)):
        return [tensor_tree(item) for item in value]
    if isinstance(value, dict):
        return {key: tensor_tree(item) for key, item in value.items()}
    raise TypeError('Unexpected captured output: {}'.format(type(value)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rig', choices=['R1', 'R1-f'], required=True)
    parser.add_argument('--model', choices=['r1', 'r1f'], required=True)
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--sample-token', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    os.chdir(str(ROOT))
    cfg = Config.fromfile(str(CONFIG))
    if cfg.get('plugin', False):
        importlib.import_module(os.path.dirname(
            cfg.get('plugin_dir', '')).replace('/', '.'))
    cfg.model.pretrained = None
    definition = cfg.data.test.copy()
    definition.data_root = 'data/pccr/{}/'.format(args.rig)
    definition.ann_file = 'data/pccr/{0}/{0}_infos_{1}.pkl'.format(
        args.rig, args.split)
    definition.test_mode = True
    dataset = build_dataset(definition)
    relocate_dataset_paths(dataset, definition.data_root)
    indices = {info['token']: index for index, info in enumerate(dataset.data_infos)}
    index = indices[args.sample_token]
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    if cfg.get('fp16'):
        wrap_fp16_model(model)
    load_checkpoint(model, str(CHECKPOINTS[args.model]), map_location='cpu')
    device = torch.device(args.device)
    model.to(device).eval()
    data = prepare_one(dataset, index, device.index if device.type == 'cuda'
                       else device)
    images, metas = first_augmentation(data)
    captured = {}
    handles = []
    for name, module in [('backbone', model.img_backbone),
                         ('fpn', model.img_neck),
                         ('input_proj', model.pts_bbox_head.input_proj)]:
        def save_output(_module, _inputs, output, key=name):
            captured[key] = tensor_tree(output)
        handles.append(module.register_forward_hook(save_output))
    try:
        with torch.no_grad():
            features = model.extract_feat(img=images, img_metas=metas)
            head = model.pts_bbox_head
            g3d, gmv = position_components(head, features, metas)
            outputs = head(features, metas)
    finally:
        for handle in handles:
            handle.remove()
    # The model forward above is ordinary inference. No GT has been passed in.
    result = {
        'rig': args.rig, 'model': args.model,
        'sample_token': args.sample_token,
        'backbone': captured['backbone'],
        'fpn': captured['fpn'],
        'input_proj': captured['input_proj'],
        'g3d': tensor_tree(g3d),
        'gmv': tensor_tree(gmv),
        'reference_points_normalized': tensor_tree(head.reference_points.weight),
        'all_cls_scores': tensor_tree(outputs['all_cls_scores']),
        'all_bbox_preds': tensor_tree(outputs['all_bbox_preds']),
        'pad_shape': [list(shape) for shape in metas[0]['pad_shape']],
        'lidar2img': np.asarray(metas[0]['lidar2img']).tolist(),
        'num_cameras': len(metas[0]['lidar2img']),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, str(args.output))
    print('Saved stage tensors:', args.output)


if __name__ == '__main__':
    main()
