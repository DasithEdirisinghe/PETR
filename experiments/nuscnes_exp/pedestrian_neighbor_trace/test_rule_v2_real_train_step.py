"""Run one real train step for V6 or V7 without writing a checkpoint."""

import argparse
import importlib
import random

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import build_optimizer, load_checkpoint
from mmdet.datasets import build_dataloader
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA GPU is required')

    cfg = Config.fromfile(args.config)
    if cfg.get('custom_imports'):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg.custom_imports)
    if cfg.get('plugin', False):
        importlib.import_module('projects.mmdet3d_plugin')
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    dataset = build_dataset(cfg.data.train)
    loader = build_dataloader(
        dataset, samples_per_gpu=cfg.data.samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        num_gpus=1, dist=False, shuffle=False, seed=0)
    model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'),
                        test_cfg=cfg.get('test_cfg'))
    model.init_weights()
    load_checkpoint(model, cfg.load_from, map_location='cpu')
    model = MMDataParallel(model.cuda(), device_ids=[0])
    model.train()
    optimizer = build_optimizer(model, cfg.optimizer)
    optimizer.zero_grad()

    batch = next(iter(loader))
    outputs = model(return_loss=True, **batch)
    losses = {name: value.mean() for name, value in outputs.items()
              if name.startswith('loss') or '.loss' in name}
    if not losses:
        raise RuntimeError('Model produced no losses')
    for name, value in losses.items():
        if not torch.isfinite(value):
            raise RuntimeError('{} is nonfinite: {}'.format(name, value))
    total = sum(losses.values())
    total.backward()

    gradients = [(name, parameter.grad) for name, parameter in
                 model.module.named_parameters() if parameter.requires_grad]
    nonzero = [(name, grad) for name, grad in gradients
               if grad is not None and torch.isfinite(grad).all() and
               grad.abs().sum().item() > 0]
    bad = [name for name, grad in gradients if grad is not None and
           not torch.isfinite(grad).all()]
    if bad or not nonzero:
        raise RuntimeError('Invalid gradients: nonfinite={}, nonzero={}'.format(
            bad[:10], len(nonzero)))
    optimizer.step()

    diagnostics = {name: float(value.detach()) for name, value in outputs.items()
                   if name.startswith('rule_v2_diag_')}
    print('PASS: {} real train batch, forward, backward, optimizer step'.format(
        cfg.model.pts_bbox_head.rule_v2_cfg.matching), flush=True)
    print('losses:', {name: float(value.detach()) for name, value in losses.items()},
          flush=True)
    print('trainable parameters with nonzero finite gradients:', len(nonzero),
          flush=True)
    print('diagnostics:', diagnostics, flush=True)


if __name__ == '__main__':
    main()
