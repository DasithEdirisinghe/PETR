"""Compare V2 rule and PETR assignments on the same pretrained train batches.

This makes no optimizer step and does not change a checkpoint. Run from the
PETR root in its CUDA environment, for example:

    python experiments/nuscnes_exp/pedestrian_neighbor_trace/audit_rule_v2_assignments.py \
        --config projects/configs/petr/petr_r50dcn_gridmask_p4_ray_rule_decoder_heads_only_v6.py \
        --max-batches 20 --output-dir results/rule_v2_assignment_audit
"""

import argparse
import importlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet.datasets import build_dataloader
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--max-batches', type=int, default=20)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--max-examples', type=int, default=500)
    args = parser.parse_args()
    if args.max_batches < 1:
        raise ValueError('max-batches must be positive')
    cfg = Config.fromfile(args.config)
    if cfg.get('custom_imports'):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg.custom_imports)
    if cfg.get('plugin', False):
        importlib.import_module('projects.mmdet3d_plugin')
    if cfg.model.pts_bbox_head.get('rule_v2_cfg') is None:
        raise ValueError('Config must enable rule_v2_cfg')
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
    # The head computes both rule and original assignments independently,
    # regardless of whether V6 or V7 is passed as the audit config.
    totals, examples, batches = {}, [], 0
    for data in loader:
        if batches >= args.max_batches:
            break
        with torch.no_grad():
            output = model(return_loss=True, **data)
        for name, value in output.items():
            if name.startswith('rule_v2_diag_'):
                key = name[len('rule_v2_diag_'):]
                totals[key] = totals.get(key, 0.0) + float(value.detach())
        remaining = args.max_examples - len(examples)
        if remaining > 0:
            examples.extend(model.module.pts_bbox_head.rule_v2_last_examples[:remaining])
        batches += 1
        print('audited batches: {}/{}'.format(batches, args.max_batches), flush=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = dict(
        checkpoint=cfg.load_from, config=args.config,
        batches=batches, frames=batches * cfg.data.samples_per_gpu,
        decoder_layer_aggregated_totals=totals,
        changed_fraction_of_gt=totals.get('changed', 0.0) /
            max(1.0, totals.get('gt', 0.0)),
        changed_mean_old_center_m=totals.get('changed_old_center_sum', 0.0) /
            max(1.0, totals.get('changed', 0.0)),
        changed_mean_new_center_m=totals.get('changed_new_center_sum', 0.0) /
            max(1.0, totals.get('changed', 0.0)))
    (output_dir / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    with (output_dir / 'switched_pairs.jsonl').open('w') as file:
        for example in examples:
            file.write(json.dumps(example) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
