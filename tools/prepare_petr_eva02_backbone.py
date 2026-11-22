#!/usr/bin/env python3
"""Extract compatible EVA02-L backbone tensors for PETR initialization.

The source is the StreamPETR EVA02-L 2D pretraining checkpoint. The output
contains only img_backbone keys, so PETR's detection head starts from scratch.
"""

import argparse
import json
from pathlib import Path

import torch
from mmcv import Config
from mmdet.models.builder import build_backbone

import projects.mmdet3d_plugin  # noqa: F401 - registers EVAViT


PREFIXES = ('module.img_backbone.', 'img_backbone.',
            'module.backbone.', 'backbone.', 'module.', '')


def stripped_key(key, target):
    # The published 2D checkpoint stores its feature pyramid as backbone.simfp_*.
    if key.startswith('backbone.simfp_'):
        candidate = 'adapter.' + key[len('backbone.'):]
        if candidate in target:
            return candidate
    for prefix in PREFIXES:
        if key.startswith(prefix):
            candidate = key[len(prefix):]
            if candidate in target:
                return candidate
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    for path in (args.config, args.source):
        if not path.is_file():
            parser.error('Missing input: {}'.format(path))

    cfg = Config.fromfile(str(args.config))
    if cfg.model.img_backbone.type != 'EVAViT' or cfg.model.img_backbone.flash_attn:
        parser.error('Expected no-flash EVAViT backbone in config')
    backbone = build_backbone(cfg.model.img_backbone)
    target = backbone.state_dict()
    source_checkpoint = torch.load(str(args.source), map_location='cpu')
    if isinstance(source_checkpoint, dict):
        source = source_checkpoint.get('state_dict',
                 source_checkpoint.get('model', source_checkpoint))
    else:
        source = source_checkpoint
    if not isinstance(source, dict):
        raise TypeError('Checkpoint has no tensor state dictionary')

    matched = {}
    shape_mismatches = []
    for key, value in source.items():
        if not torch.is_tensor(value):
            continue
        name = stripped_key(key, target)
        if name is None:
            continue
        if tuple(value.shape) == tuple(target[name].shape):
            matched['img_backbone.' + name] = value
        else:
            shape_mismatches.append((key, tuple(value.shape),
                                     tuple(target[name].shape)))

    core = {name: tensor for name, tensor in backbone.named_parameters()
            if name.startswith(('patch_embed.', 'blocks.')) or name == 'pos_embed'}
    matched_core = sum(tensor.numel() for name, tensor in core.items()
                       if 'img_backbone.' + name in matched)
    total_core = sum(tensor.numel() for tensor in core.values())
    coverage = matched_core / total_core if total_core else 0.0
    print('EVA core parameter coverage: {:.2%} ({:,}/{:,} values)'.format(
        coverage, matched_core, total_core), flush=True)
    print('Matching backbone tensors: {} of {}'.format(len(matched), len(target)),
          flush=True)
    if shape_mismatches:
        print('Shape mismatches (first 10): {}'.format(shape_mismatches[:10]),
              flush=True)
    if coverage < 0.95:
        missing = [name for name in core if 'img_backbone.' + name not in matched]
        raise RuntimeError('Less than 95% of EVA core parameters match; first '
                           'missing keys: {}'.format(missing[:15]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + '.tmp')
    torch.save({'state_dict': matched,
                'meta': {'source': str(args.source.resolve()),
                         'config': str(args.config.resolve()),
                         'eva_core_coverage': coverage}}, str(temporary))
    temporary.replace(args.output)
    report = args.output.with_suffix('.json')
    report.write_text(json.dumps({
        'source': str(args.source.resolve()),
        'output': str(args.output.resolve()),
        'core_parameter_coverage': coverage,
        'matched_tensors': len(matched),
        'target_tensors': len(target),
        'shape_mismatches': [str(item) for item in shape_mismatches[:20]],
        'head_initialized_from_pretraining': False,
    }, indent=2) + '\n')
    print('Wrote {} and {}'.format(args.output, report), flush=True)


if __name__ == '__main__':
    main()
