#!/usr/bin/env python3
"""Build an oracle config and verify that only its adapter is trainable."""

import argparse
import importlib
import os

from mmcv import Config, DictAction
from mmdet3d.models import build_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('--options', nargs='+', action=DictAction)
    return parser.parse_args()


def main():
    args = parse_args()
    config = Config.fromfile(args.config)
    if args.options:
        config.merge_from_dict(args.options)
    if config.get('plugin', False):
        plugin_dir = config.get('plugin_dir', 'projects/mmdet3d_plugin/')
        module_path = os.path.dirname(plugin_dir).replace('/', '.')
        importlib.import_module(module_path)

    model = build_model(
        config.model,
        train_cfg=config.get('train_cfg'),
        test_cfg=config.get('test_cfg'))
    trainable = [
        (name, parameter.numel()) for name, parameter in model.named_parameters()
        if parameter.requires_grad]
    if not trainable:
        raise RuntimeError('oracle config has no trainable parameters')
    unexpected = [
        name for name, _ in trainable
        if not name.startswith('pts_bbox_head.oracle_adapter.')]
    if unexpected:
        raise RuntimeError(
            'non-adapter parameters are trainable: {}'.format(unexpected))

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable_total = sum(count for _, count in trainable)
    print('Oracle adapter sanity check passed')
    print('Trainable parameters: {:,} / {:,} ({:.4f}%)'.format(
        trainable_total, total, 100.0 * trainable_total / total))
    for name, count in trainable:
        print('  {}: {:,}'.format(name, count))


if __name__ == '__main__':
    main()
