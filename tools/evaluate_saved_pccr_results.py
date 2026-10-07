#!/usr/bin/env python3
"""Evaluate saved PCCR prediction PKLs without running model inference."""

import argparse
import importlib
import os

import mmcv
from mmcv import Config, DictAction
from mmdet3d.datasets import build_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('predictions')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    return parser.parse_args()


def import_plugin(config):
    if not config.get('plugin', False):
        return
    plugin_dir = config.get('plugin_dir', 'projects/mmdet3d_plugin/')
    module_path = os.path.dirname(plugin_dir).replace('/', '.')
    importlib.import_module(module_path)


def main():
    args = parse_args()
    config = Config.fromfile(args.config)
    if args.cfg_options:
        config.merge_from_dict(args.cfg_options)
    import_plugin(config)

    config.data.test.test_mode = True
    dataset = build_dataset(config.data.test)
    predictions = mmcv.load(args.predictions)
    if len(predictions) != len(dataset):
        raise ValueError(
            'prediction count {} does not match dataset size {}'.format(
                len(predictions), len(dataset)))

    eval_kwargs = config.get('evaluation', {}).copy()
    for key in ('interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                'rule', 'pipeline'):
        eval_kwargs.pop(key, None)
    eval_kwargs['metric'] = ['bbox']
    metrics = dataset.evaluate(predictions, **eval_kwargs)
    print(metrics)


if __name__ == '__main__':
    main()
