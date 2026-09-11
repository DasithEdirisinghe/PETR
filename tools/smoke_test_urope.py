"""CPU smoke tests for the PETR URoPE integration.

Run inside the pinned PETR conda environment before submitting a full job.
"""

import argparse
import importlib

import numpy as np
import torch

from projects.mmdet3d_plugin.models.utils.petr_transformer import \
    PETRMultiheadAttention
from projects.mmdet3d_plugin.models.utils.urope import \
    build_urope_3d_positions


def test_zero_position_attention():
    """Zero rotary coordinates must reproduce standard MHA exactly."""
    torch.manual_seed(0)
    module = PETRMultiheadAttention(
        embed_dims=24, num_heads=4, attn_drop=0.0,
        dropout_layer=None, urope=False).eval()
    query = torch.randn(5, 2, 24, requires_grad=True)
    key = torch.randn(7, 2, 24, requires_grad=True)
    value = torch.randn(7, 2, 24, requires_grad=True)
    padding_mask = torch.tensor([
        [False, False, False, False, False, True, True],
        [False, False, False, False, False, False, True],
    ])

    expected = module(
        query, key, value, key_padding_mask=padding_mask)
    module.urope = True
    query_points = torch.zeros(2, 5, 3)
    key_points = torch.zeros(2, 4, 7, 3)
    actual = module(
        query, key, value, key_padding_mask=padding_mask,
        urope_query_points=query_points,
        urope_key_points=key_points)
    assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-6)
    actual.sum().backward()
    assert all(parameter.grad is not None
               for parameter in module.parameters())


def test_identity_camera_geometry():
    """Identity projection should yield [u*d, v*d, d] key points."""
    references = torch.full((1, 2, 3), 0.5)
    img_metas = [dict(
        pad_shape=[(4, 6, 3)],
        lidar2img=[np.eye(4, dtype=np.float32)])]
    query_points, key_points = build_urope_3d_positions(
        references, img_metas, (2, 3),
        [-1, -2, -3, 1, 2, 3], num_heads=8,
        depth_num=4, min_depth=1.0, max_depth=5.0)

    assert torch.allclose(query_points, torch.zeros_like(query_points))
    assert tuple(key_points.shape) == (1, 8, 6, 3)
    assert torch.allclose(
        key_points[0, 0, 0], torch.tensor([1.0, 1.0, 1.0]))
    assert torch.allclose(
        key_points[0, 2, 0], torch.tensor([2.0, 2.0, 2.0]))


def test_config_build(config_path):
    """Build the configured detector to verify registries and nesting."""
    from mmcv import Config
    from mmdet3d.models import build_detector

    cfg = Config.fromfile(config_path)
    importlib.import_module('projects.mmdet3d_plugin')
    model = build_detector(
        cfg.model, train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    assert model.pts_bbox_head.position_embedding_mode == 'urope'
    assert model.pts_bbox_head.urope_num_heads == 8
    expected_multiview_pe = cfg.model.pts_bbox_head.get(
        'urope_with_multiview_pe', False)
    assert model.pts_bbox_head.urope_with_multiview_pe == expected_multiview_pe
    has_adapt_pos3d_parameters = any(
        'adapt_pos3d' in name for name, _ in model.named_parameters())
    assert has_adapt_pos3d_parameters == expected_multiview_pe
    return cfg, model


def test_full_training_step(cfg, model):
    """Run one real PCCR sample through loss computation and backward."""
    if not torch.cuda.is_available():
        raise RuntimeError('--full-forward requires a CUDA GPU')

    from mmcv.parallel import MMDataParallel, collate
    from mmdet3d.datasets import build_dataset

    dataset = build_dataset(cfg.data.train)
    data = collate([dataset[0]], samples_per_gpu=1)
    model.CLASSES = dataset.CLASSES
    model = MMDataParallel(model.cuda(), device_ids=[0])
    model.train()

    losses = model(return_loss=True, **data)
    loss, log_vars = model.module._parse_losses(losses)
    if not torch.isfinite(loss):
        raise AssertionError('Non-finite loss in URoPE full-forward test')
    loss.backward()

    reference_grad = model.module.pts_bbox_head.reference_points.weight.grad
    if reference_grad is None or not torch.isfinite(reference_grad).all():
        raise AssertionError(
            'Missing or non-finite reference-point gradients')
    print('Full-forward loss: {:.6f}'.format(loss.item()))
    print('Loss terms: {}'.format(log_vars))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        default='projects/configs/petr/'
                'petr_r50dcn_gridmask_p4_800x320_pccr_urope.py')
    parser.add_argument('--skip-config-build', action='store_true')
    parser.add_argument(
        '--full-forward', action='store_true',
        help='Run one real PCCR sample through forward, loss, and backward')
    args = parser.parse_args()

    if args.skip_config_build and args.full_forward:
        parser.error('--full-forward cannot be used with --skip-config-build')

    test_zero_position_attention()
    test_identity_camera_geometry()
    if not args.skip_config_build:
        cfg, model = test_config_build(args.config)
        if args.full_forward:
            test_full_training_step(cfg, model)
    print('URoPE smoke tests passed')


if __name__ == '__main__':
    main()
