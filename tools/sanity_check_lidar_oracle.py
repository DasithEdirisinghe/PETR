"""Sanity checks for PETR's LiDAR-oracle positional encoding.

Synthetic checks run on CPU and do not require PCCR data. Run ``--full-forward``
inside the PETR training environment on a GPU to additionally execute one
real PCCR training sample through loss and backward.
"""

import argparse
import importlib
import math
import sys
from pathlib import Path

import numpy as np
import torch

# Make the script directly runnable as
# ``python tools/sanity_check_lidar_oracle.py`` without requiring PYTHONPATH.
PETR_ROOT = Path(__file__).resolve().parents[1]
if str(PETR_ROOT) not in sys.path:
    sys.path.insert(0, str(PETR_ROOT))

from projects.mmdet3d_plugin.datasets.pipelines.transform_3d import \
    GlobalRotScaleTransImage
from projects.mmdet3d_plugin.models.utils.lidar_oracle_pe import (
    build_lidar_oracle_geometry,
    build_lidar_oracle_position_embedding,
    encode_lidar_xyz,
    sine_position_embedding_1d,
)


def _camera_matrix():
    return np.array([
        [4.0, 0.0, 2.0, 0.0],
        [0.0, 4.0, 2.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float32)


def _image_meta(matrix=None):
    if matrix is None:
        matrix = _camera_matrix()
    return dict(
        pad_shape=[(4, 4, 3)],
        img_shape=[(4, 4, 3)],
        lidar2img=[matrix])


def test_fixed_axis_encoding():
    """The PE must use 84/84/88 channels in PETR's y/x/z order."""
    xyz = torch.tensor([[[[[0.25, -0.5, 1.0]]]]])
    position_range = [-2.0, -2.0, -2.0, 2.0, 2.0, 2.0]
    encoded = encode_lidar_xyz(xyz, position_range)
    assert tuple(encoded.shape) == (1, 1, 1, 1, 256)
    assert torch.isfinite(encoded).all()

    normalized = (xyz + 2.0) / 4.0
    expected_y = sine_position_embedding_1d(normalized[..., 1], 84)
    expected_x = sine_position_embedding_1d(normalized[..., 0], 84)
    expected_z = sine_position_embedding_1d(normalized[..., 2], 88)
    assert torch.allclose(encoded[..., :84], expected_y)
    assert torch.allclose(encoded[..., 84:168], expected_x)
    assert torch.allclose(encoded[..., 168:], expected_z)


def test_projection_z_buffer_and_interpolation():
    """Nearest camera depth wins and sparse cells receive finite geometry."""
    # The first two returns project to feature cell (0,0); z=2 must win.
    # The final return projects to feature cell (1,1).
    points = [torch.tensor([
        [-0.75, -0.625, 2.0, 1.0, 0.0],
        [-1.5, -1.25, 4.0, 1.0, 0.0],
        [0.75, 0.75, 3.0, 1.0, 0.0],
    ])]
    position_range = [-10.0, -10.0, -2.0, 10.0, 10.0, 10.0]
    xyz, depth, occupied, distance = build_lidar_oracle_geometry(
        points, [_image_meta()], (2, 2), position_range)

    assert tuple(xyz.shape) == (1, 1, 2, 2, 3)
    assert occupied.sum().item() == 2
    assert depth[0, 0, 0, 0].item() == 2.0
    assert depth[0, 0, 1, 1].item() == 3.0
    # Occupied cells preserve the actual nearest return, rather than replacing
    # it with a point reconstructed at the feature-cell centre.
    assert torch.allclose(
        xyz[0, 0, 0, 0], torch.tensor([-0.75, -0.625, 2.0]))
    assert torch.isfinite(xyz).all()
    assert torch.isfinite(depth).all()
    assert torch.isfinite(distance[occupied]).all()

    embedding = build_lidar_oracle_position_embedding(
        points, [_image_meta()], (2, 2), position_range)
    assert tuple(embedding.shape) == (1, 1, 256, 2, 2)
    assert torch.isfinite(embedding).all()


def test_empty_camera_fallback():
    """A camera with no visible returns must use the configured depth."""
    points = [torch.tensor([[0.0, 0.0, -1.0, 1.0, 0.0]])]
    xyz, depth, occupied, distance = build_lidar_oracle_geometry(
        points, [_image_meta()], (2, 2),
        [-10.0, -10.0, -2.0, 10.0, 10.0, 10.0],
        fallback_depth=7.0)
    assert not occupied.any()
    assert torch.all(depth == 7.0)
    assert torch.isfinite(xyz).all()
    assert torch.isinf(distance).all()


class _DummyPoints:
    def __init__(self, tensor):
        self.tensor = tensor.clone()

    def rotate(self, angle):
        angle = float(angle)
        cosine, sine = math.cos(angle), math.sin(angle)
        rotation = self.tensor.new_tensor([
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ])
        self.tensor[:, :3] = self.tensor[:, :3] @ rotation.t()

    def scale(self, ratio):
        self.tensor[:, :3] *= float(ratio)


class _DummyBoxes:
    def rotate(self, angle):
        self.angle = angle

    def scale(self, ratio):
        self.ratio = ratio


def test_global_augmentation_alignment():
    """Augmented calibration and points must preserve image projections."""
    matrix = _camera_matrix()
    original_xyz = torch.tensor([[1.0, -0.5, 5.0]])
    original_h = torch.cat((original_xyz, torch.ones(1, 1)), dim=-1)
    original_projection = original_h @ torch.from_numpy(matrix).t()

    results = dict(
        lidar2img=[matrix.copy()],
        extrinsics=[np.eye(4, dtype=np.float32)],
        points=_DummyPoints(original_xyz),
        gt_bboxes_3d=_DummyBoxes())
    transform = GlobalRotScaleTransImage(
        rot_range=[0.2, 0.2],
        scale_ratio_range=[1.1, 1.1],
        translation_std=[0, 0, 0],
        reverse_angle=True,
        transform_points=True,
        training=True)
    transformed = transform(results)
    transformed_h = torch.cat(
        (transformed['points'].tensor, torch.ones(1, 1)), dim=-1)
    transformed_projection = transformed_h @ torch.from_numpy(
        transformed['lidar2img'][0]).t()
    assert torch.allclose(
        transformed_projection, original_projection, rtol=1e-5, atol=1e-5)


def test_point_augmentation_is_opt_in():
    """The transform default must preserve behavior of existing pipelines."""
    original_xyz = torch.tensor([[1.0, -0.5, 5.0]])
    results = dict(
        lidar2img=[_camera_matrix()],
        extrinsics=[np.eye(4, dtype=np.float32)],
        points=_DummyPoints(original_xyz),
        gt_bboxes_3d=_DummyBoxes())
    transform = GlobalRotScaleTransImage(
        rot_range=[0.2, 0.2],
        scale_ratio_range=[1.1, 1.1],
        reverse_angle=True,
        training=True)
    transformed = transform(results)
    assert torch.equal(transformed['points'].tensor, original_xyz)


def test_config_build(config_path):
    """Build the detector and check the opt-in configuration and pipelines."""
    from mmcv import Config
    from mmdet3d.models import build_detector

    cfg = Config.fromfile(config_path)
    importlib.import_module('projects.mmdet3d_plugin')
    model = build_detector(
        cfg.model, train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    head = model.pts_bbox_head
    assert head.position_embedding_mode == 'lidar_oracle'
    assert head.with_lidar_oracle
    assert not head.with_position
    assert not head.with_urope
    assert head.lidar_oracle_with_multiview_pe
    assert not hasattr(head, 'position_encoder')
    assert any('adapt_pos3d' in name for name, _ in model.named_parameters())

    for split in ('train', 'val', 'test'):
        pipeline_text = repr(cfg.data[split].pipeline)
        assert 'LoadPointsFromFile' in pipeline_text
        assert 'points' in pipeline_text
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
        raise AssertionError('Non-finite loss in oracle full-forward test')
    loss.backward()
    reference_grad = model.module.pts_bbox_head.reference_points.weight.grad
    if reference_grad is None or not torch.isfinite(reference_grad).all():
        raise AssertionError('Missing or non-finite reference-point gradients')
    print('Full-forward loss: {:.6f}'.format(loss.item()))
    print('Loss terms: {}'.format(log_vars))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        default='projects/configs/petr/'
                'petr_r50dcn_gridmask_p4_800x320_pccr_lidar_oracle.py')
    parser.add_argument('--skip-config-build', action='store_true')
    parser.add_argument(
        '--full-forward', action='store_true',
        help='Run one real PCCR sample through forward, loss, and backward')
    args = parser.parse_args()
    if args.skip_config_build and args.full_forward:
        parser.error('--full-forward cannot be used with --skip-config-build')

    test_fixed_axis_encoding()
    test_projection_z_buffer_and_interpolation()
    test_empty_camera_fallback()
    test_global_augmentation_alignment()
    test_point_augmentation_is_opt_in()
    if not args.skip_config_build:
        cfg, model = test_config_build(args.config)
        if args.full_forward:
            test_full_training_step(cfg, model)
    print('LiDAR-oracle sanity checks passed')


if __name__ == '__main__':
    main()
