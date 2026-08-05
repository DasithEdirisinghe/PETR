"""URoPE geometry and rotary transforms for PETR 2D--3D attention.

This implements the 2D--3D extension described in the URoPE paper: image
tokens are lifted at fixed camera-depth anchors and compared with 3D object
queries in the common LiDAR coordinate frame.  Unlike PETR's 3DPE, these
coordinates are parameter-free and are applied relatively to Q/K features.
"""

import numpy as np
import torch


def build_urope_3d_positions(reference_points, img_metas, feature_shape,
                             pc_range, num_heads, depth_num=4,
                             min_depth=1.0, max_depth=60.0, lid=False):
    """Build query points and one lifted image point per attention head.

    Args:
        reference_points (Tensor): Normalized PETR references, [B, Q, 3].
        img_metas (list[dict]): Metadata containing augmented ``lidar2img``.
        feature_shape (tuple[int, int]): Image feature height and width.
        pc_range (sequence[float]): LiDAR range used to normalize references.
        num_heads (int): Number of cross-attention heads.
        depth_num (int): Number of fixed depth anchors shared across heads.
        min_depth/max_depth (float): Camera-z anchor range in metres.
        lid (bool): Use linearly increasing discretization instead of uniform.

    Returns:
        tuple[Tensor, Tensor]: metric query points [B,Q,3] and lifted key
        points [B,num_heads,N*H*W,3], both in the LiDAR frame.
    """
    if depth_num < 1 or depth_num > num_heads:
        raise ValueError('depth_num must be in [1, num_heads]')
    if max_depth <= min_depth:
        raise ValueError('max_depth must be greater than min_depth')

    device = reference_points.device
    dtype = reference_points.dtype
    batch_size = reference_points.size(0)
    height, width = feature_shape
    num_cams = len(img_metas[0]['lidar2img'])

    ranges = reference_points.new_tensor(pc_range)
    query_points = reference_points * (ranges[3:] - ranges[:3]) + ranges[:3]

    anchor_index = torch.arange(depth_num, device=device, dtype=dtype)
    if lid:
        bin_size = ((max_depth - min_depth) /
                    (depth_num * (depth_num + 1)))
        anchors = min_depth + bin_size * anchor_index * (anchor_index + 1)
    else:
        bin_size = (max_depth - min_depth) / depth_num
        anchors = min_depth + bin_size * anchor_index

    # Contiguous groups of heads share an anchor when depth_num < num_heads.
    head_anchor_ids = (
        torch.arange(num_heads, device=device) * depth_num) // num_heads
    head_depths = anchors[head_anchor_ids]

    pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
    coords_h = (torch.arange(height, device=device, dtype=dtype) + 0.5)
    coords_w = (torch.arange(width, device=device, dtype=dtype) + 0.5)
    coords_h = coords_h * pad_h / height
    coords_w = coords_w * pad_w / width
    # meshgrid without ``indexing`` is compatible with PETR's older PyTorch.
    grid_w, grid_h, grid_d = torch.meshgrid(coords_w, coords_h, head_depths)
    coords = torch.stack((grid_w * grid_d, grid_h * grid_d, grid_d), dim=-1)
    coords = torch.cat((coords, torch.ones_like(coords[..., :1])), dim=-1)

    img2lidars = []
    for img_meta in img_metas:
        if len(img_meta['lidar2img']) != num_cams:
            raise ValueError('All samples must contain the same camera count')
        img2lidars.append([
            np.linalg.inv(lidar2img) for lidar2img in img_meta['lidar2img']
        ])
    img2lidars = coords.new_tensor(np.asarray(img2lidars))

    # [B,N,W,H,heads,4] -> [B,heads,N,H,W,3] -> [B,heads,L,3].
    coords = coords.view(1, 1, width, height, num_heads, 4, 1)
    points = torch.matmul(
        img2lidars[:, :, None, None, None], coords).squeeze(-1)[..., :3]
    key_points = points.permute(0, 4, 1, 3, 2, 5).contiguous()
    key_points = key_points.view(batch_size, num_heads,
                                 num_cams * height * width, 3)
    return query_points, key_points


def apply_urope_3d(feats, positions, freq_base=100.0, freq_scale=1.0,
                   inverse=False):
    """Apply axial 3D RoPE to [B,heads,L,head_dim] projected features."""
    if feats.dim() != 4 or positions.size(-1) != 3:
        raise ValueError('Expected feats [B,H,L,D] and positions [...,L,3]')
    head_dim = feats.size(-1)
    axis_dim = (head_dim // 6) * 2
    if axis_dim == 0:
        raise ValueError('URoPE requires a per-head dimension of at least 6')

    if positions.dim() == 3:
        positions = positions[:, None].expand(-1, feats.size(1), -1, -1)
    if positions.shape[:3] != feats.shape[:3]:
        raise ValueError('URoPE feature and position shapes do not align')

    num_freqs = axis_dim // 2
    frequencies = freq_scale * (
        freq_base ** (-torch.arange(num_freqs, device=feats.device,
                                    dtype=torch.float32) / num_freqs))
    blocks = []
    for axis in range(3):
        start = axis * axis_dim
        block = feats[..., start:start + axis_dim]
        angles = positions[..., axis, None].float() * frequencies
        cos = torch.cos(angles).to(dtype=feats.dtype)
        sin = torch.sin(angles).to(dtype=feats.dtype)
        first, second = torch.split(block, num_freqs, dim=-1)
        if inverse:
            rotated = (cos * first - sin * second,
                       sin * first + cos * second)
        else:
            rotated = (cos * first + sin * second,
                       -sin * first + cos * second)
        blocks.append(torch.cat(rotated, dim=-1))
    if 3 * axis_dim < head_dim:
        blocks.append(feats[..., 3 * axis_dim:])
    return torch.cat(blocks, dim=-1)
