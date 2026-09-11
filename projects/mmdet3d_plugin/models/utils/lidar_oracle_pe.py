"""LiDAR-oracle image positional encoding for PETR experiments.

This module intentionally contains no trainable parameters.  It projects the
current LiDAR scan into every camera, keeps the nearest return in each image
feature cell, fills missing cells with the nearest observed camera depth, and
back-projects that depth through the receiving cell's own ray.
"""

import math

import numpy as np
import torch


def sine_position_embedding_1d(position, num_pos_feats, temperature=10000):
    """Encode one normalized scalar using complete sine/cosine pairs."""
    if num_pos_feats <= 0 or num_pos_feats % 2:
        raise ValueError('num_pos_feats must be a positive even integer')

    position = position.float() * (2 * math.pi)
    dim_t = torch.arange(
        num_pos_feats, dtype=torch.float32, device=position.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    phase = position[..., None] / dim_t
    return torch.stack(
        (phase[..., 0::2].sin(), phase[..., 1::2].cos()),
        dim=-1).flatten(-2)


def encode_lidar_xyz(xyz, position_range, x_num_feats=84,
                     y_num_feats=84, z_num_feats=88, temperature=10000,
                     clamp=True):
    """Encode metric LiDAR XYZ as a fixed 256-channel PETR-style PE.

    The returned channel order is ``(y, x, z)``, matching PETR's query-side
    ``pos2posemb3d`` convention.
    """
    if xyz.size(-1) != 3:
        raise ValueError('xyz must have a final dimension of 3')
    if x_num_feats + y_num_feats + z_num_feats != 256:
        raise ValueError('LiDAR-oracle axis dimensions must sum to 256')

    ranges = xyz.new_tensor(position_range, dtype=torch.float32)
    if ranges.numel() != 6 or torch.any(ranges[3:] <= ranges[:3]):
        raise ValueError('position_range must contain valid XYZ min/max values')
    xyz_norm = ((xyz.float() - ranges[:3]) /
                (ranges[3:] - ranges[:3]))
    if clamp:
        xyz_norm = xyz_norm.clamp(0.0, 1.0)

    x_pe = sine_position_embedding_1d(
        xyz_norm[..., 0], x_num_feats, temperature)
    y_pe = sine_position_embedding_1d(
        xyz_norm[..., 1], y_num_feats, temperature)
    z_pe = sine_position_embedding_1d(
        xyz_norm[..., 2], z_num_feats, temperature)
    return torch.cat((y_pe, x_pe, z_pe), dim=-1)


def _as_point_tensor(points, device):
    """Return XYZ from either a BasePoints object or a tensor."""
    if hasattr(points, 'tensor'):
        points = points.tensor
    if not torch.is_tensor(points):
        points = torch.as_tensor(points)
    if points.dim() != 2 or points.size(-1) < 3:
        raise ValueError('Each point cloud must have shape [num_points, >=3]')
    return points[:, :3].to(device=device, dtype=torch.float32)


def _z_buffer(flat_cells, depths, num_cells):
    """Keep the minimum depth and its input index in every occupied cell."""
    dense = depths.new_full((num_cells,), float('inf'))
    selected_indices = flat_cells.new_full((num_cells,), -1)
    if flat_cells.numel() == 0:
        return dense, selected_indices

    # Sorting cell first and depth second gives the closest return as the
    # first entry of each cell group. Float64 avoids loss of depth precision
    # when the cell component is large.
    stride = depths.detach().max().double() + 1.0
    keys = flat_cells.double() * stride + depths.double()
    order = torch.argsort(keys)
    sorted_cells = flat_cells[order]
    first = torch.ones_like(sorted_cells, dtype=torch.bool)
    first[1:] = sorted_cells[1:] != sorted_cells[:-1]
    selected_cells = sorted_cells[first]
    dense[selected_cells] = depths[order[first]]
    selected_indices[selected_cells] = order[first]
    return dense, selected_indices


def _nearest_depth_fill(sparse_depth, fallback_depth):
    """Fill every cell from the spatially nearest cell with a return."""
    height, width = sparse_depth.shape
    occupied = torch.isfinite(sparse_depth)
    grid_y, grid_x = torch.meshgrid(
        torch.arange(height, device=sparse_depth.device),
        torch.arange(width, device=sparse_depth.device))
    grid = torch.stack((grid_y, grid_x), dim=-1).reshape(-1, 2).float()
    occupied_coords = torch.nonzero(occupied, as_tuple=False)

    if occupied_coords.numel() == 0:
        filled = sparse_depth.new_full(
            (height, width), float(fallback_depth))
        distance = sparse_depth.new_full((height, width), float('inf'))
        return filled, occupied, distance

    distances = ((grid[:, None, :] - occupied_coords[None].float()) ** 2)
    distances = distances.sum(-1)
    nearest_distance, nearest_index = distances.min(dim=1)
    nearest_coords = occupied_coords[nearest_index]
    filled = sparse_depth[
        nearest_coords[:, 0], nearest_coords[:, 1]].view(height, width)
    return filled, occupied, nearest_distance.sqrt().view(height, width)


@torch.no_grad()
def build_lidar_oracle_geometry(points, img_metas, feature_shape,
                                position_range, fallback_depth=20.0,
                                interpolation='nearest_depth'):
    """Construct dense ray-consistent XYZ maps from sparse LiDAR returns.

    Returns:
        tuple: XYZ ``[B,N,H,W,3]``, filled camera depth ``[B,N,H,W]``,
        original occupancy, and feature-cell interpolation distance.
    """
    if interpolation != 'nearest_depth':
        raise ValueError('Only nearest_depth interpolation is supported')
    if fallback_depth <= 0:
        raise ValueError('fallback_depth must be positive')
    if len(points) != len(img_metas):
        raise ValueError('points and img_metas must have the same batch size')
    if not points:
        raise ValueError('LiDAR-oracle geometry requires a non-empty batch')

    height, width = feature_shape
    if height <= 0 or width <= 0:
        raise ValueError('feature_shape must be positive')
    device = torch.device('cpu')
    if len(points):
        raw = points[0].tensor if hasattr(points[0], 'tensor') else points[0]
        if torch.is_tensor(raw):
            device = raw.device

    num_cams = len(img_metas[0]['lidar2img'])
    ranges = torch.tensor(position_range, device=device, dtype=torch.float32)
    if ranges.numel() != 6 or torch.any(ranges[3:] <= ranges[:3]):
        raise ValueError('position_range must contain valid XYZ min/max values')
    xyz_batches = []
    depth_batches = []
    occupancy_batches = []
    distance_batches = []

    for batch_index, (sample_points, img_meta) in enumerate(
            zip(points, img_metas)):
        if len(img_meta['lidar2img']) != num_cams:
            raise ValueError('All samples must contain the same camera count')
        xyz_points = _as_point_tensor(sample_points, device)
        in_range = ((xyz_points >= ranges[:3]) &
                    (xyz_points <= ranges[3:])).all(dim=-1)
        xyz_points = xyz_points[in_range]
        homogeneous = torch.cat(
            (xyz_points, xyz_points.new_ones((xyz_points.size(0), 1))),
            dim=-1)

        pad_h, pad_w, _ = img_meta['pad_shape'][0]
        sample_xyz = []
        sample_depth = []
        sample_occupancy = []
        sample_distance = []
        for camera_index, lidar2img in enumerate(img_meta['lidar2img']):
            matrix = torch.as_tensor(
                np.asarray(lidar2img), device=device, dtype=torch.float32)
            if tuple(matrix.shape) != (4, 4):
                raise ValueError('lidar2img matrices must have shape [4,4]')
            projected = homogeneous @ matrix.t()
            camera_depth = projected[:, 2]
            valid_depth = camera_depth > 1e-5
            safe_depth = camera_depth.clamp_min(1e-5)
            image_u = projected[:, 0] / safe_depth
            image_v = projected[:, 1] / safe_depth

            img_h, img_w, _ = img_meta['img_shape'][camera_index]
            valid = (valid_depth & (image_u >= 0) & (image_u < img_w) &
                     (image_v >= 0) & (image_v < img_h))
            cell_x = torch.floor(image_u[valid] * width / pad_w).long()
            cell_y = torch.floor(image_v[valid] * height / pad_h).long()
            cell_x = cell_x.clamp(0, width - 1)
            cell_y = cell_y.clamp(0, height - 1)
            flat_cells = cell_y * width + cell_x
            sparse, selected_indices = _z_buffer(
                flat_cells, camera_depth[valid], height * width)
            sparse = sparse.view(height, width)
            selected_indices = selected_indices.view(height, width)
            filled, occupied, distance = _nearest_depth_fill(
                sparse, fallback_depth)

            grid_y, grid_x = torch.meshgrid(
                torch.arange(height, device=device, dtype=torch.float32),
                torch.arange(width, device=device, dtype=torch.float32))
            # Use cell centres for a symmetric representative camera ray.
            ray_u = (grid_x + 0.5) * float(pad_w) / width
            ray_v = (grid_y + 0.5) * float(pad_h) / height
            image_coords = torch.stack(
                (ray_u * filled, ray_v * filled, filled,
                 torch.ones_like(filled)), dim=-1)
            img2lidar = torch.inverse(matrix)
            lidar_coords = image_coords @ img2lidar.t()
            lidar_xyz = (lidar_coords[..., :3] /
                         lidar_coords[..., 3:].clamp_min(1e-5))
            # Preserve the exact nearest LiDAR return in occupied cells.
            # Only empty cells use copied depth and their own centre ray.
            if occupied.any():
                visible_xyz = xyz_points[valid]
                lidar_xyz[occupied] = visible_xyz[
                    selected_indices[occupied]]

            sample_xyz.append(lidar_xyz)
            sample_depth.append(filled)
            sample_occupancy.append(occupied)
            sample_distance.append(distance)

        xyz_batches.append(torch.stack(sample_xyz))
        depth_batches.append(torch.stack(sample_depth))
        occupancy_batches.append(torch.stack(sample_occupancy))
        distance_batches.append(torch.stack(sample_distance))

    return (torch.stack(xyz_batches), torch.stack(depth_batches),
            torch.stack(occupancy_batches), torch.stack(distance_batches))


@torch.no_grad()
def build_lidar_oracle_position_embedding(
        points, img_metas, feature_shape, position_range,
        x_num_feats=84, y_num_feats=84, z_num_feats=88,
        temperature=10000, clamp=True, fallback_depth=20.0,
        interpolation='nearest_depth'):
    """Build fixed LiDAR-oracle PE as ``[B,N,256,H,W]``."""
    xyz, _, _, _ = build_lidar_oracle_geometry(
        points, img_metas, feature_shape, position_range,
        fallback_depth=fallback_depth, interpolation=interpolation)
    embedding = encode_lidar_xyz(
        xyz, position_range, x_num_feats=x_num_feats,
        y_num_feats=y_num_feats, z_num_feats=z_num_feats,
        temperature=temperature, clamp=clamp)
    return embedding.permute(0, 1, 4, 2, 3).contiguous()
