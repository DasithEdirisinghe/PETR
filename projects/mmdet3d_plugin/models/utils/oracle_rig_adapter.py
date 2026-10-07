"""Small calibration-conditioned adapters for cross-rig PETR diagnostics.

These modules intentionally use target-domain labels during training.  They
are oracle diagnostics, not deployable test-time adaptation by themselves.
"""

import math

import numpy as np
import torch
import torch.nn as nn


def _zero_linear(layer):
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


class OracleRigAdapter(nn.Module):
    """Calibration encoder and residual heads shared by all oracle modes."""

    VALID_MODES = ('output', 'reference', 'query', 'key', 'query_key')

    def __init__(self, mode, embed_dims=256, hidden_dims=256,
                 calibration_dims=128, residual_scale=1.0,
                 translation_scale=50.0):
        super().__init__()
        if mode not in self.VALID_MODES:
            raise ValueError(
                'oracle adapter mode must be one of {}, got {}'.format(
                    self.VALID_MODES, mode))
        self.mode = mode
        self.embed_dims = embed_dims
        self.residual_scale = residual_scale
        self.translation_scale = translation_scale

        # [fx/W, fy/H, cx/W, cy/H, FOVx/pi, FOVy/pi,
        #  R[:,0], R[:,1], camera_center_xyz/translation_scale]
        calibration_input_dims = 15
        self.calibration_encoder = nn.Sequential(
            nn.Linear(calibration_input_dims, hidden_dims),
            nn.LayerNorm(hidden_dims),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dims, calibration_dims),
            nn.ReLU(inplace=True))
        # Key-only adaptation consumes per-camera embeddings directly.  Do
        # not create a rig-level branch in that mode: it would be unused in
        # the loss graph and make strict DDP reduction fail.
        if self.uses_rig_embedding:
            self.rig_encoder = nn.Sequential(
                nn.Linear(2 * calibration_dims, calibration_dims),
                nn.ReLU(inplace=True))
        else:
            self.rig_encoder = None

        if self.adapts_reference:
            self.reference_head = nn.Sequential(
                nn.Linear(embed_dims + calibration_dims, hidden_dims),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dims, 3))
            _zero_linear(self.reference_head[-1])

        if self.adapts_query:
            self.query_head = nn.Sequential(
                nn.Linear(embed_dims + calibration_dims, hidden_dims),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dims, embed_dims))
            _zero_linear(self.query_head[-1])

        if self.adapts_key:
            # FiLM is applied to PETR's per-camera G3D positional feature.
            self.key_head = nn.Sequential(
                nn.Linear(calibration_dims, hidden_dims),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dims, 2 * embed_dims))
            _zero_linear(self.key_head[-1])

        if self.adapts_output:
            self.output_head = nn.Sequential(
                nn.Linear(embed_dims + calibration_dims, hidden_dims),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dims, 3))
            _zero_linear(self.output_head[-1])

    @property
    def adapts_reference(self):
        return self.mode == 'reference'

    @property
    def adapts_query(self):
        return self.mode in ('query', 'query_key')

    @property
    def adapts_key(self):
        return self.mode in ('key', 'query_key')

    @property
    def adapts_output(self):
        return self.mode == 'output'

    @property
    def uses_rig_embedding(self):
        return self.adapts_reference or self.adapts_query or self.adapts_output

    def encode_calibration(self, img_metas, device, dtype):
        """Return per-camera [B,N,D] and pooled rig [B,D] embeddings."""
        features = []
        camera_counts = []
        for img_meta in img_metas:
            intrinsics = img_meta.get(
                'intrinsics', img_meta.get('camera_intrinsics'))
            lidar2img = img_meta.get(
                'lidar2img', img_meta.get('lidar2image'))
            if lidar2img is None:
                raise KeyError(
                    'oracle adapter requires lidar2img in img_metas')
            if intrinsics is not None and len(intrinsics) != len(lidar2img):
                raise ValueError('intrinsics and lidar2img camera counts differ')

            pad_h, pad_w = img_meta['pad_shape'][0][:2]
            sample_features = []
            for camera_index, projection in enumerate(lidar2img):
                p = np.asarray(projection, dtype=np.float64)[:3, :4]
                if intrinsics is None:
                    k, rotation = self._rq_decompose(p[:, :3])
                else:
                    k = np.asarray(
                        intrinsics[camera_index], dtype=np.float64)[:3, :3]
                    try:
                        rotation = np.linalg.solve(k, p[:, :3])
                    except np.linalg.LinAlgError as error:
                        raise ValueError(
                            'camera intrinsic matrix is singular') from error
                # C = -M^-1 p4 is independent of the K/R factorization.
                camera_center = -np.linalg.pinv(p[:, :3]) @ p[:, 3]

                fx, fy = abs(float(k[0, 0])), abs(float(k[1, 1]))
                cx, cy = float(k[0, 2]), float(k[1, 2])
                fov_x = 2.0 * math.atan(float(pad_w) / max(2.0 * fx, 1e-8))
                fov_y = 2.0 * math.atan(float(pad_h) / max(2.0 * fy, 1e-8))
                feature = np.concatenate((
                    np.asarray([
                        fx / pad_w, fy / pad_h, cx / pad_w, cy / pad_h,
                        fov_x / math.pi, fov_y / math.pi],
                        dtype=np.float32),
                    rotation[:, :2].reshape(-1).astype(np.float32),
                    (camera_center / self.translation_scale).astype(np.float32)))
                sample_features.append(feature)
            features.append(np.stack(sample_features))
            camera_counts.append(len(sample_features))

        if len(set(camera_counts)) != 1:
            raise ValueError(
                'a training batch must have the same camera count per sample')
        calibration = torch.as_tensor(
            np.stack(features), device=device, dtype=torch.float32)
        camera_embedding = self.calibration_encoder(calibration)
        rig_embedding = None
        if self.rig_encoder is not None:
            rig_embedding = self.rig_encoder(torch.cat((
                camera_embedding.mean(dim=1),
                camera_embedding.max(dim=1).values), dim=-1))
            rig_embedding = rig_embedding.to(dtype=dtype)
        return camera_embedding.to(dtype=dtype), rig_embedding

    @staticmethod
    def _rq_decompose(matrix):
        """NumPy-only RQ decomposition of a 3x3 camera matrix."""
        q, r = np.linalg.qr(np.flipud(matrix).T)
        upper = np.fliplr(np.flipud(r.T))
        rotation = np.flipud(q.T)
        signs = np.sign(np.diag(upper))
        signs[signs == 0] = 1
        sign_matrix = np.diag(signs)
        upper = upper @ sign_matrix
        rotation = sign_matrix @ rotation
        if abs(upper[2, 2]) > 1e-8:
            upper = upper / upper[2, 2]
        return upper, rotation

    def adapt_reference(self, reference_points, query_embedding,
                        rig_embedding, eps=1e-5):
        batch_size, num_query = reference_points.shape[:2]
        query = query_embedding.unsqueeze(0).expand(batch_size, -1, -1)
        rig = rig_embedding.unsqueeze(1).expand(-1, num_query, -1)
        delta = self.reference_head(torch.cat((query, rig), dim=-1))
        reference_logits = torch.logit(reference_points.clamp(eps, 1 - eps))
        return torch.sigmoid(
            reference_logits + self.residual_scale * delta)

    def adapt_query(self, query_embedding, rig_embedding):
        batch_size = rig_embedding.size(0)
        num_query = query_embedding.size(-2)
        if query_embedding.dim() == 2:
            base = query_embedding.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            base = query_embedding
        rig = rig_embedding.unsqueeze(1).expand(-1, num_query, -1)
        delta = self.query_head(torch.cat((base, rig), dim=-1))
        return base + self.residual_scale * delta

    def adapt_key(self, position_embedding, camera_embedding):
        film = self.key_head(camera_embedding)
        gamma, beta = film.chunk(2, dim=-1)
        gamma = gamma[..., None, None]
        beta = beta[..., None, None]
        delta = gamma * position_embedding + beta
        return position_embedding + self.residual_scale * delta

    def output_delta(self, decoder_feature, rig_embedding):
        num_query = decoder_feature.size(-2)
        rig = rig_embedding.unsqueeze(1).expand(-1, num_query, -1)
        return self.residual_scale * self.output_head(
            torch.cat((decoder_feature, rig), dim=-1))
