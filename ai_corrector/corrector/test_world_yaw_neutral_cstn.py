"""Tests for world-axis yaw neutralization of local right-multiplied deltas."""

import unittest

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from ai_corrector.corrector.yaw_neutral_cstn import (
    torch_lower_body_rotmat_from_features,
    torch_swing_twist_decomposition_y,
    torch_world_swing_local_delta,
    torch_slerp_from_identity,
)
from ai_corrector.corrector.hybrid_composer import slerp_from_identity_wxyz


class TestWorldYawNeutralProjection(unittest.TestCase):
    def test_projected_local_delta_has_zero_world_y_twist(self):
        rng = np.random.RandomState(20261018)
        count = 128
        raw_rot = Rotation.random(count, random_state=rng).as_matrix().astype(np.float32)
        local_delta = Rotation.from_rotvec(
            rng.normal(0.0, 0.12, size=(count, 3))
        ).as_quat().astype(np.float32)
        local_delta = np.concatenate([local_delta[:, 3:4], local_delta[:, :3]], axis=-1)

        x = np.zeros((count, 1, 124), dtype=np.float32)
        rot6 = np.concatenate([raw_rot[..., :, 0], raw_rot[..., :, 1]], axis=-1)
        for offset in (68, 82, 96, 110):
            x[:, 0, offset:offset + 6] = rot6
        q = np.broadcast_to(local_delta[:, None, None, :], (count, 1, 4, 4)).copy()

        x_t = torch.from_numpy(x)
        q_t = torch.from_numpy(q)
        projected_local, discarded_world_twist = torch_world_swing_local_delta(x_t, q_t)
        projected_raw_rot = torch_lower_body_rotmat_from_features(x_t)
        projected_world_vec = torch.matmul(
            projected_raw_rot, projected_local[..., 1:].unsqueeze(-1)
        ).squeeze(-1)
        projected_world_q = torch.nn.functional.normalize(
            torch.cat([projected_local[..., :1], projected_world_vec], dim=-1), dim=-1
        )
        _, output_world_twist = torch_swing_twist_decomposition_y(projected_world_q)

        identity = torch.zeros_like(output_world_twist)
        identity[..., 0] = 1.0
        max_twist = torch.max(torch.abs(output_world_twist - identity)).item()
        self.assertLess(max_twist, 2e-6)

        # The pre-projection twist is reported in world coordinates and remains observable.
        self.assertGreater(torch.linalg.vector_norm(discarded_world_twist[..., 1:], dim=-1).mean().item(), 0.0)

    def test_confidence_slerp_matches_runtime_operator(self):
        rng = np.random.RandomState(11)
        q_xyzw = Rotation.random(128, random_state=rng).as_quat()
        q_wxyz = np.concatenate([q_xyzw[:, 3:4], q_xyzw[:, :3]], axis=-1)
        weights = rng.uniform(0.0, 1.0, size=128)
        actual = torch_slerp_from_identity(
            torch.from_numpy(q_wxyz.astype(np.float32)),
            torch.from_numpy(weights.astype(np.float32)),
            threshold=0.3,
        ).numpy()
        expected = np.stack([
            slerp_from_identity_wxyz(q, w if w >= 0.3 else 0.0)
            for q, w in zip(q_wxyz, weights)
        ])
        dot = np.abs(np.sum(actual * expected, axis=-1))
        self.assertLess(float(np.max(1.0 - dot)), 2e-6)


if __name__ == "__main__":
    unittest.main()
