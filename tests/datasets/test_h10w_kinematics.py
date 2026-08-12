import numpy as np
import pytest

from lerobot.data_platform.precompute.preprocess.h10w_kinematics import (
    H10WDVT2ArmKinematics,
)
from lerobot.data_platform.precompute.preprocess.stage_return_height_alignment import (
    _align_episode_height,
)


@pytest.mark.parametrize("side", ["left", "right"])
def test_h10w_dvt2_tcp_jacobian_matches_finite_difference(side: str):
    kinematics = H10WDVT2ArmKinematics(side=side)
    joints = (kinematics.lower + kinematics.upper) / 2.0

    _, jacobian = kinematics.pose_and_jacobian(joints)
    epsilon = 1e-6
    numeric = np.stack(
        [
            (
                kinematics.pose(joints + np.eye(7)[index] * epsilon)[:3, 3]
                - kinematics.pose(joints - np.eye(7)[index] * epsilon)[:3, 3]
            )
            / (2.0 * epsilon)
            for index in range(7)
        ],
        axis=1,
    )

    assert np.allclose(jacobian, numeric, atol=1e-8)


@pytest.mark.parametrize("side", ["left", "right"])
def test_h10w_dvt2_height_projection_is_small_and_accurate(side: str):
    kinematics = H10WDVT2ArmKinematics(side=side)
    joints = (kinematics.lower + kinematics.upper) / 2.0
    original_height = float(kinematics.pose(joints)[2, 3])

    projected = kinematics.project_to_height(joints, original_height + 0.03)

    assert abs(kinematics.pose(projected)[2, 3] - (original_height + 0.03)) <= 2e-6
    assert np.linalg.norm(projected - joints) < 0.1
    assert np.all(projected >= kinematics.lower)
    assert np.all(projected <= kinematics.upper)


def test_h10w_dvt2_zero_pose_is_symmetric():
    left = H10WDVT2ArmKinematics(side="left").pose(np.zeros(7))[:3, 3]
    right = H10WDVT2ArmKinematics(side="right").pose(np.zeros(7))[:3, 3]

    assert np.allclose(left[[0, 2]], right[[0, 2]], atol=1e-9)
    assert left[1] == pytest.approx(-right[1], abs=1e-9)


def test_height_alignment_changes_only_active_arm_and_has_no_height_backtrack():
    left = H10WDVT2ArmKinematics(side="left")
    right = H10WDVT2ArmKinematics(side="right")
    left_mid = (left.lower + left.upper) / 2.0
    right_mid = (right.lower + right.upper) / 2.0
    action = np.zeros((20, 16), dtype=np.float64)
    action[:, :7] = left_mid
    action[:, 8:15] = right_mid
    action[:, 1] += 0.08 * np.sin(np.linspace(0.0, 2.0 * np.pi, len(action)))
    action[:, 7] = 0.25
    action[:, 15] = 0.75
    target = float(left.pose(left_mid)[2, 3] + 0.03)

    aligned, metrics = _align_episode_height(
        action,
        start=3,
        boundary=12,
        target_height=target,
        side="left",
        kinematics=left,
        height_tolerance_m=1e-5,
        max_step_rad=0.1,
    )

    heights = left.positions(aligned[3:, :7])[:, 2]
    distance = np.abs(heights[:10] - target)
    assert np.array_equal(aligned[:3], action[:3])
    assert np.array_equal(aligned[:, 8:15], action[:, 8:15])
    assert np.array_equal(aligned[:, [7, 15]], action[:, [7, 15]])
    assert np.all(np.diff(distance) <= 1e-5)
    assert np.allclose(heights[9:], target, atol=1e-5)
    assert metrics["aligned_return_height_backtrack_m"] <= 9e-5
