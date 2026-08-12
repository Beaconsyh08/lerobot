"""Small NumPy forward-kinematics helper for the H10W DVT2 arm chains."""

from __future__ import annotations

import xml.etree.ElementTree as ET  # nosec B405 - URDF files are trusted local robot assets.
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_H10W_DVT2_URDF = (
    Path(__file__).resolve().parents[4] / "urdf" / "h10w_dvt2" / "H10W_DVT2_TekkenD_with_Camera_2.urdf"
)
TCP_LINKS = {
    "left": "LeftGripperTCP",
    "right": "RightGripperTCP",
}


def _vector(element: ET.Element | None, attribute: str, default: str) -> np.ndarray:
    text = element.get(attribute, default) if element is not None else default
    return np.fromstring(text, sep=" ", dtype=np.float64)


def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def _origin_transform(joint: ET.Element) -> np.ndarray:
    origin = joint.find("origin")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = _rpy_matrix(_vector(origin, "rpy", "0 0 0"))
    transform[:3, 3] = _vector(origin, "xyz", "0 0 0")
    return transform


def _axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    rotation = np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    return transform


@dataclass(frozen=True)
class _ChainJoint:
    name: str
    kind: str
    origin: np.ndarray
    axis: np.ndarray
    q_index: int | None
    lower: float | None
    upper: float | None


class H10WDVT2ArmKinematics:
    """Forward kinematics from the torso frame to a calibrated gripper TCP."""

    def __init__(
        self,
        urdf_path: str | Path = DEFAULT_H10W_DVT2_URDF,
        *,
        side: str,
        base_link: str = "Torso",
        tcp_link: str | None = None,
    ) -> None:
        if side not in TCP_LINKS:
            raise ValueError(f"side must be one of {sorted(TCP_LINKS)}, got {side!r}")
        self.urdf_path = Path(urdf_path).resolve()
        self.side = side
        self.base_link = base_link
        self.tcp_link = tcp_link or TCP_LINKS[side]
        if not self.urdf_path.is_file():
            raise FileNotFoundError(self.urdf_path)

        root = ET.parse(self.urdf_path).getroot()  # nosec B314 - path is an explicit trusted URDF.
        by_child = {joint.find("child").get("link"): joint for joint in root.findall("joint")}
        xml_chain: list[ET.Element] = []
        current = self.tcp_link
        while current != self.base_link:
            joint = by_child.get(current)
            if joint is None:
                raise ValueError(
                    f"No URDF chain from {self.base_link!r} to {self.tcp_link!r}; stopped at {current!r}"
                )
            xml_chain.append(joint)
            current = joint.find("parent").get("link")
        xml_chain.reverse()

        moving = [joint for joint in xml_chain if joint.get("type") == "revolute"]
        if len(moving) != 7:
            names = [joint.get("name") for joint in moving]
            raise ValueError(f"Expected seven revolute arm joints, found {names}")
        q_indices = {joint.get("name"): index for index, joint in enumerate(moving)}

        chain = []
        lower = np.empty(7, dtype=np.float64)
        upper = np.empty(7, dtype=np.float64)
        for joint in xml_chain:
            kind = joint.get("type")
            if kind not in {"fixed", "revolute"}:
                raise ValueError(f"Unsupported joint {joint.get('name')!r} of type {kind!r}")
            q_index = q_indices.get(joint.get("name"))
            axis = _vector(joint.find("axis"), "xyz", "0 0 1")
            low = high = None
            if q_index is not None:
                limit = joint.find("limit")
                if limit is None:
                    raise ValueError(f"Moving joint {joint.get('name')!r} has no limits")
                low, high = float(limit.get("lower")), float(limit.get("upper"))
                lower[q_index], upper[q_index] = low, high
            chain.append(
                _ChainJoint(
                    name=joint.get("name"),
                    kind=kind,
                    origin=_origin_transform(joint),
                    axis=axis,
                    q_index=q_index,
                    lower=low,
                    upper=high,
                )
            )

        self._chain = tuple(chain)
        self.lower = lower
        self.upper = upper
        self.joint_names = tuple(joint.get("name") for joint in moving)

    def pose_and_jacobian(self, joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        joints = np.asarray(joints, dtype=np.float64)
        if joints.shape != (7,):
            raise ValueError(f"Expected joints shape (7,), got {joints.shape}")

        transform = np.eye(4, dtype=np.float64)
        joint_origins: list[np.ndarray] = []
        joint_axes: list[np.ndarray] = []
        for joint in self._chain:
            transform = transform @ joint.origin
            if joint.q_index is None:
                continue
            joint_origins.append(transform[:3, 3].copy())
            joint_axes.append(transform[:3, :3] @ joint.axis)
            transform = transform @ _axis_rotation(joint.axis, joints[joint.q_index])

        tcp = transform[:3, 3]
        jacobian = np.empty((3, 7), dtype=np.float64)
        for index, (origin, axis) in enumerate(zip(joint_origins, joint_axes, strict=True)):
            jacobian[:, index] = np.cross(axis, tcp - origin)
        return transform, jacobian

    def pose(self, joints: np.ndarray) -> np.ndarray:
        return self.pose_and_jacobian(joints)[0]

    def positions(self, joints: np.ndarray) -> np.ndarray:
        values = np.asarray(joints, dtype=np.float64)
        if values.shape[-1:] != (7,):
            raise ValueError(f"Expected final joint dimension 7, got {values.shape}")
        flat = values.reshape(-1, 7)
        result = np.stack([self.pose(row)[:3, 3] for row in flat])
        return result.reshape(*values.shape[:-1], 3)

    def project_to_height(
        self,
        joints: np.ndarray,
        target_height: float,
        *,
        tolerance_m: float = 2e-6,
        max_iterations: int = 40,
        max_iteration_step_rad: float = 0.12,
    ) -> np.ndarray:
        """Project one joint pose to a height with a damped minimum-norm update."""
        reference = np.asarray(joints, dtype=np.float64)
        if reference.shape != (7,) or not np.all(np.isfinite(reference)):
            raise ValueError(f"Expected seven finite joints, got {reference}")
        if np.any(reference < self.lower - 1e-5) or np.any(reference > self.upper + 1e-5):
            raise ValueError(f"Joint pose is outside {self.side} URDF limits: {reference.tolist()}")

        solution = np.clip(reference, self.lower, self.upper)
        for _ in range(max_iterations):
            pose, jacobian = self.pose_and_jacobian(solution)
            error = float(target_height - pose[2, 3])
            if abs(error) <= tolerance_m:
                return solution
            gradient = jacobian[2]
            free = ~(
                ((solution <= self.lower + 1e-9) & (gradient * error < 0))
                | ((solution >= self.upper - 1e-9) & (gradient * error > 0))
            )
            gradient = gradient * free
            denominator = float(gradient @ gradient)
            if denominator < 1e-12:
                break
            step = gradient * error / (denominator + 1e-9)
            step_norm = float(np.linalg.norm(step))
            if step_norm > max_iteration_step_rad:
                step *= max_iteration_step_rad / step_norm

            previous_error = abs(error)
            accepted = False
            for scale in (1.0, 0.5, 0.25, 0.125, 0.0625):
                candidate = np.clip(solution + scale * step, self.lower, self.upper)
                candidate_error = abs(target_height - self.pose(candidate)[2, 3])
                if candidate_error < previous_error:
                    solution = candidate
                    accepted = True
                    break
            if not accepted:
                break

        residual = abs(target_height - self.pose(solution)[2, 3])
        raise ValueError(
            f"Could not project {self.side} TCP to {target_height:.6f} m; residual={residual:.6g} m"
        )
