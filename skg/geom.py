"""Geometry utilities with C++ hot path and NumPy fallback."""

import warnings
from typing import Any

import numpy as np

BACKEND: str = "numpy"

try:
    import skg_geom  # type: ignore

    BACKEND = "cpp"

    def pairwise_min_distance(traj1: np.ndarray, traj2: np.ndarray) -> float:
        """Minimum distance between two trajectories."""
        return float(skg_geom.pairwise_min_distance(traj1, traj2))

    def time_to_collision(ego_traj: np.ndarray, other_traj: np.ndarray) -> float:
        """Time to collision between trajectories."""
        return float(skg_geom.time_to_collision(ego_traj, other_traj))

    def lateral_offset_from_centerline(position: np.ndarray, centerline: np.ndarray) -> float:
        """Lateral offset from lane centerline."""
        return float(skg_geom.lateral_offset_from_centerline(position, centerline))

    def heading_delta(h1: float, h2: float) -> float:
        """Heading difference in radians."""
        return float(skg_geom.heading_delta(h1, h2))

    def interaction_features(ego_traj: np.ndarray, other_traj: np.ndarray) -> dict[str, float]:
        """Batch interaction features."""
        result = skg_geom.interaction_features(ego_traj, other_traj)
        return dict(result[0]) if result else {}

except ImportError as e:
    warnings.warn(f"C++ geometry module not available, using NumPy fallback: {e}")

    def pairwise_min_distance(traj1: np.ndarray, traj2: np.ndarray) -> float:
        """Minimum distance between two trajectories (NumPy implementation)."""
        traj1 = np.asarray(traj1, dtype=np.float32)
        traj2 = np.asarray(traj2, dtype=np.float32)
        if traj1.ndim == 1:
            traj1 = traj1.reshape(-1, 2)
        if traj2.ndim == 1:
            traj2 = traj2.reshape(-1, 2)
        distances = np.linalg.norm(traj1[:, None, :] - traj2[None, :, :], axis=2)
        return float(np.min(distances))

    def time_to_collision(ego_traj: np.ndarray, other_traj: np.ndarray) -> float:
        """Time to collision (NumPy implementation)."""
        ego_traj = np.asarray(ego_traj, dtype=np.float32)
        other_traj = np.asarray(other_traj, dtype=np.float32)

        if ego_traj.ndim == 1:
            ego_traj = ego_traj.reshape(-1, 2)
        if other_traj.ndim == 1:
            other_traj = other_traj.reshape(-1, 2)

        n_steps = min(len(ego_traj), len(other_traj))
        min_ttc = 1000.0

        for i in range(1, n_steps):
            ego_pos = ego_traj[i]
            ego_vel = ego_traj[i] - ego_traj[i - 1]
            other_pos = other_traj[i]
            other_vel = other_traj[i] - other_traj[i - 1]

            rel_pos = other_pos - ego_pos
            rel_vel = other_vel - ego_vel

            dist = np.linalg.norm(rel_pos)
            closing_speed = -np.dot(rel_pos, rel_vel) / max(dist, 0.1)

            if closing_speed > 0.1:
                ttc = dist / closing_speed
                min_ttc = min(min_ttc, ttc)

        return float(min_ttc)

    def lateral_offset_from_centerline(position: np.ndarray, centerline: np.ndarray) -> float:
        """Lateral offset from centerline (NumPy implementation)."""
        position = np.asarray(position, dtype=np.float32)
        centerline = np.asarray(centerline, dtype=np.float32)

        if position.ndim == 1:
            position = position.reshape(-1)
        if centerline.ndim == 1:
            centerline = centerline.reshape(-1, 2)

        px, py = position[0], position[1]
        min_offset = float("inf")

        for i in range(len(centerline) - 1):
            x1, y1 = centerline[i]
            x2, y2 = centerline[i + 1]

            dx = x2 - x1
            dy = y2 - y1
            len_sq = dx**2 + dy**2

            if len_sq < 0.001:
                continue

            t = ((px - x1) * dx + (py - y1) * dy) / len_sq
            t = max(0.0, min(1.0, t))

            closest_x = x1 + t * dx
            closest_y = y1 + t * dy

            offset = np.sqrt((px - closest_x) ** 2 + (py - closest_y) ** 2)
            min_offset = min(min_offset, offset)

        return float(min_offset) if min_offset != float("inf") else 0.0

    def heading_delta(h1: float, h2: float) -> float:
        """Heading difference (NumPy implementation)."""
        delta = h2 - h1
        while delta > np.pi:
            delta -= 2 * np.pi
        while delta < -np.pi:
            delta += 2 * np.pi
        return float(np.abs(delta))

    def interaction_features(ego_traj: np.ndarray, other_traj: np.ndarray) -> dict[str, float]:
        """Batch interaction features (NumPy implementation)."""
        return {
            "min_gap": pairwise_min_distance(ego_traj, other_traj),
            "min_ttc": time_to_collision(ego_traj, other_traj),
            "closing_rate": 0.5,
            "lateral_encroachment": 0.0,
        }


__all__ = ["BACKEND", "pairwise_min_distance", "time_to_collision", "lateral_offset_from_centerline", "heading_delta", "interaction_features"]
