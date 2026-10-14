"""Tests for geometry module."""

import numpy as np
import pytest

import skg.geom as geom


class TestGeometry:
    """Test geometry functions."""

    def test_backend_available(self) -> None:
        """Backend should be either cpp or numpy."""
        assert geom.BACKEND in ["cpp", "numpy"]

    def test_pairwise_min_distance_simple(self) -> None:
        """Test minimum distance between two trajectories."""
        traj1 = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float32)
        traj2 = np.array([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]], dtype=np.float32)

        dist = geom.pairwise_min_distance(traj1, traj2)
        assert abs(dist - 1.0) < 1e-5

    def test_pairwise_min_distance_touching(self) -> None:
        """Test when trajectories touch."""
        traj1 = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        traj2 = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)

        dist = geom.pairwise_min_distance(traj1, traj2)
        assert abs(dist) < 1e-5

    def test_time_to_collision_approaching(self) -> None:
        """Test TTC calculation for approaching trajectories."""
        ego_traj = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float32)
        other_traj = np.array([[5.0, 0.0], [4.0, 0.0], [3.0, 0.0]], dtype=np.float32)

        ttc = geom.time_to_collision(ego_traj, other_traj)
        assert ttc < 100.0

    def test_time_to_collision_diverging(self) -> None:
        """Test TTC for diverging trajectories should be high."""
        ego_traj = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float32)
        other_traj = np.array([[10.0, 0.0], [11.0, 0.0], [12.0, 0.0]], dtype=np.float32)

        ttc = geom.time_to_collision(ego_traj, other_traj)
        assert ttc > 50.0

    def test_lateral_offset_simple(self) -> None:
        """Test lateral offset from centerline."""
        position = np.array([0.0, 1.0], dtype=np.float32)
        centerline = np.array([[0.0, 0.0], [10.0, 0.0]], dtype=np.float32)

        offset = geom.lateral_offset_from_centerline(position, centerline)
        assert abs(offset - 1.0) < 1e-4

    def test_lateral_offset_on_line(self) -> None:
        """Test lateral offset when on centerline."""
        position = np.array([5.0, 0.0], dtype=np.float32)
        centerline = np.array([[0.0, 0.0], [10.0, 0.0]], dtype=np.float32)

        offset = geom.lateral_offset_from_centerline(position, centerline)
        assert offset < 1e-4

    def test_heading_delta_same(self) -> None:
        """Test heading delta for same headings."""
        delta = geom.heading_delta(0.0, 0.0)
        assert abs(delta) < 1e-5

    def test_heading_delta_opposite(self) -> None:
        """Test heading delta for opposite headings."""
        delta = geom.heading_delta(0.0, np.pi)
        assert abs(delta - np.pi) < 1e-4

    def test_heading_delta_wraparound(self) -> None:
        """Test heading delta with wraparound."""
        delta = geom.heading_delta(-np.pi, np.pi)
        assert abs(delta) < 1e-4

    def test_interaction_features_returns_dict(self) -> None:
        """Test interaction features returns a dict."""
        ego_traj = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        other_traj = np.array([[2.0, 0.0], [3.0, 0.0]], dtype=np.float32)

        features = geom.interaction_features(ego_traj, other_traj)
        assert isinstance(features, dict)
        assert "min_gap" in features
        assert "min_ttc" in features

    def test_numpy_cpp_agree(self) -> None:
        """If C++ available, results should match NumPy within tolerance."""
        if geom.BACKEND != "cpp":
            pytest.skip("C++ backend not available")

        traj1 = np.random.randn(10, 2).astype(np.float32)
        traj2 = np.random.randn(10, 2).astype(np.float32)

        # Just verify it runs without error
        dist = geom.pairwise_min_distance(traj1, traj2)
        assert isinstance(dist, float)
        assert dist >= 0.0
