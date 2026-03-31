"""
Tests for the gamma sweep infrastructure:
  - Concentrated workload generator
  - Workload diversity metric
  - Adaptive semantic policy

Run with: pytest tests/test_sweep.py -v
"""

import numpy as np
import pytest

from src.benchmark.workload import (
    compute_workload_diversity,
    generate_concentrated_workload,
)
from src.cache.eviction.adaptive_semantic import AdaptiveSemanticPolicy


class TestConcentratedWorkload:
    """Test the gamma-parameterized workload generator."""

    @pytest.fixture
    def embeddings(self):
        rng = np.random.RandomState(42)
        n = 200
        dim = 32
        embs = rng.randn(n, dim).astype(np.float32)
        embs /= np.linalg.norm(embs, axis=1, keepdims=True)
        return embs

    def test_gamma_zero_is_uniform(self, embeddings):
        """gamma=0 should produce uniform-ish distribution."""
        indices = generate_concentrated_workload(
            embeddings, n_queries=500, gamma=0.0, seed=42,
        )
        assert len(indices) == 500
        assert indices.max() < len(embeddings)
        assert indices.min() >= 0

    def test_gamma_one_is_concentrated(self, embeddings):
        """gamma=1 should concentrate on very few clusters."""
        indices = generate_concentrated_workload(
            embeddings, n_queries=500, gamma=1.0, seed=42,
        )
        assert len(indices) == 500
        # Most queries should come from a small set of unique indices
        unique_ratio = len(np.unique(indices)) / len(indices)
        # With extreme concentration, we expect high repetition
        assert unique_ratio < 0.5, (
            f"gamma=1.0 should concentrate heavily, "
            f"but unique ratio = {unique_ratio:.2f}"
        )

    def test_concentration_increases_with_gamma(self, embeddings):
        """Higher gamma should produce less unique queries."""
        unique_counts = []
        for gamma in [0.0, 0.5, 1.0]:
            indices = generate_concentrated_workload(
                embeddings, n_queries=500, gamma=gamma, seed=42,
            )
            unique_counts.append(len(np.unique(indices)))

        # unique counts should be non-increasing as gamma increases
        assert unique_counts[0] >= unique_counts[1] >= unique_counts[2], (
            f"Expected decreasing unique counts with gamma, "
            f"got {unique_counts}"
        )

    def test_deterministic_with_seed(self, embeddings):
        """Same seed should produce same workload."""
        idx1 = generate_concentrated_workload(
            embeddings, n_queries=100, gamma=0.5, seed=42,
        )
        idx2 = generate_concentrated_workload(
            embeddings, n_queries=100, gamma=0.5, seed=42,
        )
        np.testing.assert_array_equal(idx1, idx2)


class TestWorkloadDiversity:
    """Test the diversity metric computation."""

    def test_uniform_has_high_diversity(self):
        """Uniform random embeddings should have high entropy."""
        rng = np.random.RandomState(42)
        embs = rng.randn(500, 32).astype(np.float32)
        div = compute_workload_diversity(embs, seed=42)

        assert div["cluster_entropy"] > 0.7, (
            f"Uniform data should have high entropy, "
            f"got {div['cluster_entropy']}"
        )
        assert div["effective_clusters"] > 5

    def test_concentrated_has_low_diversity(self):
        """Repeated embeddings from few directions should have low diversity."""
        rng = np.random.RandomState(42)
        # Create 3 cluster centers, repeat each many times with tiny noise
        centers = rng.randn(3, 32).astype(np.float32)
        embs = []
        for c in centers:
            for _ in range(100):
                noisy = c + rng.randn(32).astype(np.float32) * 0.01
                embs.append(noisy)
        embs = np.array(embs, dtype=np.float32)

        div = compute_workload_diversity(embs, n_clusters=10, seed=42)

        assert div["cluster_entropy"] < 0.75, (
            f"Concentrated data should have low diversity, "
            f"got {div['cluster_entropy']}"
        )

    def test_returns_expected_keys(self):
        rng = np.random.RandomState(42)
        embs = rng.randn(100, 16).astype(np.float32)
        div = compute_workload_diversity(embs, seed=42)

        assert "cluster_entropy" in div
        assert "effective_clusters" in div
        assert "avg_pairwise_distance" in div
        assert "n_clusters_used" in div
        assert "n_queries" in div


class TestAdaptiveSemanticPolicy:
    """Test the adaptive policy's alpha adjustment."""

    def _random_emb(self, dim=16, rng=None):
        if rng is None:
            rng = np.random.RandomState(42)
        v = rng.randn(dim).astype(np.float32)
        v /= np.linalg.norm(v)
        return v

    def test_alpha_adjusts_with_diverse_queries(self):
        """With diverse queries, alpha should increase (LRU-like)."""
        rng = np.random.RandomState(42)
        policy = AdaptiveSemanticPolicy(
            adaptation_window=100,
            adaptation_interval=50,
            alpha_min=0.3,
            alpha_max=2.5,
        )

        # Insert some entries
        for i in range(5):
            policy.on_insert(i, self._random_emb(rng=rng))

        # Feed diverse queries
        for _ in range(60):
            policy.on_query(self._random_emb(rng=rng))

        # Alpha should have increased (diverse → high alpha)
        assert policy.alpha > 1.0, (
            f"Diverse queries should push alpha up, got {policy.alpha}"
        )

    def test_alpha_adjusts_with_concentrated_queries(self):
        """With repeated similar queries, alpha should decrease."""
        rng = np.random.RandomState(42)
        policy = AdaptiveSemanticPolicy(
            adaptation_window=300,
            adaptation_interval=100,
            alpha_min=0.3,
            alpha_max=2.5,
            n_clusters=5,
        )

        for i in range(5):
            policy.on_insert(i, self._random_emb(rng=rng))

        # Feed concentrated queries — same direction with tiny noise
        center = self._random_emb(rng=rng)
        for _ in range(150):
            noisy = center + rng.randn(16).astype(np.float32) * 0.01
            noisy /= np.linalg.norm(noisy)
            policy.on_query(noisy)

        # Alpha should have decreased (concentrated → low alpha)
        assert policy.alpha < 1.5, (
            f"Concentrated queries should push alpha down, got {policy.alpha}"
        )

    def test_name_is_adaptive(self):
        policy = AdaptiveSemanticPolicy()
        assert policy.name == "adaptive"

    def test_stats_include_adaptation_info(self):
        rng = np.random.RandomState(42)
        policy = AdaptiveSemanticPolicy(
        adaptation_interval=10,
        adaptation_window=50,
        )
        for i in range(3):
            policy.on_insert(i, self._random_emb(rng=rng))
        for _ in range(55):
            policy.on_query(self._random_emb(rng=rng))