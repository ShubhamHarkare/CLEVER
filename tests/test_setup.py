"""
Tests for Phase 0 (environment setup) and data pipeline.

Run with: pytest tests/test_setup.py -v
"""

import numpy as np
import pytest


class TestFAISS:
    """Verify FAISS installation and basic functionality."""

    def test_import(self):
        import faiss
        assert faiss is not None

    def test_flat_index_build_and_search(self):
        import faiss

        d = 384
        n = 1000
        xb = np.random.rand(n, d).astype("float32")

        index = faiss.IndexFlatL2(d)
        index.add(xb)

        assert index.ntotal == n

        # Search: first 5 vectors should find themselves as nearest neighbor
        D, I = index.search(xb[:5], k=10)
        assert I.shape == (5, 10)
        assert D.shape == (5, 10)

        for i in range(5):
            assert I[i, 0] == i, f"Self-search failed for query {i}"
            assert D[i, 0] == pytest.approx(0.0, abs=1e-6)

    def test_hnsw_index(self):
        import faiss

        d = 384
        n = 500
        xb = np.random.rand(n, d).astype("float32")

        index = faiss.IndexHNSWFlat(d, 16)
        index.hnsw.efConstruction = 64
        index.hnsw.efSearch = 32
        index.add(xb)

        assert index.ntotal == n

        D, I = index.search(xb[:3], k=5)
        assert I.shape == (3, 5)

    def test_ivf_index(self):
        import faiss

        d = 384
        n = 1000
        nlist = 8
        xb = np.random.rand(n, d).astype("float32")

        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist)
        index.train(xb)
        index.add(xb)
        index.nprobe = 4

        assert index.ntotal == n

        D, I = index.search(xb[:3], k=5)
        assert I.shape == (3, 5)

    def test_lsh_index(self):
        import faiss

        d = 384
        n = 500
        xb = np.random.rand(n, d).astype("float32")

        index = faiss.IndexLSH(d, 768)
        index.add(xb)

        assert index.ntotal == n

        D, I = index.search(xb[:3], k=5)
        assert I.shape == (3, 5)


class TestEncoder:
    """Verify sentence-transformers encoder."""

    def test_import(self):
        from sentence_transformers import SentenceTransformer
        assert SentenceTransformer is not None

    def test_encode_basic(self):
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2")
        emb = model.encode(["test query for CLEVER"])

        assert emb.shape == (1, 384)
        assert emb.dtype == np.float32

    def test_encode_normalized(self):
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2")
        emb = model.encode(
            ["test query"],
            normalize_embeddings=True,
        )

        norm = np.linalg.norm(emb[0])
        assert abs(norm - 1.0) < 1e-5, f"Norm should be 1.0, got {norm}"

    def test_semantic_similarity(self):
        """Similar queries should have high cosine similarity."""
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2")
        embs = model.encode(
            [
                "What is photosynthesis?",
                "How does photosynthesis work?",
                "Best pizza in NYC?",
            ],
            normalize_embeddings=True,
        )

        # Cosine similarity via dot product (embeddings are normalized)
        sim_related = np.dot(embs[0], embs[1])
        sim_unrelated = np.dot(embs[0], embs[2])

        assert sim_related > 0.7, f"Related queries should be similar: {sim_related}"
        assert sim_unrelated < 0.4, f"Unrelated queries should not be similar: {sim_unrelated}"


class TestQueryEncoder:
    """Test our QueryEncoder wrapper."""

    def test_encoder_init(self):
        from src.embeddings.encoder import QueryEncoder

        encoder = QueryEncoder(device="cpu")
        assert encoder.embedding_dim == 384

    def test_encoder_encode(self):
        from src.embeddings.encoder import QueryEncoder

        encoder = QueryEncoder(device="cpu")
        embs = encoder.encode(
            ["Hello world", "Test query"],
            batch_size=2,
            normalize=True,
            show_progress=False,
        )
        assert embs.shape == (2, 384)
        assert embs.dtype == np.float32

        # Check normalization
        norms = np.linalg.norm(embs, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_encoder_single(self):
        from src.embeddings.encoder import QueryEncoder

        encoder = QueryEncoder(device="cpu")
        emb = encoder.encode_single("Single query test")
        assert emb.shape == (1, 384)


class TestPreprocessor:
    """Test query preprocessing logic."""

    def test_filter_empty(self):
        import pandas as pd
        from src.data.preprocessor import filter_queries

        df = pd.DataFrame({
            "query_text": ["valid query here", "", "   ", "another valid one"],
            "query_id": [0, 1, 2, 3],
        })
        result = filter_queries(df, min_tokens=1, max_tokens=100)
        assert len(result) == 2

    def test_filter_length(self):
        import pandas as pd
        from src.data.preprocessor import filter_queries

        df = pd.DataFrame({
            "query_text": ["hi", "this is a valid query", "a " * 600],
            "query_id": [0, 1, 2],
        })
        result = filter_queries(df, min_tokens=3, max_tokens=512)
        assert len(result) == 1
        assert result.iloc[0]["query_text"] == "this is a valid query"

    def test_deduplicate(self):
        import pandas as pd
        from src.data.preprocessor import deduplicate_queries

        df = pd.DataFrame({
            "query_text": ["hello", "world", "hello", "hello"],
            "query_id": [0, 1, 2, 3],
        })
        result = deduplicate_queries(df)
        assert len(result) == 2
        hello_row = result[result["query_text"] == "hello"].iloc[0]
        assert hello_row["frequency"] == 3


class TestSampler:
    """Test deterministic subset creation."""

    def test_create_subsets(self, tmp_path):
        import pandas as pd
        from src.data.sampler import create_subsets

        # Create a fake dataset of 1000 queries
        df = pd.DataFrame({
            "query_id": range(1000),
            "query_text": [f"query {i}" for i in range(1000)],
        })

        sizes = {"tiny": 100, "small": 500}
        saved = create_subsets(df, output_dir=tmp_path, sizes=sizes, seed=42)

        assert "tiny" in saved
        assert "small" in saved
        assert "full" in saved

        tiny_df = pd.read_parquet(saved["tiny"])
        assert len(tiny_df) == 100

        small_df = pd.read_parquet(saved["small"])
        assert len(small_df) == 500

    def test_subsets_are_deterministic(self, tmp_path):
        import pandas as pd
        from src.data.sampler import create_subsets

        df = pd.DataFrame({
            "query_id": range(500),
            "query_text": [f"query {i}" for i in range(500)],
        })

        sizes = {"test": 100}
        saved1 = create_subsets(df, output_dir=tmp_path / "run1", sizes=sizes, seed=42)
        saved2 = create_subsets(df, output_dir=tmp_path / "run2", sizes=sizes, seed=42)

        df1 = pd.read_parquet(saved1["test"])
        df2 = pd.read_parquet(saved2["test"])

        # Same seed should produce same subset
        assert list(df1["query_text"]) == list(df2["query_text"])

    def test_skip_oversized_subset(self, tmp_path):
        import pandas as pd
        from src.data.sampler import create_subsets

        df = pd.DataFrame({
            "query_id": range(50),
            "query_text": [f"query {i}" for i in range(50)],
        })

        sizes = {"too_big": 100}
        saved = create_subsets(df, output_dir=tmp_path, sizes=sizes, seed=42)

        assert "too_big" not in saved
        assert "full" in saved
