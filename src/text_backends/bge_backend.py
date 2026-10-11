"""Frozen BGE-M3 dense text embedder (replace.md Step 3).

Prefers ``FlagEmbedding.BGEM3FlagModel`` (the model authors' own inference
package); falls back to ``sentence-transformers`` if ``FlagEmbedding`` is not
installed. Only the dense embedding head is used (sparse / ColBERT vectors are
not requested).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .base import Embedder

DEFAULT_BGE_MODEL = "BAAI/bge-m3"


class BGEM3Embedder(Embedder):
    dim = 1024

    def __init__(
        self,
        model_name: str = DEFAULT_BGE_MODEL,
        *,
        max_length: int = 512,
        batch_size: int = 32,
        use_fp16: bool = True,
        device: str | None = None,
    ) -> None:
        self._max_length = max_length
        self._batch_size = batch_size
        try:
            from FlagEmbedding import BGEM3FlagModel

            self._model = BGEM3FlagModel(model_name, use_fp16=use_fp16, device=device)
            self._backend = "flagembedding"
        except ImportError:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(model_name, device=device)
            self._backend = "sentence-transformers"

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        texts = [text if text else "" for text in texts]
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        if self._backend == "flagembedding":
            output = self._model.encode(
                texts,
                batch_size=self._batch_size,
                max_length=self._max_length,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )
            dense = np.asarray(output["dense_vecs"], dtype=np.float32)
        else:
            dense = np.asarray(
                self._model.encode(
                    texts,
                    batch_size=self._batch_size,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                ),
                dtype=np.float32,
            )

        # BGE-M3 dense vectors are normalized by default; normalize defensively
        # so callers can rely on unit-norm vectors regardless of backend.
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return dense / norms
