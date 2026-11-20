"""Disposable verified embedding cache and deterministic cosine ranking."""

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pydantic_ai import Embedder
from pydantic_ai.embeddings import EmbeddingModel

from .._events import emit
from .models import ClaimRecord, RetrievalMatch, SyncResult
from .storage import _hash


@dataclass(frozen=True)
class VerifiedSnapshot:
    records: list[ClaimRecord]
    matrix: np.ndarray
    sync: SyncResult


def _vector(value) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.ndim != 1 or not vector.size or not np.isfinite(vector).all():
        raise ValueError("Embedding must be a nonempty finite vector")
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("Embedding must have a finite nonzero norm")
    return vector / norm


async def embed_texts(
    texts: list[str], embedding_model: str | EmbeddingModel, *, query: bool = False
) -> list[np.ndarray]:
    if not texts:
        return []
    embedder = Embedder(embedding_model)
    vectors = []
    for start in range(0, len(texts), 64):
        batch = texts[start : start + 64]
        response = await embedder.embed(
            batch, input_type="query" if query else "document"
        )
        if len(response.embeddings) != len(batch):
            raise ValueError("Embedding count does not match input count")
        vectors.extend(_vector(v) for v in response.embeddings)
    if len({len(v) for v in vectors}) != 1:
        raise ValueError("Embedding dimensions are inconsistent")
    return vectors


def _cached_vectors(records: list[ClaimRecord], cached: dict, model_id: str):
    """Separate reusable vectors from records needing regeneration."""
    vectors = {}
    changed = []
    for record in records:
        row = cached.get(record.id)
        if row and row[:2] == (_hash(record.text), model_id):
            try:
                vectors[record.id] = _vector(json.loads(row[2]))
                continue
            except ValueError, TypeError:
                pass  # Invalid cached vectors can be regenerated from JSON.
        changed.append(record)
    return vectors, changed


async def sync_snapshot(
    root: Path,
    records: list[ClaimRecord],
    embedding_model: str | EmbeddingModel,
    model_id: str,
) -> VerifiedSnapshot:
    connection = sqlite3.connect(root / "index.sqlite")
    try:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS embeddings (id TEXT PRIMARY KEY, text_hash TEXT NOT NULL, model TEXT NOT NULL, vector TEXT NOT NULL)"
        )
        cached = {
            row[0]: row[1:]
            for row in connection.execute(
                "SELECT id, text_hash, model, vector FROM embeddings"
            )
        }
        vectors, changed = _cached_vectors(records, cached, model_id)
        emit(
            "cache_sync",
            status="running",
            verified_count=len(records),
            refresh_count=len(changed),
        )
        fresh = await embed_texts([r.text for r in changed], embedding_model)
        vectors.update((r.id, v) for r, v in zip(changed, fresh, strict=True))
        if len({len(v) for v in vectors.values()}) > 1:
            raise ValueError(
                "Cached embedding dimensions differ; remove index.sqlite and rebuild"
            )
        removed = set(cached) - {r.id for r in records}
        with connection:
            connection.executemany(
                "DELETE FROM embeddings WHERE id = ?", [(id,) for id in removed]
            )
            connection.executemany(
                "INSERT OR REPLACE INTO embeddings VALUES (?, ?, ?, ?)",
                [
                    (r.id, _hash(r.text), model_id, json.dumps(vectors[r.id].tolist()))
                    for r in changed
                ],
            )
        matrix = (
            np.stack([vectors[r.id] for r in records]) if records else np.empty((0, 0))
        )
        emit(
            "cache_sync",
            verified_count=len(records),
            embedded_count=len(changed),
            removed_count=len(removed),
        )
        return VerifiedSnapshot(
            records,
            matrix,
            SyncResult(
                verified_count=len(records),
                embedded_count=len(changed),
                removed_count=len(removed),
            ),
        )
    finally:
        connection.close()


def rank(vector, records, matrix, top_k) -> list[RetrievalMatch]:
    if matrix.shape[1] != len(vector):
        raise ValueError("Query and verified embeddings have different dimensions")
    scores = np.clip(matrix @ vector, -1, 1)
    indices = np.argsort(-scores, kind="stable")[:top_k]
    return [
        RetrievalMatch(claim=records[i], similarity=float(scores[i])) for i in indices
    ]
