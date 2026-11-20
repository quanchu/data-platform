"""Public facade for storage, verified retrieval, and conflict checking."""

import asyncio
from pathlib import Path

from pydantic_ai.embeddings import EmbeddingModel
from pydantic_ai.models import Model

from .._async import positive_int
from ..claimify.models import ExtractionResult
from .comparison import compare_claims
from .index import VerifiedSnapshot, embed_texts, rank, sync_snapshot
from .models import CheckReport, ClaimRecord, RetrievalMatch, Status, SyncResult
from .storage import ClaimFiles


class ClaimStore:
    def __init__(
        self,
        root: str | Path = ".data/claims",
        *,
        embedding_model: str | EmbeddingModel = "openai:text-embedding-3-small",
        comparison_model: str | Model = "typesafe:jev-latest",
        max_concurrency: int = 20,
    ):
        positive_int(max_concurrency, "max_concurrency")
        self.root = Path(root)
        self.embedding_model = embedding_model
        self.comparison_model = comparison_model
        self.max_concurrency = max_concurrency
        self._lock = asyncio.Lock()
        self._files = ClaimFiles(self.root)
        for status in ("unverified", "verified"):
            (self.root / status).mkdir(parents=True, exist_ok=True)

    @property
    def model_id(self) -> str:
        model = self.embedding_model
        return model if isinstance(model, str) else f"{model.system}:{model.model_name}"

    def list_claims(self, status: Status) -> list[ClaimRecord]:
        return self._files.list_claims(status)

    def get_claim(self, id: str) -> ClaimRecord | None:
        return self._files.get_claim(id)

    def store_unverified(
        self, result: ExtractionResult, *, source: str
    ) -> list[ClaimRecord]:
        return self._files.store_unverified(result, source=source)

    async def _snapshot(self) -> VerifiedSnapshot:
        async with self._lock:
            return await sync_snapshot(
                self.root,
                self.list_claims("verified"),
                self.embedding_model,
                self.model_id,
            )

    async def sync_index(self) -> SyncResult:
        return (await self._snapshot()).sync

    async def retrieve(self, text: str, top_k: int = 20) -> list[RetrievalMatch]:
        positive_int(top_k, "top_k")
        if not text.strip():
            raise ValueError("Query text must not be blank")
        snapshot = await self._snapshot()
        if not snapshot.records:
            return []
        vector = (await embed_texts([text], self.embedding_model, query=True))[0]
        return rank(vector, snapshot.records, snapshot.matrix, top_k)

    async def check(self, records: list[ClaimRecord], top_k: int = 20) -> CheckReport:
        positive_int(top_k, "top_k")
        snapshot = await self._snapshot()
        report = CheckReport(
            claims=records,
            comparisons=[],
            top_k=top_k,
            verified_count=len(snapshot.records),
        )
        if not snapshot.records or not records:
            return report
        vectors = await embed_texts(
            [r.text for r in records], self.embedding_model, query=True
        )
        report.comparisons = await compare_claims(
            records,
            vectors,
            snapshot.records,
            snapshot.matrix,
            top_k=top_k,
            model=self.comparison_model,
            max_concurrency=self.max_concurrency,
        )
        return report
