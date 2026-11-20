"""Thin adapters: live tasks use the same library APIs as notebooks."""

from dataclasses import asdict
from pathlib import Path

from ..digest._events import emit
from ..digest.source import parse_document
from .models import RunRequest
from .runs import blocking_call


class Tasks:
    def __init__(self, store_root: Path):
        self.store_root = store_root
        self.manager = None
        self.entity_models = {}

    async def __call__(self, request: RunRequest) -> dict:
        handlers = {
            "segmentation": self.segment,
            "claimify": self.claimify,
            "openie": self.openie,
            "entities": self.entities,
            "claim_store": self.claim_store,
        }
        return await handlers[request.task](request)

    async def segment(self, request):
        from ..digest.segmenter import texttiling
        from ..digest.segmenter.llm_e2e import llm_segment

        if request.method == "llm":
            markdown = await llm_segment(request.source, model=request.generation_model)
            return {"markdown": markdown, "sections": []}
        structure = (
            await texttiling.detect_document_structure(
                request.source,
                decision_model=request.generation_model,
                max_concurrency=request.max_concurrency,
            )
            if request.detect_headings
            else texttiling.parse_document_structure(request.source)
        )
        if request.method == "semantic":
            sections = await texttiling.segment_article_semantic(
                request.source,
                structure=structure,
                embedding_model=request.embedding_model,
                local_files_only=request.local_files_only,
                local_model_path=request.local_model_path,
            )
        else:
            sections = texttiling.segment_article(
                request.source, structure=structure, vectorizer=request.method
            )
        if request.generate_headings:
            sections = await texttiling.generate_headings(
                sections,
                article_text=request.source,
                model=request.generation_model,
                max_concurrency=request.max_concurrency,
            )
        return {
            "markdown": texttiling.render_markdown(sections),
            "sections": [asdict(s) for s in sections],
        }

    async def claimify(self, request):
        from ..digest.claimify import extract_claims

        result = await extract_claims(
            request.source,
            question=request.question,
            decision_model=request.decision_model,
            generation_model=request.generation_model,
            max_concurrency=request.max_concurrency,
        )
        return result.model_dump(mode="json")

    async def openie(self, request):
        from ..digest.openie import extract_claims

        result = await extract_claims(
            request.source,
            resolution_model=request.generation_model,
            extraction_model=request.generation_model,
            max_concurrency=request.max_concurrency,
        )
        return result.model_dump(mode="json")

    def _tag_entities(self, request):
        from gliner import GLiNER

        from ..digest.gliner import extract_entities

        key = (request.entity_model, request.local_files_only)
        if key not in self.entity_models:
            emit("model_loading", status="running", model=request.entity_model)
            model = GLiNER.from_pretrained(
                request.entity_model, local_files_only=request.local_files_only
            )
            model.to("cpu")
            model.eval()
            self.entity_models[key] = model
            emit("model_loading", model=request.entity_model)
        return extract_entities(
            request.source,
            model=self.entity_models[key],
            labels=request.labels,
            threshold=request.threshold,
        )

    async def entities(self, request):
        return {"entities": await blocking_call(self._tag_entities, request)}

    async def claim_store(self, request):
        from ..digest.claim_store import ClaimStore
        from ..digest.claimify.models import ExtractionResult

        store = ClaimStore(
            self.store_root,
            embedding_model=request.embedding_model or "openai:text-embedding-3-small",
            comparison_model=request.decision_model,
            max_concurrency=request.max_concurrency,
        )
        if request.store_action == "retrieve":
            matches = await store.retrieve(request.source, top_k=request.top_k)
            emit(
                "retrieval",
                matches=[match.model_dump(mode="json") for match in matches],
            )
            return {"matches": [match.model_dump(mode="json") for match in matches]}
        parent = self.manager.runs[request.parent_run_id]
        extraction = ExtractionResult.model_validate(parent.result)
        records = store.store_unverified(extraction, source=request.source_name)
        if request.store_action == "save":
            return {"records": [record.model_dump(mode="json") for record in records]}
        report = await store.check(records, top_k=request.top_k)
        return report.model_dump(mode="json")

    def validate(self, request):
        """Validate structure and handoffs before starting a paid task."""
        if request.task == "claim_store" and request.store_action != "retrieve":
            parent = self.manager.runs.get(request.parent_run_id)
            if (
                not parent
                or parent.status != "completed"
                or parent.request.task != "claimify"
            ):
                raise ValueError(
                    "Select a completed Claimify run before saving or comparing claims."
                )
            request = request.model_copy(
                update={
                    "source": parent.request.source,
                    "source_name": parent.request.source_name,
                }
            )
        if request.task in {"claimify", "openie"}:
            parse_document(request.source)
        if not request.source.strip():
            raise ValueError("Choose an example or provide source text first.")
        return request
