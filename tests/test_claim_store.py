"""Offline storage, retrieval and comparison contracts."""

from support import Embeddings

import asyncio
import json
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from pydantic_ai.embeddings.test import TestEmbeddingModel
from pydantic_ai.messages import ModelResponse, ToolCallPart, UserPromptPart
from pydantic_ai.models.function import FunctionModel

from data_platform.digest.claim_store import ClaimStore
from data_platform.digest.claimify import parse_document
from data_platform.digest.claimify.models import Claim, ExtractionResult


def extraction(text="Acme revenue was $10m in 2025."):
    document = parse_document("# Report\n\n## Results\n\n" + text + "\n")
    sentence = document.sections[0].sentences[0]
    return ExtractionResult(
        document=document,
        claims=[
            Claim(
                text=text,
                start=sentence.start,
                end=sentence.end,
                section_index=0,
                sentence_index=0,
            )
        ],
    )


class StoreTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = Embeddings()
        self.store = ClaimStore(self.root, embedding_model=self.model)

    def add(self, text="Acme revenue was $10m in 2025.", verified=False):
        record = self.store.store_unverified(extraction(text), source="article.md")[0]
        if verified:
            (self.root / "unverified" / f"{record.id}.json").rename(
                self.root / "verified" / f"{record.id}.json"
            )
        return record

    def edit(self, record, **updates):
        path = self.root / "verified" / f"{record.id}.json"
        data = json.loads(path.read_text())
        data.update(updates)
        path.write_text(json.dumps(data))

    async def test_storage_idempotence_and_manual_promotion(self):
        record = self.add()
        self.assertEqual(self.add(), record)
        self.assertEqual(len(self.store.list_claims("unverified")), 1)
        self.assertEqual(await self.store.retrieve("revenue"), [])
        self.assertEqual(self.model.calls, [])
        self.add(verified=True)
        self.assertEqual(self.add(), record)
        self.assertEqual(self.store.list_claims("unverified"), [])
        self.assertEqual(self.store.get_claim(record.id), record)
        self.assertEqual((await self.store.retrieve("revenue"))[0].claim.id, record.id)
        self.assertEqual(record.original_sentence, record.text)
        self.assertEqual(record.document_title, "Report")
        self.assertEqual(record.section_title, "Results")
        other = self.store.store_unverified(extraction(), source="other.md")[0]
        self.assertNotEqual(other.id, record.id)

    async def test_sync_edits_deletions_model_change_and_rebuild(self):
        record = self.add(verified=True)
        self.assertEqual((await self.store.sync_index()).embedded_count, 1)
        self.assertEqual((await self.store.sync_index()).embedded_count, 0)
        self.edit(record, source="updated.md")
        self.assertEqual((await self.store.sync_index()).embedded_count, 0)
        self.edit(record, text="Acme revenue was $12m in 2025.")
        self.assertEqual((await self.store.sync_index()).embedded_count, 1)
        replacement = ClaimStore(self.root, embedding_model=Embeddings("new-model"))
        self.assertEqual((await replacement.sync_index()).embedded_count, 1)
        (self.root / "index.sqlite").unlink()
        self.assertEqual((await replacement.sync_index()).embedded_count, 1)
        (self.root / "verified" / f"{record.id}.json").rename(
            self.root / "unverified" / f"{record.id}.json"
        )
        self.assertEqual((await replacement.sync_index()).removed_count, 1)
        self.assertEqual(await replacement.retrieve("revenue"), [])

    async def test_ranking_and_exclusion(self):
        first = self.add(verified=True)
        second = self.add("Acme makes cars.", verified=True)
        self.add("Another revenue claim.")
        matches = await self.store.retrieve("revenue", top_k=1)
        self.assertEqual([m.claim.id for m in matches], [first.id])
        self.assertAlmostEqual(matches[0].similarity, 1.0)
        matches = await self.store.retrieve("cars")
        self.assertEqual([m.claim.id for m in matches], [second.id, first.id])

    async def test_invalid_files_and_arguments(self):
        for status in ("verified", "unverified"):
            path = self.root / status / "broken.json"
            path.write_text("{}")
            with self.assertRaisesRegex(ValueError, "broken.json"):
                self.store.list_claims(status)
            path.unlink()
        for value in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                await self.store.retrieve("revenue", top_k=value)
            with self.assertRaises(ValueError):
                ClaimStore(self.root, max_concurrency=value)
        with self.assertRaises(ValueError):
            self.store.get_claim("../bad")
        record = self.add(verified=True)
        self.edit(record, end=0)
        with self.assertRaises(ValueError):
            await self.store.sync_index()

    async def test_comparison_report_and_single_sync(self):
        approved = self.add(verified=True)
        new = self.add("Acme revenue was $12m in 2025.")
        seen = []

        def respond(messages, info):
            payload = next(
                json.loads(p.content)
                for m in reversed(messages)
                for p in m.parts
                if isinstance(p, UserPromptPart)
            )
            seen.append(payload)
            return ModelResponse(
                parts=[ToolCallPart(info.output_tools[0].name, {"verdict": "conflict"})]
            )

        self.store.comparison_model = FunctionModel(respond)
        with patch.object(
            self.store, "_snapshot", wraps=self.store._snapshot
        ) as snapshot:
            report = await self.store.check([new])
            snapshot.assert_awaited_once()
        self.assertEqual(len(report.conflicts), 1)
        self.assertEqual(report.review_required, report.conflicts)
        self.assertEqual(report.uncertain, [])
        self.assertEqual(report.comparisons[0].verified_claim.id, approved.id)
        self.assertEqual(seen[0]["claim"]["source"], "article.md")
        self.assertEqual(len(self.model.calls), 2)
        self.assertEqual(self.store.list_claims("verified"), [approved])

    async def test_failures_propagate_and_cache_remains_retryable(self):
        self.add(verified=True)
        self.model.fail = True
        with self.assertRaisesRegex(RuntimeError, "embedding failed"):
            await self.store.sync_index()
        self.model.fail = False
        self.assertEqual((await self.store.sync_index()).embedded_count, 1)

        def fail(messages, info):
            raise RuntimeError("comparison failed")

        self.store.comparison_model = FunctionModel(fail)
        with self.assertRaisesRegex(RuntimeError, "comparison failed"):
            await self.store.check([self.add("New revenue.")])

    async def test_concurrency_limit_and_all_verdicts(self):
        for i in range(5):
            self.add(f"Acme revenue was {i} million.", verified=True)
        active = peak = 0
        verdicts = iter(
            ["conflict", "equivalent", "compatible", "unrelated", "uncertain"]
        )

        async def respond(messages, info):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            verdict = next(verdicts)
            await asyncio.sleep(0.01)
            active -= 1
            return ModelResponse(
                parts=[ToolCallPart(info.output_tools[0].name, {"verdict": verdict})]
            )

        self.store.max_concurrency = 2
        self.store.comparison_model = FunctionModel(respond)
        report = await self.store.check([self.add("New revenue.")])
        self.assertEqual(peak, 2)
        self.assertEqual(len(report.comparisons), 5)
        self.assertEqual(len(report.review_required), 2)

    async def test_typesafe_comparison_schema(self):
        from pydantic_ai import Agent
        from pydantic_ai.models.decision import ChoiceAnswer, DecisionResponse
        from pydantic_ai.models.typesafe import TypeSafeModel
        from pydantic_ai.providers.typesafe import TypeSafeProvider

        from data_platform.digest.claim_store import ComparisonDecision

        model = TypeSafeModel(
            "jev-latest", provider=TypeSafeProvider(api_key="offline-test")
        )

        async def decide(request, model_settings):
            question = request.questions["verdict"]
            self.assertEqual(question.type, "choice")
            self.assertEqual(
                set(question.criteria),
                {"conflict", "equivalent", "compatible", "unrelated", "uncertain"},
            )
            return DecisionResponse(
                answers={
                    "verdict": ChoiceAnswer(
                        choice="uncertain",
                        confidence=1.0,
                        probabilities={"uncertain": 1.0},
                    )
                },
                model_name="jev-offline-test",
            )

        with patch.object(model, "decide", decide):
            result = await Agent(model, output_type=ComparisonDecision).run(
                "Compare the supplied claims."
            )
        self.assertEqual(result.output.verdict, "uncertain")

    async def test_comparison_failure_and_cancellation_wait_for_siblings(self):
        self.add(verified=True)
        self.add("Acme makes cars.", verified=True)
        records = [self.add("New revenue.")]
        for caller_cancel in (False, True):
            started, finished = asyncio.Event(), asyncio.Event()
            calls = 0

            async def respond(messages, info):
                nonlocal calls
                calls += 1
                if calls == 1:
                    await started.wait()
                    if not caller_cancel:
                        raise RuntimeError("comparison failed")
                else:
                    started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    finished.set()

            self.store.comparison_model = FunctionModel(respond)
            task = asyncio.create_task(self.store.check(records))
            await started.wait()
            if caller_cancel:
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
            else:
                with self.assertRaisesRegex(RuntimeError, "comparison failed"):
                    await task
            self.assertTrue(finished.is_set())

    async def test_invalid_embedding_and_corrupt_cached_vector(self):
        import sqlite3

        record = self.add(verified=True)
        await self.store.sync_index()
        with closing(sqlite3.connect(self.root / "index.sqlite")) as connection:
            with connection:
                connection.execute("UPDATE embeddings SET vector = ?", ("[0, 0]",))
        self.assertEqual((await self.store.sync_index()).embedded_count, 1)
        self.edit(record, text="Changed revenue.")
        original = self.model.embed

        async def invalid(*args, **kwargs):
            response = await original(*args, **kwargs)
            response.embeddings = [[float("nan"), 0]]
            return response

        with patch.object(self.model, "embed", invalid):
            with self.assertRaisesRegex(ValueError, "finite vector"):
                await self.store.sync_index()
        self.assertEqual((await self.store.sync_index()).embedded_count, 1)


if __name__ == "__main__":
    unittest.main()
