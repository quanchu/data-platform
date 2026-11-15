import asyncio
import json
import os
import unittest
from dataclasses import replace
from importlib.util import find_spec
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from pydantic_ai.embeddings import TestEmbeddingModel

from data_platform.digest.segmenter.texttiling import (
    EmbeddingGenerationError,
    HeadingGenerationError,
    Section,
    generate_headings,
    resolve_embedding_model,
    segment_article,
    segment_article_semantic,
)
from data_platform.digest.segmenter.texttiling.heading_generator import (
    HEADING_INSTRUCTIONS,
)
from data_platform.digest.segmenter.texttiling.semantic_segmenter import (
    LOCAL_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
    _chunks,
    _sentences,
)

ROOT = Path(__file__).resolve().parents[1]


class FixtureEmbeddingModel(TestEmbeddingModel):
    def __init__(self, *, limit=8000, corrupt=None):
        super().__init__()
        self.calls = []
        self.limit = limit
        self.corrupt = corrupt

    async def max_input_tokens(self):
        return self.limit

    async def count_tokens(self, text):
        return len(text)

    async def embed(self, inputs, *, input_type, settings=None):
        self.calls.append(list(inputs))
        result = await super().embed(inputs, input_type=input_type, settings=settings)
        vectors = [
            [0.0, 1.0]
            if any(
                word in text.casefold()
                for word in ("engine", "wheel", "motor", "vehicle")
            )
            else [1.0, 0.0]
            for text in inputs
        ]
        if self.corrupt:
            vectors = self.corrupt(vectors)
        return replace(result, embeddings=vectors)


class SemanticTests(unittest.IsolatedAsyncioTestCase):
    def assert_preserved(self, text, sections):
        self.assertEqual("".join(s.text for s in sections), text)
        offset = 0
        for section in sections:
            self.assertEqual(section.start, offset)
            self.assertEqual(section.text, text[section.start : section.end])
            offset = section.end
            if section.kind == "prose":
                self.assertLessEqual(len(section.text.strip().split("\n\n")), 4)
        self.assertEqual(offset, len(text))

    async def test_backend_selection(self):
        for key in ("", "  ", "secret"):
            with patch.dict(os.environ, {"OPENAI_API_KEY": key}):
                expected = (
                    OPENAI_EMBEDDING_MODEL if key.strip() else LOCAL_EMBEDDING_MODEL
                )
                self.assertEqual(resolve_embedding_model(), expected)
                self.assertEqual(
                    resolve_embedding_model("openai:custom"), "openai:custom"
                )
                custom = FixtureEmbeddingModel()
                self.assertIs(resolve_embedding_model(custom), custom)
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(resolve_embedding_model(), LOCAL_EMBEDDING_MODEL)

    async def test_no_embedding_needed(self):
        with patch(
            "data_platform.digest.segmenter.texttiling.semantic_segmenter._make_embedder"
        ) as make:
            for text in (
                "",
                " \n ",
                "One paragraph.",
                "> Quote.\n\n“Another.”",
                "Prose.\n\n> Quote.\n\nAfter.",
            ):
                result = await segment_article_semantic(text)
                if text.strip():
                    self.assert_preserved(text, result)
                else:
                    self.assertEqual(result, [])
            make.assert_not_called()

    async def test_fixture_articles_and_quote_isolation(self):
        for lang in ("en", "vi"):
            text = (
                ROOT / f".data/article-{lang}-01.md"
            ).read_text() + "\n\n> Isolated quote."
            model = FixtureEmbeddingModel()
            sections = await segment_article_semantic(text, embedding_model=model)
            self.assert_preserved(text, sections)
            self.assertEqual(sections[-1].kind, "quote")
            self.assertFalse(
                any("Isolated quote" in t for batch in model.calls for t in batch)
            )
            if find_spec("sklearn"):
                self.assert_preserved(text, segment_article(text, vectorizer="tfidf"))

    async def test_semantic_topic_transition(self):
        paragraphs = ["Apples grow. Orchards bloom. Farmers harvest. Fruit ripens."] * 3
        paragraphs += ["Engines start. Wheels turn. Motors run. Vehicles move."] * 3
        text = "\n\n".join(paragraphs)
        sections = await segment_article_semantic(
            text, embedding_model=FixtureEmbeddingModel()
        )
        self.assert_preserved(text, sections)
        self.assertIn(len("\n\n".join(paragraphs[:3])) + 2, [s.start for s in sections])

    async def test_uniform_vectors_only_cap(self):
        text = "\n\n".join(["Different words in the same semantic topic."] * 9)
        sections = await segment_article_semantic(
            text, embedding_model=FixtureEmbeddingModel()
        )
        self.assertEqual(
            [len(s.text.strip().split("\n\n")) for s in sections], [4, 4, 1]
        )

    async def test_sentence_spans(self):
        text = "Mr. A. Nguyen met Dr. Tran at 3.14. Visit https://example.com/path. Ông nói “Xin chào!” Tiếp theo."
        spans = _sentences(text, 0, len(text))
        sentences = [text[a:b].strip() for a, b in spans]
        self.assertEqual(
            sentences,
            [
                "Mr. A. Nguyen met Dr. Tran at 3.14.",
                "Visit https://example.com/path.",
                "Ông nói “Xin chào!”",
                "Tiếp theo.",
            ],
        )
        text = "PGS. TS. Nguyễn nghiên cứu ô tô. Không dấu kết thúc"
        self.assertEqual(len(_sentences(text, 0, len(text))), 2)

    async def test_budget_chunks_and_batches(self):
        text = "\n\n".join(["đ" * 9000 + "."] * 9)
        model = FixtureEmbeddingModel()
        self.assert_preserved(
            text, await segment_article_semantic(text, embedding_model=model)
        )
        self.assertGreater(len(model.calls), 1)
        for batch in model.calls:
            self.assertLessEqual(len(batch), 64)
            self.assertLessEqual(sum(map(len, batch)), 64000)
            self.assertTrue(all(len(t) <= 8000 for t in batch))
        small = FixtureEmbeddingModel(limit=7)
        chunks = [
            chunk
            async for chunk, _ in _chunks(
                "Tiếng Việt còn nguyên", small.count_tokens, 7
            )
        ]
        self.assertEqual("".join(chunks), "Tiếng Việt còn nguyên")
        many = FixtureEmbeddingModel()
        await segment_article_semantic(
            "One. " * 130 + "\n\nOther.", embedding_model=many
        )
        self.assertTrue(all(len(batch) <= 64 for batch in many.calls))

    async def test_bad_vectors_fail_with_context(self):
        for corrupt in (
            lambda _: [],
            lambda v: [[0, 0]] * len(v),
            lambda v: [[float("nan"), 1]] * len(v),
            lambda v: [[1], *v[1:]],
        ):
            with self.assertRaises(EmbeddingGenerationError) as caught:
                await segment_article_semantic(
                    "First.\n\nSecond.",
                    embedding_model=FixtureEmbeddingModel(corrupt=corrupt),
                )
            self.assertIsNotNone(caught.exception.__cause__)
            self.assertEqual(caught.exception.batch_index, 0)

    async def test_no_fallback_after_provider_failure(self):
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "present"}),
            patch(
                "data_platform.digest.segmenter.texttiling.semantic_segmenter._make_embedder",
                side_effect=RuntimeError("quota"),
            ) as make,
            self.assertRaisesRegex(EmbeddingGenerationError, "quota"),
        ):
            await segment_article_semantic("First.\n\nSecond.")
        make.assert_awaited_once_with(OPENAI_EMBEDDING_MODEL, False, None)

    async def test_local_options_forwarded(self):
        from pydantic_ai import Embedder

        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": ""}),
            patch(
                "data_platform.digest.segmenter.texttiling.semantic_segmenter._make_embedder",
                return_value=(Embedder(FixtureEmbeddingModel()), ""),
            ) as make,
        ):
            await segment_article_semantic(
                "First.\n\nSecond.",
                local_files_only=True,
                local_model_path="/prepared/model",
            )
        make.assert_awaited_once_with(LOCAL_EMBEDDING_MODEL, True, "/prepared/model")

    @unittest.skipUnless(find_spec("sklearn"), "Install offline extra for TF-IDF")
    async def test_tfidf_empty_and_unicode(self):
        from data_platform.digest.segmenter.texttiling.texttiling_segmenter import (
            _fit_tfidf,
        )

        self.assertEqual(segment_article(" ", vectorizer="tfidf"), [])
        self.assert_preserved(
            "!!!\n\n???", segment_article("!!!\n\n???", vectorizer="tfidf")
        )
        fitted = _fit_tfidf("kiểm thử xe\n\nkiểm thử động cơ")
        self.assertIn("kiểm thử", fitted.vocabulary_)
        self.assertIn("động", fitted.vocabulary_)
        self.assertLess(
            fitted.idf_[fitted.vocabulary_["kiểm"]],
            fitted.idf_[fitted.vocabulary_["động"]],
        )


class ConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    def sections(self):
        return [Section(f"Claim {i}.", i * 8, i * 8 + 8, "prose") for i in range(6)]

    async def test_parallel_order_and_context(self):
        active = peak = 0
        requests = []

        async def run(prompt):
            nonlocal active, peak
            request = json.loads(prompt)
            requests.append(request)
            index = request["target_section"]["index"]
            active += 1
            peak = max(peak, active)
            await asyncio.sleep((6 - index) * 0.002)
            active -= 1
            return SimpleNamespace(
                output=SimpleNamespace(heading=f"Entity completes action {index}.")
            )

        for concurrency in (1, 3):
            peak = 0
            with patch(
                "data_platform.digest.segmenter.texttiling.heading_generator.Agent"
            ) as agent:
                agent.return_value.run = run
                result = await generate_headings(
                    self.sections(),
                    article_text="Mr. Nguyen, Chief Executive Officer, plans a project.",
                    max_concurrency=concurrency,
                )
            self.assertEqual(peak, concurrency)
            self.assertEqual(
                [s.heading for s in result],
                [f"Entity completes action {i}." for i in range(6)],
            )
            self.assertTrue(
                all(r["article_context"].startswith("Mr. Nguyen") for r in requests)
            )
            self.assertTrue(all(s.heading is None for s in self.sections()))

    async def test_failure_cancels_workers(self):
        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def run(prompt):
            index = json.loads(prompt)["target_section"]["index"]
            if index == 0:
                await started.wait()
                raise RuntimeError("failed")
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        with patch(
            "data_platform.digest.segmenter.texttiling.heading_generator.Agent"
        ) as agent:
            agent.return_value.run = run
            with self.assertRaises(HeadingGenerationError) as caught:
                await generate_headings(self.sections(), max_concurrency=2)
        self.assertEqual(caught.exception.section_index, 0)
        self.assertTrue(cancelled.is_set())

    async def test_caller_cancellation(self):
        started = asyncio.Event()
        finished = asyncio.Event()

        async def run(_):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                finished.set()

        with patch(
            "data_platform.digest.segmenter.texttiling.heading_generator.Agent"
        ) as agent:
            agent.return_value.run = run
            task = asyncio.create_task(
                generate_headings(self.sections(), max_concurrency=1)
            )
            await started.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertTrue(finished.is_set())

    async def test_invalid_concurrency(self):
        for value in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                await generate_headings([], max_concurrency=value)

    async def test_heading_policy_and_quote_payload(self):
        context = "Mr. Tran, Chief Executive Officer of Example, said: “Example plans a factory.”"
        sections = [Section("“Example plans a factory.”", 0, 26, "quote")]
        for required in (
            "subject, verb, and object",
            "first substantive",
            "completed, planned, or expected",
            "neutral reporting verb",
            "honorific",
            "full title",
            "unnamed speaker",
        ):
            self.assertIn(required, HEADING_INSTRUCTIONS)
        with patch(
            "data_platform.digest.segmenter.texttiling.heading_generator.Agent"
        ) as agent:
            agent.return_value.run = AsyncMock(
                return_value=SimpleNamespace(
                    output=SimpleNamespace(heading="A heading.")
                )
            )
            await generate_headings(sections, article_text=context)
            request = json.loads(agent.return_value.run.call_args.args[0])
            self.assertEqual(request["article_context"], context)
            self.assertEqual(request["target_section"]["kind"], "quote")


@unittest.skipUnless(
    os.getenv("RUN_LOCAL_EMBEDDINGS") == "1", "Opt-in local model smoke test"
)
class LocalEmbeddingSmokeTests(unittest.IsolatedAsyncioTestCase):
    async def test_articles(self):
        for language in ("en", "vi"):
            text = (ROOT / f".data/article-{language}-01.md").read_text()
            sections = await segment_article_semantic(
                text,
                embedding_model=LOCAL_EMBEDDING_MODEL,
                local_files_only=os.getenv("LOCAL_FILES_ONLY") == "1",
            )
            self.assertEqual("".join(s.text for s in sections), text)
            print(language, "local boundaries:", [s.start for s in sections])
