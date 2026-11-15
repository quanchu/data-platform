import asyncio
import json
import os
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from markdown_it import MarkdownIt
from pydantic import ValidationError
from pydantic_ai.embeddings import TestEmbeddingModel
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from data_platform.digest.segmenter.texttiling import (
    HeadingDetectionError,
    detect_document_structure,
    generate_headings,
    parse_document_structure,
    render_markdown,
    segment_article,
    segment_article_semantic,
)
from data_platform.digest.segmenter.texttiling.document_structure import (
    confirm_candidates,
)
from data_platform.digest.segmenter.texttiling.heading_detection import HeadingDecision

ROOT = Path(__file__).resolve().parents[1]


def decisions_model(decide):
    async def respond(messages, info):
        prompt = next(
            p.content for p in messages[0].parts if p.part_kind == "user-prompt"
        )
        decision = decide(json.loads(prompt))
        if asyncio.iscoroutine(decision):
            decision = await decision
        return ModelResponse(
            parts=[ToolCallPart(info.output_tools[0].name, {"is_heading": decision})]
        )

    return FunctionModel(respond)


class StructureTests(unittest.IsolatedAsyncioTestCase):
    def assert_preserved(self, text, sections):
        self.assertEqual("".join(s.text for s in sections), text)
        cursor = 0
        for section in sections:
            self.assertEqual(section.start, cursor)
            self.assertEqual(section.text, text[section.start : section.end])
            cursor = section.end
        self.assertEqual(cursor, len(text))

    async def test_real_articles_title_first(self):
        for language in ("en", "vi"):
            source = (ROOT / f".data/article-{language}-01.md").read_text()
            structure = await detect_document_structure(
                source,
                decision_model=decisions_model(
                    lambda p: p["candidate"].strip().startswith("ABOUT ")
                ),
            )
            for sections in (
                segment_article(source, structure=structure),
                await segment_article_semantic(
                    source, structure=structure, embedding_model=TestEmbeddingModel()
                ),
            ):
                self.assert_preserved(source, sections)
                self.assertEqual([s.kind for s in sections[:2]], ["title", "metadata"])
                headed = [
                    replace(s, heading="An entity performs an action.")
                    for s in sections
                ]
                result = render_markdown(headed)
                self.assertTrue(result.startswith("# "))
                metadata = "Published date" if language == "en" else "Ngày đăng bài"
                self.assertLess(result.index(metadata), result.index("## "))
                self.assertEqual(
                    sum(
                        t.type == "heading_open" and t.tag == "h1"
                        for t in MarkdownIt().parse(result)
                    ),
                    1,
                )
                if language == "en":
                    self.assertIn("## ABOUT VINGROUP\n", result)
                    self.assertIn("## ABOUT LEGENDS GLOBAL\n", result)
                    self.assertNotIn("## ABOUT VINGROUP\n\n## ", result)

    async def test_candidates_exclusions_and_unicode(self):
        source = """# Title
Published date 2026-01-01

ABOUT VINGROUP

A factory makes cars.

First sentence. Second sentence.

**GIỚI THIỆU VINFAST**

> A quoted sentence.

“Another quotation.”

- List label

1. Numbered label

    Indented code

```markdown
FAKE HEADING
# Not a heading
```

https://example.com

[Link](https://example.com)

<https://example.com>

`inline_code()`

By Alice Nguyen

## Existing heading
"""
        structure = parse_document_structure(source)
        candidates = [c.text.strip() for c in structure.candidates]
        self.assertEqual(
            candidates,
            ["ABOUT VINGROUP", "A factory makes cars.", "**GIỚI THIỆU VINFAST**"],
        )
        self.assertTrue(all(c.is_heading is None for c in structure.candidates))
        with patch(
            "data_platform.digest.segmenter.texttiling.heading_detection.Agent"
        ) as agent:
            segment_article(source)
            agent.assert_not_called()

    async def test_candidate_length_limits(self):
        for line in ("word " * 41, "x" * 241, "Line one\nLine two"):
            self.assertEqual(parse_document_structure(line).candidates, ())

    async def test_unformatted_title_and_metadata(self):
        source = "An article title\n\nNgày đăng bài 22-09-2026\n\nBody content.\n"
        structure = await detect_document_structure(
            source,
            decision_model=decisions_model(
                lambda p: p["candidate"].strip() == "An article title"
            ),
        )
        sections = segment_article(source, structure=structure)
        self.assertEqual([s.kind for s in sections], ["title", "metadata", "prose"])
        self.assert_preserved(source, sections)
        result = render_markdown(
            [replace(s, heading="Body states a fact.") for s in sections]
        )
        self.assertTrue(result.startswith("# An article title\n\nNgày đăng bài"))

    async def test_existing_levels_and_second_h1(self):
        for source in (
            "# Title\n\n# Chapter\n\nBody.\n\n### Detail\n\nMore.",
            "Title\n=====\n\nChapter\n=======\n\nBody.\n\nDetail\n------\n\nMore.",
        ):
            sections = segment_article(source)
            self.assert_preserved(source, sections)
            result = render_markdown(sections)
            headings = [
                t.tag for t in MarkdownIt().parse(result) if t.type == "heading_open"
            ]
            self.assertEqual(headings[0:2], ["h1", "h2"])
            self.assertIn(headings[2], ("h2", "h3"))
            self.assertIn("Chapter", result)

    async def test_skips_structural_and_first_content_headings(self):
        source = (
            "# Title\n\nPublished date 2026-01-01\n\n## Existing\n\n"
            + "\n\n".join(["Uniform body content."] * 5)
        )
        sections = segment_article(source)
        with patch(
            "data_platform.digest.segmenter.texttiling.heading_generator.Agent"
        ) as agent:
            agent.return_value.run = AsyncMock(
                return_value=SimpleNamespace(
                    output=SimpleNamespace(heading="An entity completes an action.")
                )
            )
            result = await generate_headings(sections)
        self.assertEqual(agent.return_value.run.await_count, 1)
        self.assertEqual([s.heading for s in result[:-1]], [None] * (len(result) - 1))
        self.assertEqual(result[-1].heading, "An entity completes an action.")
        self.assertTrue(all(s.heading is None for s in sections))

    async def test_only_structure_needs_no_models(self):
        source = "# Title\n\nPublished date 2026-01-01\n\n## Existing\n"
        with (
            patch(
                "data_platform.digest.segmenter.texttiling.heading_generator.Agent"
            ) as agent,
            patch(
                "data_platform.digest.segmenter.texttiling.semantic_segmenter._make_embedder"
            ) as embed,
        ):
            sections = await segment_article_semantic(source)
            self.assertEqual(await generate_headings(sections), sections)
            agent.assert_not_called()
            embed.assert_not_called()

    async def test_source_validation(self):
        with self.assertRaisesRegex(ValueError, "different source"):
            segment_article(
                "Changed source.", structure=parse_document_structure("Original.")
            )

    async def test_whitespace_and_crlf(self):
        for source in (
            "\n\n# Title\r\n\r\nBody.\r\n\r\n",
            "# Title\nBody.",
            "# Title",
            "  ",
            "# Title\n\n## Empty\n\n## Next\n\nBody.",
        ):
            sections = segment_article(source)
            if source.strip():
                self.assert_preserved(source, sections)
                self.assertEqual(render_markdown(sections), source)
            else:
                self.assertEqual(sections, [])

    async def test_no_cross_heading_embeddings(self):
        source = "# Title\n\nPublished date 2026-01-01\n\nBody one.\n\n## Another section\n\nBody two."
        with patch(
            "data_platform.digest.segmenter.texttiling.semantic_segmenter._make_embedder"
        ) as embed:
            sections = await segment_article_semantic(source)
            embed.assert_not_called()
        self.assertEqual(
            [s.kind for s in sections],
            ["title", "metadata", "prose", "heading", "prose"],
        )


class DecisionTests(unittest.IsolatedAsyncioTestCase):
    async def test_strict_boolean_and_retries(self):
        for value in ("true", 1, None):
            with self.assertRaises(ValidationError):
                HeadingDecision(is_heading=value)
        responses = iter(["true", 1, True])
        structure = await detect_document_structure(
            "ABOUT VINGROUP", decision_model=decisions_model(lambda _: next(responses))
        )
        self.assertTrue(structure.candidates[0].is_heading)
        self.assertEqual(structure.blocks[0].kind, "title")

    async def test_exhaustion_and_failure_index(self):
        with self.assertRaises(HeadingDetectionError) as caught:
            await detect_document_structure(
                "ABOUT VINGROUP", decision_model=decisions_model(lambda _: "not a bool")
            )
        self.assertEqual(caught.exception.candidate_index, 0)
        self.assertIsNotNone(caught.exception.__cause__)

    async def test_missing_credentials_and_no_candidates(self):
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": " "}),
            patch(
                "data_platform.digest.segmenter.texttiling.heading_detection.Agent"
            ) as agent,
        ):
            result = await detect_document_structure("# Title\n\nABOUT VINGROUP")
            self.assertTrue(result.classification_skipped)
            self.assertIsNone(result.candidates[0].is_heading)
            self.assertEqual([b.kind for b in result.blocks], ["title"])
            empty = await detect_document_structure("# Title")
            self.assertFalse(empty.classification_skipped)
            agent.assert_not_called()

    async def test_order_concurrency_and_context(self):
        active = peak = 0

        async def decide(payload):
            nonlocal active, peak
            candidate = payload["candidate"].strip()
            active += 1
            peak = max(peak, active)
            self.assertEqual(payload["document_title"].strip(), "# Title")
            await asyncio.sleep(0.01 if candidate == "ABOUT FIRST" else 0)
            active -= 1
            return candidate.startswith("ABOUT")

        source = "# Title\n\nABOUT FIRST\n\nA factory makes cars.\n\nABOUT SECOND"
        structure = await detect_document_structure(
            source, decision_model=decisions_model(decide), max_concurrency=2
        )
        self.assertEqual(peak, 2)
        self.assertEqual(
            [c.is_heading for c in structure.candidates], [True, False, True]
        )
        self.assertEqual(
            [b.kind for b in structure.blocks], ["title", "heading", "heading"]
        )

    async def test_cancellation_cleanup(self):
        started = asyncio.Event()
        ended = asyncio.Event()

        async def run(_):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                ended.set()

        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test"}),
            patch(
                "data_platform.digest.segmenter.texttiling.heading_detection.Agent"
            ) as agent,
        ):
            agent.return_value.run = run
            task = asyncio.create_task(detect_document_structure("ABOUT FIRST"))
            await started.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertTrue(ended.is_set())

    async def test_bad_decision_count(self):
        with self.assertRaises(ValueError):
            confirm_candidates(parse_document_structure("A label"), [])
