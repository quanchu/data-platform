import asyncio
import json
import os
import re
import unicodedata
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from pydantic_ai.embeddings import TestEmbeddingModel
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from data_platform.digest.segmenter.texttiling import (
    HeadingGenerationError,
    generate_headings,
    parse_document_structure,
    render_markdown,
    segment_article,
    segment_with_headings,
    texttiling_segement,
)
from data_platform.digest.segmenter.texttiling.heading_generator import (
    DEFAULT_MODEL,
    Heading,
)

ROOT = Path(__file__).resolve().parents[1]


class SegmentationTests(unittest.TestCase):
    def assert_preserved(self, text, sections):
        self.assertEqual("".join(s.text for s in sections), text)
        cursor = 0
        for section in sections:
            self.assertEqual(section.start, cursor)
            self.assertEqual(section.text, text[section.start : section.end])
            self.assertTrue(section.text.strip())
            self.assertIsNone(section.heading)
            cursor = section.end
            if section.kind == "prose":
                self.assertLessEqual(len(re.split(r"\n\s*\n", section.text.strip())), 4)
        self.assertEqual(cursor, len(text))

    def test_real_articles(self):
        for language in ("en", "vi"):
            with self.subTest(language=language):
                text = (ROOT / f".data/article-{language}-01.md").read_text()
                sections = segment_article(text)
                self.assert_preserved(text, sections)
                # The English article contains three separately attributed quotations.
                self.assertEqual([s.kind for s in sections[:2]], ["title", "metadata"])
                self.assertEqual(
                    sum(s.kind == "quote" for s in sections),
                    3 if language == "en" else 0,
                )
                quoted = text + "\n\n“Standalone quotation.”\n"
                isolated = segment_article(quoted)
                self.assert_preserved(quoted, isolated)
                self.assertEqual(isolated[-1].kind, "quote")

    def test_render_without_headings_preserves_source(self):
        source = "Prose.\n\n> Quote.\n"
        self.assertEqual(render_markdown(segment_article(source)), source)
        self.assertEqual(render_markdown([]), "")

    def test_blank_and_short(self):
        for text in ("", " \n\t\r\n"):
            self.assertEqual(segment_article(text), [])
        text = "\n\n  Một đoạn ngắn.\r\n \r\nSecond paragraph.  \n"
        sections = segment_article(text)
        self.assertEqual(len(sections), 1)
        self.assert_preserved(text, sections)

    def test_quotes(self):
        paragraphs = [
            "Prose.",
            "> Markdown quote\n> second line",
            "“Tiếng Việt.”",
            '"English."',
            "‘Single curly.’",
            "'Single straight.'",
            "«Guillemets.»",
            "“Unmatched quotation",
            "An inline “quotation” stays in prose.",
        ]
        text = "\n\n".join(paragraphs)
        sections = segment_article(text)
        self.assert_preserved(text, sections)
        self.assertEqual(
            [s.kind for s in sections], ["prose"] + ["quote"] * 6 + ["prose"]
        )
        self.assertEqual([s.text.strip() for s in sections[1:7]], paragraphs[1:7])

    def test_long_quote_is_intact(self):
        text = "“" + "long " * 1000 + "”"
        self.assertEqual(len(segment_article(text)), 1)
        self.assertEqual(segment_article(text)[0].kind, "quote")

    def test_uniform_text_uses_only_paragraph_cap(self):
        text = "\n\n".join(["same word " * 40] * 9)
        sections = segment_article(text)
        self.assert_preserved(text, sections)
        self.assertEqual(
            [len(s.text.strip().split("\n\n")) for s in sections], [4, 4, 1]
        )

    def test_topic_boundary_precedes_cap(self):
        for first, second in [
            ("apple orchard harvest fruit ", "engine wheel motor vehicle "),
            ("vườn cây trái quả ", "động cơ bánh xe "),
        ]:
            paragraphs = [first * 20] * 3 + [second * 20] * 3
            text = "\n\n".join(paragraphs)
            sections = segment_article(text)
            self.assert_preserved(text, sections)
            boundary = len("\n\n".join(paragraphs[:3])) + 2
            self.assertIn(boundary, [s.start for s in sections])

    def test_unicode_source_is_never_normalized(self):
        for form in ("NFC", "NFD"):
            text = unicodedata.normalize(
                form, "\n\n".join(["Kiểm thử tiếng Việt có dấu. " * 25] * 5)
            )
            self.assert_preserved(text, segment_article(text))


class HeadingTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Heading unit tests never make structure-classifier requests, even if
        # the developer's environment contains provider credentials.
        patcher = patch(
            "data_platform.digest.segmenter.texttiling.pipeline.detect_document_structure",
            side_effect=lambda text, **_: parse_document_structure(text),
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def model(self, headings):
        self.calls = []
        values = iter(headings)

        def respond(messages, info):
            self.calls.append(messages)
            return ModelResponse(
                parts=[
                    ToolCallPart(info.output_tools[0].name, {"heading": next(values)})
                ]
            )

        return FunctionModel(respond)

    async def test_markdown_document_output(self):
        source = (
            "# Original title\r\n\r\nBody with **formatting**.\r\n\r\n> Quotation.\n"
        )
        result = await texttiling_segement(
            source,
            model=self.model(["Overview", "Quotation"]),
            embedding_model=TestEmbeddingModel(),
            max_concurrency=1,
        )
        expected = (
            "# Original title\r\n\r\n## Overview\n\nBody with **formatting**.\r\n\r\n"
            "## Quotation\n\n> Quotation.\n"
        )
        self.assertEqual(result, expected)
        self.assertEqual(len(self.calls), 2)

    async def test_markdown_heading_escaping(self):
        result = await texttiling_segement(
            "Content.", model=self.model(["A *topic* <tag>"])
        )
        self.assertEqual(result, "## A \\*topic\\* \\<tag\\>\n\nContent.")

    async def test_blank_markdown_document(self):
        with patch(
            "data_platform.digest.segmenter.texttiling.heading_generator.Agent"
        ) as agent:
            self.assertEqual(await texttiling_segement(" \n "), "")
            agent.assert_not_called()

    async def test_mapping_and_model_instance(self):
        sections = segment_article("First.\n\n“Second.”")
        result = await generate_headings(
            sections,
            model=self.model([" First heading ", "Second heading"]),
            max_concurrency=1,
        )
        self.assertEqual(
            [s.heading for s in result], ["First heading", "Second heading"]
        )
        self.assertEqual([replace(s, heading=None) for s in result], sections)
        self.assertTrue(all(s.heading is None for s in sections))
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(
            render_markdown(result),
            "## First heading\n\nFirst.\n\n## Second heading\n\n“Second.”",
        )
        for call, section in zip(self.calls, sections):
            self.assertEqual(
                json.loads(call[0].parts[-1].content)["target_section"]["text"],
                section.text,
            )

    async def test_heading_rejects_line_separators(self):
        for ending in ("\n", "\r", "\r\n", "\u2028", "\u2029"):
            with self.assertRaises(ValueError):
                Heading(heading="Heading" + ending)

    async def test_two_validation_retries(self):
        result = await segment_with_headings(
            "Text.", model=self.model([" ", "Two\nlines", "Valid"])
        )
        self.assertEqual(result[0].heading, "Valid")
        self.assertEqual(len(self.calls), 3)

    async def test_exhaustion_reports_section(self):
        with self.assertRaises(HeadingGenerationError) as caught:
            await segment_with_headings(
                "Text.\n\n“Quote.”",
                model=self.model(["Good", "", "", ""]),
                max_concurrency=1,
            )
        self.assertEqual(caught.exception.section_index, 1)
        self.assertIsNotNone(caught.exception.__cause__)
        self.assertEqual(len(self.calls), 4)

    async def test_named_override_and_default(self):
        for name in (DEFAULT_MODEL, "openai:caller-selected"):
            with patch(
                "data_platform.digest.segmenter.texttiling.heading_generator.Agent",
                side_effect=RuntimeError("Unavailable"),
            ) as agent:
                with self.assertRaises(HeadingGenerationError):
                    if name == DEFAULT_MODEL:
                        await generate_headings(segment_article("Text."))
                    else:
                        await generate_headings(segment_article("Text."), model=name)
                self.assertEqual(agent.call_args.args[0], name)
                self.assertEqual(agent.call_count, 1)

    async def test_empty_never_constructs_agent(self):
        with patch(
            "data_platform.digest.segmenter.texttiling.heading_generator.Agent"
        ) as agent:
            self.assertEqual(await segment_with_headings("  "), [])
            agent.assert_not_called()


@unittest.skipUnless(
    os.getenv("RUN_LIVE_HEADINGS") == "1", "Opt-in live model smoke test"
)
class LiveHeadingTests(unittest.TestCase):
    def test_articles(self):
        for language in ("en", "vi"):
            text = (ROOT / f".data/article-{language}-01.md").read_text()
            sections = asyncio.run(
                segment_with_headings(
                    text, model=os.getenv("HEADING_MODEL", DEFAULT_MODEL)
                )
            )
            self.assertTrue(any(s.heading for s in sections))
            self.assertTrue(
                all(
                    s.heading is None
                    for s in sections
                    if s.kind in {"title", "metadata", "heading"}
                )
            )
            for i, section in enumerate(sections):
                print(
                    f"{language} [{i}] {section.heading}\n  {section.text.strip()[:100]}"
                )


if __name__ == "__main__":
    unittest.main()
