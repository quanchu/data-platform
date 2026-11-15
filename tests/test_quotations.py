import json
import unicodedata
import unittest
from dataclasses import replace
from importlib.util import find_spec
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from markdown_it import MarkdownIt
from pydantic_ai.embeddings import TestEmbeddingModel

from data_platform.digest.segmenter.texttiling import (
    generate_headings,
    parse_document_structure,
    render_markdown,
    segment_article,
    segment_article_semantic,
)
from data_platform.digest.segmenter.texttiling.quotations import quotation_spans

ROOT = Path(__file__).resolve().parents[1]


class QuotationTests(unittest.IsolatedAsyncioTestCase):
    def assert_preserved(self, text, sections):
        self.assertEqual("".join(s.text for s in sections), text)
        cursor = 0
        for section in sections:
            self.assertEqual(section.start, cursor)
            self.assertEqual(section.text, text[section.start : section.end])
            cursor = section.end
        self.assertEqual(cursor, len(text))

    async def assert_quotes(self, text, quotes):
        results = [
            segment_article(text),
            await segment_article_semantic(text, embedding_model=TestEmbeddingModel()),
        ]
        if find_spec("sklearn"):
            results.append(segment_article(text, vectorizer="tfidf"))
        for sections in results:
            self.assert_preserved(text, sections)
            self.assertEqual(
                [s.text.strip() for s in sections if s.kind == "quote"], quotes
            )
        return results[0]

    async def test_same_speaker_same_paragraph_and_later(self):
        text = "Mr. Smith said: “First.” Mr. Smith added: “Second.”\n\nNarration.\n\nMr. Smith said: “Third.”"
        await self.assert_quotes(
            text,
            [
                "Mr. Smith said: “First.”",
                "Mr. Smith added: “Second.”",
                "Mr. Smith said: “Third.”",
            ],
        )

    async def test_suffix_attribution_and_narration(self):
        text = "Narration. “First,” said Mr. Smith, Chief Executive Officer. More narration. “Second,” Ms. Jones said."
        sections = await self.assert_quotes(
            text,
            [
                "“First,” said Mr. Smith, Chief Executive Officer.",
                "“Second,” Ms. Jones said.",
            ],
        )
        self.assertEqual(
            [s.text.strip() for s in sections if s.kind == "prose"],
            ["Narration.", "More narration."],
        )

    async def test_interrupted_utterance(self):
        text = "“We will invest,” she said, “next year.” “We will hire,” she added, “in June.”"
        await self.assert_quotes(
            text,
            [
                "“We will invest,” she said, “next year.”",
                "“We will hire,” she added, “in June.”",
            ],
        )

    async def test_complete_utterances_not_merged(self):
        await self.assert_quotes(
            "“First.” Smith said, “Second.”", ["“First.” Smith said,", "“Second.”"]
        )
        await self.assert_quotes("“First.” “Second.”", ["“First.”", "“Second.”"])

    async def test_vietnamese_and_straight_quotes(self):
        text = "Ông An nói: “Chúng tôi đầu tư.” Bà Bình cho biết: “Dự án sẽ mở rộng.” Ông An nói thêm: “Chúng tôi sẽ tuyển người.”"
        await self.assert_quotes(
            text,
            [
                "Ông An nói: “Chúng tôi đầu tư.”",
                "Bà Bình cho biết: “Dự án sẽ mở rộng.”",
                "Ông An nói thêm: “Chúng tôi sẽ tuyển người.”",
            ],
        )
        await self.assert_quotes(
            'Mr. Smith said: "We don\'t stop."', ['Mr. Smith said: "We don\'t stop."']
        )
        await self.assert_quotes(
            "Smith said: 'We invest.'", ["Smith said: 'We invest.'"]
        )

    async def test_decomposed_vietnamese_attribution(self):
        text = unicodedata.normalize("NFD", "Ông An nói: “Chúng tôi sẽ đầu tư.”")
        await self.assert_quotes(text, [text])

    async def test_nested_and_multparagraph(self):
        text = (
            "Smith said: “The team calls it ‘Project One’.\n\nThe team will build it.”"
        )
        await self.assert_quotes(text, [text])
        text = "Smith said:\n\n“First paragraph.\n\nSecond paragraph.”"
        await self.assert_quotes(text, [text])
        text = "Smith said:\n\n> First paragraph.\n>\n> Second paragraph."
        await self.assert_quotes(text, [text])

    async def test_quoted_terms_unmatched_and_code(self):
        for text in (
            "A “green” car appeared.",
            "Smith said: “No closing mark.",
            'An unmatched "mark.',
            "The drivers' vehicles don't stop.",
            "`Smith said: “Code.”`",
            "```\nSmith said: “Code.”\n```",
        ):
            await self.assert_quotes(text, [])

    async def test_recover_quote_after_unmatched_curly_mark(self):
        text = "Unmatched “opening.\n\nSmith said: “A complete quotation.”"
        await self.assert_quotes(text, ["Smith said: “A complete quotation.”"])

    async def test_heading_candidates_exclude_attributed_quotes(self):
        source = "# Title\n\nSmith said: “A claim.”\n\n“Another claim,” Smith said.\n\nABOUT THE COMPANY"
        candidates = parse_document_structure(source).candidates
        self.assertEqual([c.text.strip() for c in candidates], ["ABOUT THE COMPANY"])

    async def test_quotes_excluded_from_similarity_windows(self):
        text = "Before.\n\nSmith said: “Quote.”\n\nAfter."
        with patch(
            "data_platform.digest.segmenter.texttiling.semantic_segmenter._make_embedder"
        ) as make:
            sections = await segment_article_semantic(text)
            make.assert_not_called()
        self.assertEqual([s.kind for s in sections], ["prose", "quote", "prose"])
        from data_platform.digest.segmenter.texttiling.texttiling_segmenter import (
            _fit_tfidf,
        )

        if find_spec("sklearn"):
            fit = _fit_tfidf(text)
            self.assertNotIn("quote", fit.vocabulary_)
            self.assertNotIn("smith", fit.vocabulary_)

    async def test_real_english_article(self):
        text = (ROOT / ".data/article-en-01.md").read_text()
        quotes = [text[q.start : q.end].strip() for q in quotation_spans(text)]
        self.assertEqual(len(quotes), 3)
        for quote, person in zip(
            quotes, ("Nguyen Viet Quang", "Dan Levy", "Peter Loxton"), strict=True
        ):
            self.assertIn(person, quote)
        await self.assert_quotes(text, quotes)

    async def test_quote_heading_requests_and_rendering(self):
        text = "# Title\n\nIntro. Smith said: “First.” Smith said: “Second.” End."
        sections = segment_article(text)
        with patch(
            "data_platform.digest.segmenter.texttiling.heading_generator.Agent"
        ) as agent:
            agent.return_value.run = AsyncMock(
                return_value=SimpleNamespace(
                    output=SimpleNamespace(heading="Mr. Smith reports a claim.")
                )
            )
            headed = await generate_headings(sections, article_text=text)
            requests = [
                json.loads(call.args[0])
                for call in agent.return_value.run.call_args_list
            ]
        self.assertEqual(
            [
                r["target_section"]["text"].strip()
                for r in requests
                if r["target_section"]["kind"] == "quote"
            ],
            ["Smith said: “First.”", "Smith said: “Second.”"],
        )
        result = render_markdown(headed)
        headings = [
            t.tag for t in MarkdownIt().parse(result) if t.type == "heading_open"
        ]
        self.assertEqual(headings, ["h1", "h2", "h2", "h2", "h2"])
        self.assert_preserved(text, sections)
        self.assertEqual([replace(s, heading=None) for s in headed], sections)

    async def test_fragment_paragraph_cap(self):
        text = (
            "\n\n".join(["Uniform text. " * 30] * 3)
            + " Smith said: “One.” More body. "
            + "\n\n".join(["Uniform text. " * 30] * 5)
        )
        sections = segment_article(text)
        self.assert_preserved(text, sections)
        self.assertEqual(len([s for s in sections if s.kind == "quote"]), 1)
        for section in sections:
            if section.kind == "prose":
                self.assertLessEqual(len(section.text.strip().split("\n\n")), 4)
