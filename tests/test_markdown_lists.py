"""Lists are source-preserving units, never topic-score observations."""

import unittest
from unittest.mock import patch

from data_platform.digest._text.markdown import list_spans, parse_blocks
from data_platform.digest.segmenter.llm_e2e import split_blocks
from data_platform.digest.segmenter.texttiling import (
    segment_article,
    segment_article_semantic,
)


class ListTests(unittest.IsolatedAsyncioTestCase):
    async def test_lists_are_atomic_and_need_no_embeddings(self):
        for newline in ("\n", "\r\n"):
            for listing in (
                "- Solar revenue rose.\n- Dinosaurs disappeared.\n- 😀 Hà Nội opens a park.",
                "1. Solar revenue rose.\n\n2. Dinosaurs disappeared.\n\n   - Nested child.\n\n   Continuation paragraph.",
                "- Item one.\n\n  > A quotation.\n\n  ```text\n  - Not another list\n  ```\n\n- Item two.",
            ):
                source = (
                    "# Report\n\nOur updates:\n\n"
                    + listing
                    + "\n\nA final paragraph.\n"
                ).replace("\n", newline)
                with self.subTest(source=source):
                    sections = segment_article(source)
                    self.assertEqual("".join(s.text for s in sections), source)
                    lists = [s for s in sections if s.kind == "list"]
                    self.assertEqual(len(lists), 1)
                    self.assertTrue(lists[0].text.startswith("Our updates:"))
                    for section in sections:
                        self.assertEqual(
                            section.text, source[section.start : section.end]
                        )
                    with patch(
                        "data_platform.digest.segmenter.texttiling.semantic_segmenter._make_embedder"
                    ) as make:
                        semantic = await segment_article_semantic(source)
                        make.assert_not_called()
                    self.assertEqual(semantic, sections)
                    protected = source[slice(*list_spans(source)[0])]
                    self.assertIn(protected, split_blocks(source))

    def test_list_items_do_not_change_prose_scores(self):
        prose = "\n\n".join(["Solar energy revenue increased in the year. " * 12] * 3)
        source = (
            prose
            + "\n\n- ALIEN_LIST_TOKEN volcanoes.\n- ALIEN_LIST_TOKEN dinosaurs.\n\n"
            + prose
        )
        from data_platform.digest.segmenter.texttiling import (
            texttiling_segmenter as lexical,
        )

        original = lexical._topic_boundaries
        observations = []

        def record(text, paragraphs, tfidf=None):
            observations.append(text[paragraphs[0][0] : paragraphs[-1][1]])
            return original(text, paragraphs, tfidf)

        with patch.object(lexical, "_topic_boundaries", side_effect=record):
            sections = lexical.segment_article(source)
        self.assertEqual(len(observations), 2)
        self.assertTrue(all("ALIEN_LIST_TOKEN" not in run for run in observations))
        self.assertEqual("".join(s.text for s in sections), source)

    def test_nested_blocks_retain_original_offsets(self):
        source = "- Parent\n  - Child\n\n- Sibling\n"
        block = parse_blocks(source)[0]
        self.assertEqual(block.kind, "bullet_list")
        self.assertEqual(source[block.start : block.end], source)
        self.assertEqual(len(block.children), 2)

    def test_no_intro_attachment_across_heading(self):
        source = "An introduction:\n\n## Items\n\n- A\n- B\n"
        start, end = list_spans(source)[0]
        self.assertEqual(source[start:end], "- A\n- B\n")

    async def test_semantic_requests_exclude_lists_and_introductions(self):
        from pydantic_ai.embeddings import TestEmbeddingModel
        from data_platform.digest._events import observe

        model = TestEmbeddingModel()
        source = (
            "Energy grew.\n\nSolar panels arrived.\n\nInventory:\n\n"
            "- ALIEN_LIST_TOKEN\n- Another unrelated item.\n\n"
            "Buses arrived.\n\nDrivers trained.\n"
        )
        observed = []
        with observe(observed.append):
            sections = await segment_article_semantic(source, embedding_model=model)
        embedded = [
            source[span["start"] : span["end"]]
            for event in observed
            if event["stage"] == "embedding_batch" and event["status"] == "running"
            for span in event["spans"]
        ]
        self.assertTrue(embedded)
        self.assertTrue(
            all(
                "ALIEN_LIST_TOKEN" not in text and "Inventory" not in text
                for text in embedded
            )
        )
        self.assertEqual("".join(section.text for section in sections), source)

    def test_tfidf_training_excludes_lists(self):
        from importlib.util import find_spec

        if find_spec("sklearn") is None:
            self.skipTest("TF-IDF requires the offline extra")
        from data_platform.digest.segmenter.texttiling.texttiling_segmenter import (
            _fit_tfidf,
        )

        source = "Solar panels arrived.\n\nInventory:\n\n- alienlisttoken\n- dinosaurtoken\n\nBuses arrived."
        vectorizer = _fit_tfidf(source)
        self.assertNotIn("alienlisttoken", vectorizer.vocabulary_)
        self.assertNotIn("dinosaurtoken", vectorizer.vocabulary_)
        self.assertNotIn("inventory", vectorizer.vocabulary_)

    async def test_heading_does_not_split_long_list(self):
        from pydantic_ai.messages import ModelResponse, ToolCallPart
        from pydantic_ai.models.function import FunctionModel
        from data_platform.digest.segmenter.texttiling import (
            generate_headings,
            render_markdown,
        )

        source = (
            "# T\n\nInventory:\n\n"
            + "\n\n".join(f"- Item {i}." for i in range(7))
            + "\n"
        )
        sections = segment_article(source)

        def respond(messages, info):
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        info.output_tools[0].name,
                        {"heading": "The inventory lists seven items."},
                    )
                ]
            )

        headed = await generate_headings(sections, model=FunctionModel(respond))
        rendered = render_markdown(headed)
        self.assertEqual(len([s for s in headed if s.kind == "list"]), 1)
        self.assertEqual(rendered.count("## The inventory"), 1)
        self.assertIn(source[source.index("Inventory:") :], rendered)
