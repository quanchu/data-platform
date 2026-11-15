"""Contract and framework retry coverage without provider credentials."""

import json
import subprocess
import sys
import unittest
from unittest.mock import patch

from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import ModelResponse, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel

from data_platform.digest.segmenter.llm_e2e import (
    generate_block_headings,
    llm_segment,
    reconstruct_document,
    split_blocks,
)


class BlockTests(unittest.TestCase):
    def test_separators(self):
        self.assertEqual(split_blocks("a\r\n\r\nb\n\n\n\n"), ["a", "b", "", ""])
        self.assertEqual(split_blocks(""), [""])
        self.assertEqual(split_blocks("\n\na"), ["", "a"])
        self.assertEqual(split_blocks(" a \n\nb "), [" a ", "b "])

    def test_quotes(self):
        quote = "Mr. An said, “We build cars.\n\nWe export cars.”"
        blocks = split_blocks("Introduction.\n\n" + quote + "\n\nConclusion.")
        self.assertEqual(blocks, ["Introduction.", quote, "Conclusion."])
        repeated = "Mr. An said, “We build cars.” Mr. An said, “We export cars.”"
        blocks = split_blocks(repeated)
        self.assertEqual(sum("“" in block for block in blocks), 2)
        self.assertEqual("".join(blocks), repeated)

    def test_reconstruction(self):
        blocks = ["# Title", " text <SECTION_BOUNDARY> ", "", "Nội dung"]
        headings = ["", "## Company builds cars.", "", ""]
        self.assertEqual(
            reconstruct_document(blocks, headings),
            "# Title\n\n## Company builds cars.\n\n text <SECTION_BOUNDARY> \n\n\n\nNội dung",
        )
        self.assertEqual(blocks[1], " text <SECTION_BOUNDARY> ")

    def test_invalid_outputs(self):
        for heading in [
            "## ",
            "# Title",
            " ## Text",
            "## a\nb",
            "## a\rb",
            "## <SECTION_BOUNDARY>",
        ]:
            with self.subTest(heading=heading), self.assertRaises(ValueError):
                reconstruct_document(["body"], [heading])
        with self.assertRaises(ValueError):
            reconstruct_document(["body"], [])
        with self.assertRaises(ValueError):
            reconstruct_document(["  "], ["## A does B."])

    def test_compatibility_is_lazy(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; from data_platform.digest.segmenter import segment_article; "
                    "from data_platform.digest.segmenter.texttiling import segment_article as new; "
                    "assert segment_article is new; assert 'pydantic_ai' not in sys.modules"
                ),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


class AgentTests(unittest.IsolatedAsyncioTestCase):
    async def test_retry_feedback_and_model_override(self):
        calls = []

        def respond(messages, info):
            calls.append(messages)
            headings = ["bad"] if len(calls) == 1 else ["## Company builds cars."]
            return ModelResponse(
                parts=[ToolCallPart(info.output_tools[0].name, {"response": headings})]
            )

        output = await llm_segment("Company builds cars.", model=FunctionModel(respond))
        self.assertEqual(output, "## Company builds cars.\n\nCompany builds cars.")
        self.assertEqual(len(calls), 2)
        self.assertTrue(
            any(
                isinstance(part, ToolReturnPart)
                and part.outcome == "retried"
                and "Entry 0" in str(part.content)
                for message in calls[-1]
                for part in message.parts
            )
        )

    async def test_retry_exhaustion(self):
        calls = []

        def respond(messages, info):
            calls.append(messages)
            return ModelResponse(
                parts=[ToolCallPart(info.output_tools[0].name, {"response": []})]
            )

        with self.assertRaises(UnexpectedModelBehavior):
            await generate_block_headings(["body"], model=FunctionModel(respond))
        self.assertEqual(len(calls), 3)

    async def test_source_marker_and_empty_blocks(self):
        blocks = ["Title <SECTION_BOUNDARY>", "", "body"]

        def respond(messages, info):
            records = json.loads(messages[0].parts[-1].content)
            self.assertEqual([r["text"] for r in records], blocks)
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        info.output_tools[0].name,
                        {"response": ["", "", "## A does B."]},
                    )
                ]
            )

        headings = await generate_block_headings(blocks, model=FunctionModel(respond))
        self.assertEqual(headings, ["", "", "## A does B."])

    async def test_empty_skips_agent(self):
        with patch("data_platform.digest.segmenter.llm_e2e.pipeline.Agent") as agent:
            self.assertEqual(await generate_block_headings(["", " "]), ["", ""])
            self.assertEqual(await generate_block_headings([]), [])
            self.assertEqual(await llm_segment(""), "")
            agent.assert_not_called()
