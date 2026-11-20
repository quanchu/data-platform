"""Notebook defaults and DataFrame reviews run offline in a fresh namespace."""

import ast
import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import nbformat
import pandas as pd

from support import models, Resolver, Extractor

ROOT = Path(__file__).resolve().parents[1]
ARTICLE = "# Report\n\n## Results\n\nAcme plans two factories.\n\nRevenue grew.\n"


async def run_notebook(name, directory, *, live=False, export=False):
    notebook = nbformat.read(ROOT / "notebooks" / name, as_version=4)
    nbformat.validate(notebook)
    root = Path(directory)
    (root / ".data/segmented").mkdir(parents=True, exist_ok=True)
    for path in (
        ".data/article-en-01.md",
        ".data/article-vi-01.md",
        ".data/segmented/article-en-01.md",
    ):
        (root / path).write_text(ARTICLE, encoding="utf-8")
    namespace = {}
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        assert not cell.outputs and cell.execution_count is None
        source = cell.source
        if live:
            source = source.replace("run_live = False", "run_live = True")
        if export:
            source = source.replace("save_output = False", "save_output = True")
        compiled = compile(
            source, f"{name}:{index}", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT
        )
        with patch.object(sys, "path", [str(ROOT / "notebooks"), *sys.path]):
            output = eval(compiled, namespace)
            if inspect.isawaitable(output):
                await output
        if "project_root =" in source:
            namespace["project_root"] = root
            namespace["display"] = lambda *args: None
    return namespace


class NotebookTests(unittest.IsolatedAsyncioTestCase):
    async def test_every_notebook_runs_with_model_calls_disabled(self):
        from pydantic_ai import Agent

        for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
            with (
                self.subTest(notebook=path.name),
                tempfile.TemporaryDirectory() as directory,
            ):
                with patch.object(
                    Agent, "run", side_effect=AssertionError("Unexpected model call")
                ):
                    namespace = await run_notebook(path.name, directory)
                frames = [
                    value
                    for value in namespace.values()
                    if isinstance(value, pd.DataFrame)
                ]
                self.assertTrue(frames, "The review should use DataFrames")
                self.assertTrue(all(len(frame.columns) > 0 for frame in frames))
                self.assertEqual(list(Path(directory).rglob("*.csv")), [])
                self.assertEqual(list(Path(directory).rglob("*.json")), [])

    async def test_claimify_and_openie_reviews_and_exports(self):
        from data_platform.digest import claimify, openie

        original_claimify = claimify.extract_claims
        original_openie = openie.extract_claims
        decision, generation, _ = models()

        async def extract_claimify(source, **kwargs):
            return await original_claimify(
                source, decision_model=decision, generation_model=generation
            )

        async def extract_openie(source, **kwargs):
            return await original_openie(
                source, resolver=Resolver(), extractor=Extractor()
            )

        for name in ("claimify.ipynb", "openie.ipynb"):
            with (
                self.subTest(notebook=name),
                tempfile.TemporaryDirectory() as directory,
            ):
                with (
                    patch.object(
                        claimify, "extract_claims", side_effect=extract_claimify
                    ),
                    patch.object(openie, "extract_claims", side_effect=extract_openie),
                ):
                    namespace = await run_notebook(
                        name, directory, live=True, export=True
                    )
                frame = namespace["claims_df"]
                self.assertFalse(frame.empty)
                for row in frame.itertuples():
                    self.assertEqual(
                        row.source, namespace["markdown_doc"][row.start : row.end]
                    )
                self.assertTrue(list(Path(directory).rglob("*.csv")))
                saved = json.loads(next(Path(directory).rglob("*.json")).read_text())
                self.assertEqual(saved["source_markdown"], namespace["markdown_doc"])

    async def test_empty_extraction_retains_review_columns(self):
        from data_platform.digest import claimify

        original = claimify.extract_claims
        decision, generation, _ = models({"verdict": "none"})

        async def empty(source, **kwargs):
            return await original(
                source, decision_model=decision, generation_model=generation
            )

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(claimify, "extract_claims", side_effect=empty),
        ):
            namespace = await run_notebook("claimify.ipynb", directory, live=True)
        self.assertTrue(namespace["claims_df"].empty)
        self.assertIn("source", namespace["claims_df"].columns)
        self.assertFalse(namespace["outcomes_df"].empty)
