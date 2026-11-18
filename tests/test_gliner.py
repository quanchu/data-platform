"""Offline contracts for the GLiNER logic extracted from its notebook."""

import contextlib
import io
import json
import re
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from data_platform.digest.gliner import extract_entities
from data_platform.digest.gliner.tagger import _iter_chunks


class FakeModel:
    def __init__(self, max_len=200):
        self.config = types.SimpleNamespace(max_len=max_len)
        self.data_processor = types.SimpleNamespace(
            words_splitter=lambda text: (
                (m.group(), m.start(), m.end()) for m in re.finditer(r"\S+", text)
            )
        )
        self.calls = []

    def predict_entities(self, text, labels, *, threshold):
        self.calls.append((text, labels, threshold))
        output = []
        for term, label, score in [
            ("Acme", "company name", 0.70),
            ("Acme Inc", "company name", 0.95),
            ("Lan", "person name", 0.80),
        ]:
            for match in re.finditer(re.escape(term), text):
                output.append(
                    dict(
                        text=term,
                        label=label,
                        score=score + min(len(self.calls), 4) * 0.001,
                        start=match.start(),
                        end=match.end(),
                    )
                )
        return output

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def to(self, device):
        return self

    def eval(self):
        return self


class TaggingTests(unittest.TestCase):
    def setUp(self):
        stub = types.SimpleNamespace(inference_mode=contextlib.nullcontext)
        patcher = patch.dict(sys.modules, {"torch": stub})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_chunks_use_model_limit_overlap_and_original_offsets(self):
        text = "  Một Acme Lan bốn năm sáu  "
        chunks = list(_iter_chunks(text, FakeModel(max_len=3), 200, 1))
        self.assertEqual(
            chunks, [(0, "  Một Acme Lan"), (11, "Lan bốn năm"), (19, "năm sáu  ")]
        )
        for offset, chunk in chunks:
            self.assertEqual(chunk, text[offset : offset + len(chunk)])

    def test_duplicate_predictions_keep_highest_score(self):
        text = "Một Acme Lan bốn năm"
        model = FakeModel(max_len=3)
        result = extract_entities(
            text, model=model, labels=["company name", "person name"], chunk_overlap=2
        )
        self.assertEqual([e["text"] for e in result], ["Acme", "Lan"])
        self.assertAlmostEqual(result[0]["score"], 0.702)
        self.assertAlmostEqual(result[1]["score"], 0.803)
        for entity in result:
            self.assertEqual(entity["text"], text[entity["start"] : entity["end"]])
        self.assertTrue(all(threshold == 0.5 for _, _, threshold in model.calls))

    def test_overlap_selection_and_configuration(self):
        model = FakeModel()
        result = extract_entities(
            "Acme Inc met Lan.", model=model, labels=["company name"], threshold=0.6
        )
        self.assertEqual([e["text"] for e in result], ["Acme Inc", "Lan"])
        self.assertEqual(model.calls[0][1:], (["company name"], 0.6))

    def test_empty_invalid_overlap_and_invalid_prediction(self):
        model = FakeModel()
        for text in ("", "   "):
            self.assertEqual(extract_entities(text, model=model, labels=[]), [])
        self.assertEqual(model.calls, [])
        with self.assertRaises(ValueError):
            extract_entities(
                "text", model=model, labels=[], chunk_words=2, chunk_overlap=2
            )
        with patch.object(
            model,
            "predict_entities",
            return_value=[dict(text="wrong", label="x", score=0.5, start=0, end=4)],
        ):
            with self.assertRaises(AssertionError):
                extract_entities("text", model=model, labels=[])

    def test_import_does_not_load_model_libraries(self):
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import data_platform.digest.gliner; assert 'torch' not in sys.modules; assert 'gliner' not in sys.modules",
            ],
            check=True,
        )

    def test_notebook_with_injected_model_and_temporary_exports(self):
        root = Path(__file__).resolve().parents[1]
        notebook = json.loads(
            (root / "notebooks/gliner_entity_tagging.ipynb").read_text()
        )
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(
                sys.modules, {"gliner": types.SimpleNamespace(GLiNER=FakeModel)}
            ),
        ):
            namespace = {}
            with contextlib.redirect_stdout(io.StringIO()):
                for cell in notebook["cells"]:
                    if cell["cell_type"] != "code":
                        continue
                    source = "".join(cell["source"])
                    if source.startswith("output_dir.mkdir"):
                        namespace["output_dir"] = Path(directory)
                    exec(compile(source, "<gliner notebook>", "exec"), namespace)
            exports = list(Path(directory).glob("*.entities.json"))
            self.assertEqual(len(exports), len(namespace["input_paths"]))
            for path in exports:
                data = json.loads(path.read_text())
                for e in data["entities"]:
                    self.assertEqual(data["text"][e["start"] : e["end"]], e["text"])
