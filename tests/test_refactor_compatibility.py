"""Public and persisted contracts captured before the structural refactor."""

import hashlib
import inspect
import json
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from test_claim_store import Embeddings
from test_claimify import models
from test_openie import Extractor, Resolver

from data_platform.digest import claimify, openie
from data_platform.digest.claim_store import ClaimStore
from data_platform.digest.segmenter import texttiling
from data_platform.digest.source import parse_document

SOURCE = "# Báo cáo\n\n## Acme\n\nAcme plans two factories in 2028.\n\n## Results\n\nRevenue grew.\n"
FIXTURE = Path(__file__).parent / "fixtures/refactor/contracts.json"


async def contracts():
    decision, generation, calls = models()
    result = await claimify.extract_claims(
        SOURCE, decision_model=decision, generation_model=generation, max_concurrency=1
    )
    tuples = await openie.extract_claims(
        SOURCE, resolver=Resolver(), extractor=Extractor(), max_concurrency=1
    )
    with tempfile.TemporaryDirectory() as directory:
        store = ClaimStore(directory)
        records = store.store_unverified(result, source="article.md")
        saved = [r.model_dump(mode="json", exclude={"created_at"}) for r in records]
    return {
        "document": parse_document(SOURCE).model_dump(mode="json"),
        "claimify": result.model_dump(mode="json"),
        "openie": tuples.model_dump(mode="json"),
        "requests": calls,
        "records": saved,
        "schemas": {
            "claimify": claimify.ExtractionResult.model_json_schema(),
            "openie": openie.ExtractionResult.model_json_schema(),
        },
        "exports": {m.__name__: m.__all__ for m in (claimify, openie, texttiling)},
        "signatures": {
            name: str(inspect.signature(fn))
            for name, fn in {
                "claimify": claimify.extract_claims,
                "openie": openie.extract_claims,
                "store": ClaimStore,
                "retrieve": ClaimStore.retrieve,
                "check": ClaimStore.check,
                "store_unverified": ClaimStore.store_unverified,
                "segment_article": texttiling.segment_article,
                "texttiling_segement": texttiling.texttiling_segement,
            }.items()
        },
    }


class CompatibilityTests(unittest.IsolatedAsyncioTestCase):
    async def test_captured_contracts(self):
        actual = json.loads(json.dumps(await contracts()))
        self.assertEqual(actual, json.loads(FIXTURE.read_text(encoding="utf-8")))

    async def test_existing_json_and_sqlite_cache_need_no_migration(self):
        records = json.loads(FIXTURE.read_text(encoding="utf-8"))["records"]
        model = Embeddings()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "verified").mkdir()
            for record in records:
                record["created_at"] = "2026-01-01T00:00:00Z"
                (root / "verified" / f"{record['id']}.json").write_text(
                    json.dumps(record), encoding="utf-8"
                )
            with closing(sqlite3.connect(root / "index.sqlite")) as connection:
                with connection:
                    connection.execute(
                        "CREATE TABLE embeddings (id TEXT PRIMARY KEY, text_hash TEXT NOT NULL, model TEXT NOT NULL, vector TEXT NOT NULL)"
                    )
                    connection.executemany(
                        "INSERT INTO embeddings VALUES (?, ?, ?, ?)",
                        [
                            (
                                r["id"],
                                hashlib.sha256(r["text"].encode()).hexdigest(),
                                "test:deterministic",
                                "[1.0, 0.0]",
                            )
                            for r in records
                        ],
                    )
            store = ClaimStore(root, embedding_model=model)
            sync = await store.sync_index()
            self.assertEqual(sync.embedded_count, 0)
            self.assertEqual(model.calls, [])
            self.assertEqual(sync.verified_count, 2)
            matches = await store.retrieve("revenue")
            self.assertEqual(
                [m.claim.id for m in matches], sorted(r["id"] for r in records)
            )
            store.embedding_model = Embeddings("changed-model")
            self.assertEqual((await store.sync_index()).embedded_count, 2)

    def test_source_parsing_does_not_import_segmentation_or_providers(self):
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; from data_platform.digest.source import parse_document; "
                "parse_document('# T\\n## S\\nA fact.'); "
                "assert 'pydantic_ai' not in sys.modules; "
                "assert 'data_platform.digest.segmenter' not in sys.modules",
            ],
            check=True,
        )

    def test_prompts_unchanged(self):
        root = Path(__file__).resolve().parents[1]
        expected = json.loads(FIXTURE.with_name("prompts.json").read_text())
        for name, digest in expected.items():
            self.assertEqual(
                hashlib.sha256((root / name).read_bytes()).hexdigest(), digest
            )
