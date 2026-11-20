"""Live demo contracts with injected models and no external requests."""

import asyncio
import importlib.util
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from support import Extractor, Resolver, models

from data_platform.demo.models import RunRequest
from data_platform.demo.runs import RunManager, blocking_call
from data_platform.demo.tasks import Tasks
from data_platform.digest._events import emit

SOURCE = "# Report\n\n## Results\n\n😀 Acme plans two factories.\n"


class RunTests(unittest.IsolatedAsyncioTestCase):
    async def test_concurrent_events_keep_context_and_order(self):
        async def execute(request):
            async def child(index):
                await asyncio.sleep(0)
                emit("child", spans=[(index, index + 1)], index=index)

            await asyncio.gather(*(child(i) for i in range(3)))
            return {"done": True}

        manager = RunManager(execute)
        run = manager.start(RunRequest(task="segmentation", source=SOURCE))
        with self.assertRaises(RuntimeError):
            manager.start(run.request)
        await run.worker
        self.assertEqual(run.status, "completed")
        self.assertEqual([event["sequence"] for event in run.events], list(range(5)))
        self.assertTrue(
            all(event["document_id"] == run.document_id for event in run.events)
        )
        self.assertEqual(
            [e["data"]["index"] for e in run.events if e["stage"] == "child"], [0, 1, 2]
        )
        emit("outside")
        self.assertEqual(len(run.events), 5)

    async def test_cancellation_waits_for_local_inference(self):
        started, release = threading.Event(), threading.Event()

        def inference():
            started.set()
            release.wait(timeout=5)
            emit("inference_finished")

        async def execute(request):
            await blocking_call(inference)
            return {}

        manager = RunManager(execute)
        run = manager.start(RunRequest(task="entities", source=SOURCE))
        self.addCleanup(release.set)
        await asyncio.to_thread(started.wait, 2)
        cancelling = asyncio.create_task(manager.cancel(run))
        await asyncio.sleep(0.01)
        self.assertEqual(run.status, "cancelling")
        with self.assertRaises(RuntimeError):
            manager.start(run.request)
        release.set()
        await cancelling
        self.assertEqual(run.status, "cancelled")
        self.assertEqual(run.events[-1]["status"], "cancelled")

    async def test_immediate_cancel_and_error_releases_slot(self):
        async def execute(request):
            raise ValueError("fixture failure")

        manager = RunManager(execute)
        run = manager.start(RunRequest(task="segmentation", source=SOURCE))
        await manager.cancel(run)
        self.assertEqual(run.status, "cancelled")
        failed = manager.start(run.request)
        await failed.worker
        self.assertEqual(failed.status, "failed")
        self.assertEqual(failed.error, "fixture failure")

    async def test_real_extraction_handoffs_and_store_is_unverified(self):
        with tempfile.TemporaryDirectory() as directory:
            tasks = Tasks(Path(directory))
            manager = RunManager(tasks)
            tasks.manager = manager
            from data_platform.digest import claimify, openie

            original_claimify, original_openie = (
                claimify.extract_claims,
                openie.extract_claims,
            )
            decision, generation, _ = models()

            async def claimify_fixture(source, **kwargs):
                return await original_claimify(
                    source, decision_model=decision, generation_model=generation
                )

            async def openie_fixture(source, **kwargs):
                return await original_openie(
                    source, resolver=Resolver(), extractor=Extractor()
                )

            with patch.object(claimify, "extract_claims", side_effect=claimify_fixture):
                extraction = manager.start(RunRequest(task="claimify", source=SOURCE))
                await extraction.worker
            self.assertEqual(extraction.status, "completed", extraction.error)
            self.assertIn("selection", [event["stage"] for event in extraction.events])
            self.assertIn("coverage", [event["stage"] for event in extraction.events])
            request = tasks.validate(
                RunRequest(task="claim_store", parent_run_id=extraction.id)
            )
            saved = manager.start(request)
            await saved.worker
            self.assertEqual(saved.status, "completed", saved.error)
            self.assertEqual(
                len(list((Path(directory) / "verified").glob("*.json"))), 0
            )
            self.assertEqual(len(saved.result["records"]), 1)
            self.assertEqual(saved.document_id, extraction.document_id)
            with patch.object(openie, "extract_claims", side_effect=openie_fixture):
                relations = manager.start(RunRequest(task="openie", source=SOURCE))
                await relations.worker
            self.assertEqual(relations.status, "completed", relations.error)
            self.assertIn("resolution", [event["stage"] for event in relations.events])
            for event in extraction.events:
                for span in event["spans"]:
                    self.assertLessEqual(span["end"], len(SOURCE))


@unittest.skipUnless(
    importlib.util.find_spec("fastapi"), "Install the demo extra for HTTP tests"
)
class HTTPTests(unittest.TestCase):
    def test_live_segmentation_stream_and_validation(self):
        from fastapi.testclient import TestClient

        from data_platform.demo.app import create_app

        with (
            tempfile.TemporaryDirectory() as directory,
            TestClient(create_app(store_root=Path(directory))) as client,
        ):
            self.assertEqual(client.get("/").status_code, 200)
            self.assertIn("Follow the evidence", client.get("/").text)
            examples = client.get("/api/examples").json()
            source = next(e["source"] for e in examples if e["name"] == "list-aware.md")
            response = client.post(
                "/api/runs", json={"task": "segmentation", "source": source}
            )
            self.assertEqual(response.status_code, 201, response.text)
            run_id = response.json()["id"]
            stream = client.get(f"/api/runs/{run_id}/events")
            events = [
                json.loads(line[6:])
                for line in stream.text.splitlines()
                if line.startswith("data: ")
            ]
            self.assertEqual(events[-1]["status"], "completed")
            self.assertIn("markdown_structure", [e["stage"] for e in events])
            result = client.get(f"/api/runs/{run_id}").json()["result"]
            self.assertEqual("".join(s["text"] for s in result["sections"]), source)
            self.assertEqual(
                len([s for s in result["sections"] if s["kind"] == "list"]), 1
            )
            resumed = client.get(
                f"/api/runs/{run_id}/events",
                headers={"Last-Event-ID": str(events[-2]["sequence"])},
            )
            self.assertEqual(resumed.text.count("data: "), 1)
            self.assertEqual(
                client.post(
                    "/api/runs", json={"task": "claimify", "source": "No headings"}
                ).status_code,
                422,
            )
            self.assertEqual(
                client.post(
                    "/api/runs", json={"task": "claim_store", "source": SOURCE}
                ).status_code,
                422,
            )
            self.assertEqual(
                client.post(
                    "/api/runs",
                    json={"task": "segmentation", "source": source},
                    headers={"Origin": "https://example.com"},
                ).status_code,
                403,
            )
            self.assertEqual(client.get("/api/runs/missing").status_code, 404)

    def test_failure_is_streamed_and_results_remain_available(self):
        from fastapi.testclient import TestClient

        from data_platform.demo.app import create_app

        async def fail(request):
            emit("selection", status="running", spans=[(0, 1)])
            raise ValueError("fixture failure")

        with TestClient(create_app(executor=fail)) as client:
            run_id = client.post(
                "/api/runs", json={"task": "segmentation", "source": SOURCE}
            ).json()["id"]
            events = client.get(f"/api/runs/{run_id}/events").text
            self.assertIn('"status": "failed"', events)
            self.assertEqual(
                client.get(f"/api/runs/{run_id}").json()["error"], "fixture failure"
            )


class TaskAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_entity_adapter_observes_thread_events(self):
        with tempfile.TemporaryDirectory() as directory:
            tasks = Tasks(Path(directory))
            manager = RunManager(tasks)
            tasks.manager = manager

            def infer(request):
                emit("entity_chunk", spans=[(0, 1)])
                return [
                    {
                        "text": "😀",
                        "start": 0,
                        "end": 1,
                        "label": "symbol",
                        "score": 1.0,
                    }
                ]

            with patch.object(tasks, "_tag_entities", side_effect=infer):
                run = manager.start(RunRequest(task="entities", source=SOURCE))
                await run.worker
            self.assertEqual(run.status, "completed")
            self.assertEqual(run.result["entities"][0]["text"], "😀")
            self.assertEqual(
                [event["stage"] for event in run.events], ["run", "entity_chunk", "run"]
            )

    async def test_live_comparison_adapter_with_injected_models(self):
        from pydantic_ai.messages import ModelResponse, ToolCallPart
        from pydantic_ai.models.function import FunctionModel
        from support import Embeddings

        from data_platform.digest import claim_store, claimify

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tasks = Tasks(root)
            manager = RunManager(tasks)
            tasks.manager = manager
            decision, generation, _ = models()
            extracted = await claimify.extract_claims(
                SOURCE, decision_model=decision, generation_model=generation
            )

            async def prepared(request):
                return extracted.model_dump(mode="json")

            manager.execute = prepared
            parent = manager.start(RunRequest(task="claimify", source=SOURCE))
            await parent.worker
            manager.execute = tasks
            model = FunctionModel(
                lambda messages, info: ModelResponse(
                    parts=[
                        ToolCallPart(
                            info.output_tools[0].name, {"verdict": "equivalent"}
                        )
                    ]
                )
            )
            store = claim_store.ClaimStore(
                root, embedding_model=Embeddings(), comparison_model=model
            )
            approved = store.store_unverified(extracted, source="previous.md")[0]
            (root / "unverified" / f"{approved.id}.json").rename(
                root / "verified" / f"{approved.id}.json"
            )
            with patch.object(claim_store, "ClaimStore", return_value=store):
                request = tasks.validate(
                    RunRequest(
                        task="claim_store",
                        parent_run_id=parent.id,
                        store_action="compare",
                    )
                )
                run = manager.start(request)
                await run.worker
            self.assertEqual(run.status, "completed", run.error)
            self.assertEqual(
                run.result["comparisons"][0]["decision"]["verdict"], "equivalent"
            )
            self.assertIn("retrieval", [event["stage"] for event in run.events])
            self.assertIn("comparison", [event["stage"] for event in run.events])
            self.assertEqual(len(list((root / "verified").glob("*.json"))), 1)

    async def test_parent_survives_history_eviction(self):
        async def execute(request):
            return {}

        manager = RunManager(execute)
        parent = manager.start(RunRequest(task="claimify", source=SOURCE))
        await parent.worker
        for _ in range(11):
            run = manager.start(RunRequest(task="segmentation", source=SOURCE))
            await run.worker
        child = manager.start(
            RunRequest(task="claim_store", source=SOURCE, parent_run_id=parent.id)
        )
        await child.worker
        self.assertIn(parent.id, manager.runs)
        self.assertEqual(len(manager.runs), 12)


class DemoFileTests(unittest.TestCase):
    def test_documents_and_explicit_save(self):
        from fastapi.testclient import TestClient

        from data_platform.demo.app import create_app
        from data_platform.demo.runs import Run

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            root.mkdir()
            article = root / "article.md"
            article.write_text(SOURCE, encoding="utf-8")
            outside = Path(directory) / "outside.md"
            outside.write_text("private", encoding="utf-8")
            (root / "escape.md").symlink_to(outside)
            app = create_app(data_root=root)
            run = Run(
                RunRequest(task="segmentation", source=SOURCE),
                status="completed",
                result={"markdown": SOURCE, "sections": []},
            )
            app.state.manager.runs[run.id] = run
            with TestClient(app) as client:
                self.assertEqual(
                    client.get("/api/documents").json(), [{"name": "article.md"}]
                )
                self.assertEqual(
                    client.get(
                        "/api/documents/content", params={"path": "article.md"}
                    ).json()["source"],
                    SOURCE,
                )
                for name in ("../outside.md", "escape.md"):
                    self.assertEqual(
                        client.get(
                            "/api/documents/content", params={"path": name}
                        ).status_code,
                        422,
                    )
                self.assertFalse((root / "demo-results").exists())
                response = client.post(f"/api/runs/{run.id}/save")
                self.assertEqual(response.status_code, 200)
                self.assertEqual(
                    client.post(f"/api/runs/{run.id}/save").json(), response.json()
                )
                custom = Path(directory) / "custom-results"
                alternate = client.post(
                    f"/api/runs/{run.id}/save", json={"directory": str(custom)}
                )
                self.assertEqual(alternate.status_code, 200)
                self.assertEqual((custom / run.id / "result.md").read_text(), SOURCE)
                self.assertTrue(
                    all(
                        Path(name).is_relative_to(custom.resolve())
                        for name in alternate.json()["files"]
                    )
                )
                self.assertEqual(
                    client.post(
                        f"/api/runs/{run.id}/save", json={"directory": "  "}
                    ).json(),
                    response.json(),
                )
                saved = root / "demo-results" / run.id
                self.assertEqual((saved / "result.md").read_text(), SOURCE)
                payload = json.loads((saved / "run.json").read_text())
                self.assertEqual(payload["settings"]["max_concurrency"], 20)
                self.assertEqual(payload["source"], SOURCE)
                self.assertEqual(article.read_text(), SOURCE)
                self.assertEqual(
                    client.get("/api/documents").json(), [{"name": "article.md"}]
                )
                run.status = "running"
                self.assertEqual(
                    client.post(f"/api/runs/{run.id}/save").status_code, 409
                )


class MarkdownPreviewTests(unittest.TestCase):
    def test_rendered_structure_and_inert_untrusted_content(self):
        from data_platform.demo.markdown import render_preview

        html = render_preview(
            "# Report\n\n- **First**\n- Second\n\n<script>alert(1)</script>\n\n[bad](javascript:alert(1))\n\n![alt](https://example.com/tracker.png)"
        )
        self.assertIn("<h1>Report</h1>", html)
        self.assertIn("<ul>", html)
        self.assertIn("<strong>First</strong>", html)
        self.assertNotIn("<script>", html)
        self.assertNotIn("<img", html)
        self.assertNotIn('href="javascript:', html)
        self.assertIn("default-src 'none'", html)

    def test_preview_endpoint(self):
        from fastapi.testclient import TestClient

        from data_platform.demo.app import create_app

        with TestClient(create_app()) as client:
            response = client.post(
                "/api/preview", json={"task": "segmentation", "source": "# Hello"}
            )
            self.assertEqual(response.status_code, 200)
            self.assertIn("<h1>Hello</h1>", response.json()["html"])


class SegmentationHandoffTests(unittest.TestCase):
    def test_publication_metadata_preserves_sentence_offsets(self):
        from data_platform.digest.source import parse_document

        source = "# Report\nPublished date 28-10-2026\n\n## Results\n\nAcme plans two factories.\n"
        document = parse_document(source)
        sentence = document.sections[0].sentences[0]
        self.assertEqual(source[sentence.start : sentence.end], sentence.text)
        self.assertEqual(sentence.text, "Acme plans two factories.")
        with self.assertRaisesRegex(ValueError, "Body content"):
            parse_document(
                source.replace("Published date 28-10-2026", "Acme hired 200 people.")
            )


class SavedRunRecoveryTests(unittest.TestCase):
    def test_saved_run_survives_server_restart(self):
        from fastapi.testclient import TestClient

        from data_platform.demo.app import create_app
        from data_platform.demo.files import DemoFiles
        from data_platform.demo.runs import Run

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = Run(
                RunRequest(task="segmentation", source=SOURCE),
                status="completed",
                result={"markdown": SOURCE},
            )
            run.append({"stage": "run", "status": "completed", "data": {}, "spans": []})
            DemoFiles(root).save(run)
            with TestClient(create_app(data_root=root)) as client:
                self.assertEqual(
                    client.get(f"/api/runs/{run.id}").json()["result"], run.result
                )
                self.assertIn(
                    '"status": "completed"',
                    client.get(f"/api/runs/{run.id}/events").text,
                )
                self.assertEqual(client.get("/").headers["cache-control"], "no-store")
            self.assertIsNone(DemoFiles(root).load_run("../run"))
