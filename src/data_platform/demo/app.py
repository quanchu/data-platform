"""Loopback-only live digest demo; run with python -m data_platform.demo."""

import asyncio
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .files import DemoFiles
from .markdown import render_preview
from .models import RunRequest, SaveRequest
from .runs import RunManager
from .tasks import Tasks

HERE = Path(__file__).parent


def create_app(
    *, store_root: Path | None = None, data_root: Path | None = None, executor=None
) -> FastAPI:
    files = DemoFiles(data_root or Path(".data"))
    tasks = Tasks(store_root or files.root / "claims-demo")
    manager = RunManager(executor or tasks)
    tasks.manager = manager

    @asynccontextmanager
    async def lifespan(app):
        yield
        await manager.close()

    app = FastAPI(title="Data platform live demo", lifespan=lifespan)
    app.state.manager = manager
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["127.0.0.1", "localhost", "[::1]", "testserver"],
    )

    @app.middleware("http")
    async def same_origin(request: Request, call_next):
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            origin = request.headers.get("origin")
            if origin and origin != str(request.base_url).rstrip("/"):
                from fastapi.responses import JSONResponse

                return JSONResponse(
                    {"detail": "Use the demo's own browser tab."}, status_code=403
                )
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        return response

    def get_run(run_id):
        if run_id not in manager.runs:
            restored = files.load_run(run_id)
            if restored is not None:
                # Reading saved history must not evict an active run or its parent.
                return restored
        if run_id not in manager.runs:
            raise HTTPException(
                404, "Run not found; the server retains the latest 12 runs."
            )
        return manager.runs[run_id]

    @app.get("/")
    async def index():
        return FileResponse(HERE / "static/index.html")

    @app.get("/api/capabilities")
    async def capabilities():
        return {
            "providers": {
                name: bool(os.getenv(name, "").strip())
                for name in ("OPENAI_API_KEY", "TYPESAFE_API_KEY")
            },
            "store_root": str(tasks.store_root),
            "max_runs": 12,
            "data_root": str(files.root),
            "result_root": str(files.results),
        }

    @app.post("/api/preview")
    async def preview(request: RunRequest):
        return {"html": render_preview(request.source)}

    @app.get("/api/documents")
    async def documents():
        return files.documents()

    @app.get("/api/documents/content")
    async def document(path: str):
        try:
            return files.read(path)
        except (ValueError, OSError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/api/runs/{run_id}/save")
    async def save(run_id: str, request: SaveRequest | None = None):
        try:
            return {
                "files": files.save(
                    get_run(run_id), request.directory if request else None
                )
            }
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        except OSError as exc:
            raise HTTPException(500, "Could not save the result files.") from exc

    @app.get("/api/examples")
    async def examples():
        return [
            {"name": path.name, "source": path.read_text(encoding="utf-8")}
            for path in sorted((HERE / "examples").glob("*.md"))
        ]

    @app.get("/api/claims")
    async def claims():
        from ..digest.claim_store.storage import ClaimFiles

        files = ClaimFiles(tasks.store_root)
        return {
            status: [
                record.model_dump(mode="json") for record in files.list_claims(status)
            ]
            for status in ("unverified", "verified")
        }

    @app.post("/api/runs", status_code=201)
    async def start_run(request: RunRequest):
        try:
            if request.parent_run_id and request.parent_run_id not in manager.runs:
                restored = files.load_run(request.parent_run_id)
                if restored is not None:
                    manager.runs[restored.id] = restored
            request = tasks.validate(request)
            run = manager.start(request)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        return run.view()

    @app.get("/api/runs/{run_id}")
    async def result(run_id: str):
        return get_run(run_id).view()

    @app.post("/api/runs/{run_id}/cancel", status_code=202)
    async def cancel(run_id: str):
        run = get_run(run_id)
        # Return promptly while cleanup keeps the active slot occupied.
        asyncio.create_task(manager.cancel(run))
        return {"id": run.id, "status": "cancellation_requested"}

    @app.get("/api/runs/{run_id}/events")
    async def events(run_id: str, request: Request):
        run = get_run(run_id)
        try:
            cursor = int(request.headers.get("last-event-id", "-1")) + 1
        except ValueError:
            raise HTTPException(400, "Invalid event sequence")
        cursor = max(0, min(cursor, len(run.events)))

        async def stream():
            nonlocal cursor
            while True:
                run.changed.clear()
                while cursor < len(run.events):
                    event = run.events[cursor]
                    yield f"id: {cursor}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
                    cursor += 1
                if run.status in {"completed", "failed", "cancelled"}:
                    return
                try:
                    await asyncio.wait_for(run.changed.wait(), timeout=15)
                except TimeoutError:
                    yield ": heartbeat\n\n"

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
    return app
