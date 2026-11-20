"""One active run, real event history, and cancellation with complete cleanup."""

import asyncio
import copy
import hashlib
import threading
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from ..digest._async import run_blocking
from ..digest._events import observe
from .models import RunRequest

blocking_call = run_blocking


@dataclass
class Run:
    request: RunRequest
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    status: str = "running"
    events: list[dict] = field(default_factory=list)
    result: dict | None = None
    error: str | None = None
    changed: asyncio.Event = field(default_factory=asyncio.Event)
    worker: asyncio.Task | None = None

    @property
    def document_id(self) -> str:
        return hashlib.sha256(self.request.source.encode()).hexdigest()

    def append(self, event: dict) -> None:
        self.events.append(
            {
                **copy.deepcopy(event),
                "run_id": self.id,
                "sequence": len(self.events),
                "task": self.request.task,
                "document_id": self.document_id,
            }
        )
        self.changed.set()

    def view(self) -> dict:
        return {
            "id": self.id,
            "status": self.status,
            "task": self.request.task,
            "document_id": self.document_id,
            "source": self.request.source,
            "source_name": self.request.source_name,
            "result": self.result,
            "error": self.error,
        }


class RunManager:
    def __init__(self, execute: Callable[[RunRequest], Awaitable[dict]]):
        self.execute = execute
        self.runs: dict[str, Run] = {}

    def start(self, request: RunRequest) -> Run:
        if any(run.status in {"running", "cancelling"} for run in self.runs.values()):
            raise RuntimeError(
                "Another task is running. Cancel it or wait for completion."
            )
        while len(self.runs) >= 12:
            oldest = next(
                run_id for run_id in self.runs if run_id != request.parent_run_id
            )
            del self.runs[oldest]
        run = Run(request=request)
        self.runs[run.id] = run
        run.worker = asyncio.create_task(self._execute(run))
        return run

    async def _execute(self, run: Run) -> None:
        loop = asyncio.get_running_loop()
        owner = threading.get_ident()

        def receive(event):
            if threading.get_ident() == owner:
                run.append(event)
            else:
                loop.call_soon_threadsafe(run.append, event)

        run.append({"stage": "run", "status": "running", "spans": [], "data": {}})
        try:
            with observe(receive):
                run.result = await self.execute(run.request)
            run.status = "completed"
        except asyncio.CancelledError:
            run.status = "cancelled"
        except Exception as exc:  # noqa: BLE001 — capture failures at the run boundary.
            run.status = "failed"
            messages = [str(exc)]
            cause = exc.__cause__
            while cause is not None and len(messages) < 4:
                messages.append(f"{type(cause).__name__}: {cause}")
                cause = cause.__cause__
            run.error = " → ".join(messages)
        finally:
            run.append(
                {
                    "stage": "run",
                    "status": run.status,
                    "spans": [],
                    "data": {"error": run.error} if run.error else {},
                }
            )

    async def cancel(self, run: Run) -> None:
        if run.worker and not run.worker.done() and run.status != "cancelling":
            run.status = "cancelling"
            run.append(
                {
                    "stage": "run",
                    "status": "cancelling",
                    "spans": [],
                    "data": {
                        "message": "Waiting for active work to stop; local inference finishes its current call."
                    },
                }
            )
            run.worker.cancel()
        if run.worker:
            await asyncio.gather(run.worker, return_exceptions=True)
            if run.status == "cancelling":
                # A task cancelled before its first instruction cannot run its finally block.
                run.status = "cancelled"
                run.append(
                    {"stage": "run", "status": "cancelled", "spans": [], "data": {}}
                )

    async def close(self) -> None:
        await asyncio.gather(*(self.cancel(run) for run in list(self.runs.values())))
