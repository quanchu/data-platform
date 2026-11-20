"""Local Markdown inputs and explicitly saved, self-contained run artifacts."""

import json
import re
import shutil
import tempfile
from pathlib import Path

from .models import RunRequest
from .runs import Run


class DemoFiles:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.results = self.root / "demo-results"

    def document(self, name: str) -> Path:
        path = (self.root / name).resolve()
        if (
            not path.is_relative_to(self.root)
            or path.is_relative_to(self.results)
            or path.suffix.lower() != ".md"
            or not path.is_file()
        ):
            raise ValueError("Choose a Markdown file inside the data directory.")
        return path

    def documents(self) -> list[dict]:
        documents = []
        for path in sorted(self.root.rglob("*.md")):
            name = path.relative_to(self.root).as_posix()
            try:
                self.document(name)
            except ValueError:
                continue
            documents.append({"name": name})
        return documents

    def read(self, name: str) -> dict:
        path = self.document(name)
        if path.stat().st_size > 800_000:
            raise ValueError("Document exceeds the 200,000-character input limit.")
        source = path.read_text(encoding="utf-8")
        if len(source) > 200_000:
            raise ValueError("Document exceeds the 200,000-character input limit.")
        return {"name": name, "source": source}

    def load_run(self, run_id: str) -> Run | None:
        """Recover only explicit saves from the default result directory."""
        if not re.fullmatch(r"[0-9a-f]{32}", run_id):
            return None
        path = self.results / run_id / "run.json"
        if (
            not path.resolve().is_relative_to(self.results.resolve())
            or not path.is_file()
        ):
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload["id"] != run_id or payload["status"] != "completed":
                return None
            run = Run(
                request=RunRequest.model_validate(payload["settings"]),
                id=run_id,
                status="completed",
                result=payload["result"],
                events=payload["events"],
            )
            if not isinstance(run.result, dict) or not isinstance(run.events, list):
                return None
            return run
        except OSError, ValueError, KeyError, TypeError:
            return None

    def save(self, run: Run, directory: str | None = None) -> list[str]:
        if run.status != "completed" or run.result is None:
            raise ValueError("Only a completed run can be saved.")
        result_root = (
            Path(directory.strip()).expanduser()
            if directory and directory.strip()
            else self.results
        )
        if result_root.is_symlink():
            raise ValueError("The result directory must not be a symbolic link.")
        result_root = result_root.resolve()
        result_root.mkdir(parents=True, exist_ok=True)
        destination = result_root / run.id
        names = ["run.json", "result.json"]
        if isinstance(run.result.get("markdown"), str):
            names.append("result.md")
        if destination.is_symlink():
            raise ValueError("The run directory must not be a symbolic link.")
        if not destination.exists():
            staging = Path(tempfile.mkdtemp(prefix=".saving-", dir=result_root))
            try:
                payload = {
                    **run.view(),
                    "settings": run.request.model_dump(),
                    "events": run.events,
                }
                for name, value in (("run.json", payload), ("result.json", run.result)):
                    (staging / name).write_text(
                        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                    )
                if "result.md" in names:
                    (staging / "result.md").write_text(
                        run.result["markdown"], encoding="utf-8"
                    )
                staging.rename(destination)
            finally:
                if staging.exists():
                    shutil.rmtree(staging)
        return [str(destination / name) for name in names]
