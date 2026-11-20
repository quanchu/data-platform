"""Inputs for the local presentation API, independent of library contracts."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task: Literal["segmentation", "claimify", "openie", "entities", "claim_store"]
    source: str = Field(default="", max_length=200_000)
    source_name: str = Field(default="article.md", max_length=200)
    parent_run_id: str | None = None
    method: Literal["counts", "tfidf", "semantic", "llm"] = "counts"
    detect_headings: bool = False
    generate_headings: bool = False
    decision_model: str = "typesafe:jev-latest"
    generation_model: str = "openai:gpt-6-luna"
    embedding_model: str | None = None
    local_files_only: bool = True
    local_model_path: str | None = None
    max_concurrency: int = Field(default=20, ge=1, le=20)
    question: str | None = None
    entity_model: str = "urchade/gliner_multi-v2.1"
    labels: list[str] = Field(
        default_factory=lambda: ["company name", "date", "email", "person name"],
        min_length=1,
    )
    threshold: float = Field(default=0.5, gt=0, lt=1)
    store_action: Literal["save", "compare", "retrieve"] = "save"
    top_k: int = Field(default=5, ge=1, le=100)


class SaveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    directory: str | None = Field(default=None, max_length=4096)
