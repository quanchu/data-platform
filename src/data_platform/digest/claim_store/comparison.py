"""Typed model comparisons, kept separate from approval and retrieval."""

import json

import numpy as np
from pydantic_ai import Agent
from pydantic_ai.models import Model

from .prompts import COMPARISON_PROMPT as COMPARISON_PROMPT

from .._events import emit
from .._async import ordered_map
from .index import rank
from .models import ClaimRecord, ComparisonDecision, PairComparison


async def compare_claims(
    records: list[ClaimRecord],
    vectors: list[np.ndarray],
    verified: list[ClaimRecord],
    matrix: np.ndarray,
    *,
    top_k: int,
    model: str | Model,
    max_concurrency: int,
) -> list[PairComparison]:
    agent = Agent(model, output_type=ComparisonDecision, instructions=COMPARISON_PROMPT)
    comparisons = []
    for record, vector in zip(records, vectors, strict=True):

        async def compare(match, record=record):
            spans = [(record.start, record.end)]
            emit(
                "comparison",
                status="running",
                spans=spans,
                prompt=COMPARISON_PROMPT,
                claim=record.model_dump(mode="json"),
                verified_claim=match.claim.model_dump(mode="json"),
            )
            response = await agent.run(
                json.dumps(
                    {
                        "claim": record.model_dump(mode="json"),
                        "verified_claim": match.claim.model_dump(mode="json"),
                    },
                    ensure_ascii=False,
                )
            )
            emit(
                "comparison",
                spans=spans,
                decision=response.output.model_dump(),
                verified_claim=match.claim.model_dump(mode="json"),
                similarity=match.similarity,
            )
            return PairComparison(
                claim=record,
                verified_claim=match.claim,
                similarity=match.similarity,
                decision=response.output,
            )

        matches = rank(vector, verified, matrix, top_k)
        emit(
            "retrieval",
            spans=[(record.start, record.end)],
            matches=[match.model_dump(mode="json") for match in matches],
        )
        comparisons.extend(
            await ordered_map(compare, matches, max_concurrency=max_concurrency)
        )
    return comparisons
