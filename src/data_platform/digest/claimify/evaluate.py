"""Opt-in live evaluation; emits results for human semantic review."""
import argparse
import asyncio
import json
import os
import time
from collections import Counter
from pathlib import Path

from .extractor import extract_claims


async def evaluate(cases: list[dict]) -> dict:
    records = []
    statuses = Counter()
    started = time.monotonic()
    for case in cases:
        begin = time.monotonic()
        result = await extract_claims(case['markdown'])
        actual = [sentence.status for sentence in result.sentences]
        statuses.update(actual)
        records.append({
            **case,
            'actual_statuses': actual,
            'status_match': actual == case['expected_statuses'],
            'seconds': time.monotonic() - begin,
            'result': result.model_dump(mode='json'),
            'human_review': {'entailment': None, 'coverage': None, 'contextual_completeness': None},
        })
    count = sum(statuses.values())
    return {
        'review_status': 'pending_human_review',
        'seconds': time.monotonic() - started,
        'sentence_status_counts': dict(statuses),
        'sentence_status_rates': {k: v / count for k, v in statuses.items()} if count else {},
        'cases': records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cases', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--live', action='store_true', help='Explicitly enable paid provider requests')
    args = parser.parse_args()
    if not args.live:
        parser.error('--live is required; this evaluation calls both providers')
    if any(not os.getenv(key, '').strip() for key in ('TYPESAFE_API_KEY', 'OPENAI_API_KEY')):
        parser.error('Set TYPESAFE_API_KEY and OPENAI_API_KEY')
    if args.output.exists():
        parser.error('Output already exists; choose a new report path')
    cases = json.loads(args.cases.read_text())
    report = asyncio.run(evaluate(cases))
    with args.output.open('x') as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
