"""Opt-in live OpenIE evaluation; semantic scores require human review."""
import argparse
import asyncio
import json
import time
from collections import Counter
from pathlib import Path

from .backends import DEFAULT_MODEL
from .extractor import extract_claims


async def evaluate(cases, **kwargs):
    records = []
    statuses = Counter()
    started = time.monotonic()
    for case in cases:
        begin = time.monotonic()
        result = await extract_claims(case['markdown'], **kwargs)
        actual = [s.status for s in result.sentences]
        statuses.update(actual)
        records.append({**case, 'actual_statuses': actual,
                        'status_match': actual == case.get('expected_statuses'),
                        'seconds': time.monotonic() - begin,
                        'result': result.model_dump(mode='json'),
                        'human_review': {key: None for key in (
                            'reference_correctness', 'entailment', 'qualifier_preservation', 'coverage')}})
    return {'review_status': 'pending_human_review', 'seconds': time.monotonic() - started,
            'sentence_status_counts': dict(statuses), 'cases': records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cases', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--live', action='store_true', help='Enable paid model requests')
    parser.add_argument('--resolution-model', default=DEFAULT_MODEL)
    parser.add_argument('--extraction-model', default=DEFAULT_MODEL)
    args = parser.parse_args()
    if not args.live:
        parser.error('--live is required; this evaluation calls model providers')
    if args.output.exists():
        parser.error('Output already exists; choose a new report path')
    cases = json.loads(args.cases.read_text(encoding='utf-8'))
    report = asyncio.run(evaluate(cases, resolution_model=args.resolution_model, extraction_model=args.extraction_model))
    with args.output.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
