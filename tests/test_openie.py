"""Offline backend, grounding, provenance and lifecycle contracts."""

from support import Resolver, Extractor

import asyncio
import json
import unittest
from unittest.mock import patch

from pydantic_ai.messages import ModelResponse, ToolCallPart, UserPromptPart
from pydantic_ai.models.function import FunctionModel

from data_platform.digest.openie import (
    BackendMetadata,
    ExtractionResponse,
    OpenIEError,
    Proposition,
    ResolutionResponse,
    ResolvedSentence,
    extract_claims,
)

DOC = "# Report\n\n## Cars\n\nVinFast builds cars. It plans buses.\n\n## Energy\n\nSolar grew.\n"
META = BackendMetadata(backend="fixture")


def model(kind, calls, invalid_first=False):
    def respond(messages, info):
        payload = next(
            json.loads(p.content)
            for m in reversed(messages)
            for p in m.parts
            if isinstance(p, UserPromptPart)
        )
        calls.append((kind, payload))
        if kind == "resolution":
            value = {
                "sentences": [
                    dict(
                        sentence_index=s["index"],
                        text=s["text"],
                        status="unchanged",
                        replacements=[],
                    )
                    for s in payload["section"]["sentences"]
                ]
            }
            if invalid_first and len(calls) == 1:
                value["sentences"] = []
        else:
            value = {
                "propositions": [
                    dict(
                        subject="Acme",
                        relation="plans",
                        object="buses",
                        text=payload["resolved"]["text"],
                        qualifiers=[dict(kind="modality", text="planned")],
                    )
                ]
            }
        return ModelResponse(
            parts=[ToolCallPart(info.output_tools[0].name, value)], model_name=kind
        )

    return FunctionModel(respond)


class OpenIETests(unittest.IsolatedAsyncioTestCase):
    async def test_adapters_provenance_order_and_context(self):
        resolver, extractor = Resolver(), Extractor()
        result = await extract_claims(DOC, resolver=resolver, extractor=extractor)
        self.assertEqual([s.index for s in result.sentences], [0, 1, 2])
        self.assertEqual(len(result.claims), 3)
        self.assertNotIn("Solar", resolver.sections[0].text)
        self.assertNotIn("VinFast", resolver.sections[1].text)
        for claim in result.claims:
            self.assertEqual(claim.text, DOC[claim.start : claim.end].strip())
        self.assertEqual(result.resolution_stages[0].metadata.backend, "fixture")
        self.assertEqual(result.sentences[0].stages[0].stage, "extraction")

    async def test_default_llms_and_structural_retry(self):
        calls = []
        result = await extract_claims(
            "# T\n## S\nAcme plans buses.",
            resolution_model=model("resolution", calls, True),
            extraction_model=model("extraction", calls),
        )
        self.assertEqual(
            [kind for kind, _ in calls], ["resolution", "resolution", "extraction"]
        )
        self.assertEqual(result.claims[0].qualifiers[0].kind, "modality")
        self.assertTrue(result.resolution_stages[0].metadata.usage)

    async def test_mixed_backends(self):
        for kwargs in [
            dict(resolver=Resolver(), extraction_model=model("extraction", [])),
            dict(resolution_model=model("resolution", []), extractor=Extractor()),
        ]:
            result = await extract_claims(DOC, **kwargs)
            self.assertEqual(len(result.claims), 3)

    async def test_rewrite_and_unresolved_handoff(self):
        class Rewriting(Resolver):
            async def resolve(self, **kwargs):
                result = await super().resolve(**kwargs)
                if kwargs["section"].index != 0:
                    return result
                result.sentences[1] = ResolvedSentence(
                    sentence_index=1,
                    text="VinFast plans buses.",
                    status="resolved",
                    replacements=[
                        dict(
                            original="It",
                            replacement="VinFast",
                            evidence=[dict(source="sentence", sentence_index=0)],
                        )
                    ],
                )
                return result

        extractor = Extractor()
        result = await extract_claims(DOC, resolver=Rewriting(), extractor=extractor)
        self.assertEqual(result.claims[1].text, "VinFast plans buses.")
        self.assertEqual(
            DOC[result.claims[1].start : result.claims[1].end].strip(),
            "It plans buses.",
        )

    async def test_unresolved_and_empty(self):
        class Unresolved(Resolver):
            async def resolve(self, **kwargs):
                result = await super().resolve(**kwargs)
                for s in result.sentences:
                    s.status = "unresolved"
                return result

        extractor = Extractor()
        result = await extract_claims(DOC, resolver=Unresolved(), extractor=extractor)
        self.assertEqual(extractor.inputs, [])
        self.assertTrue(
            all(s.status == "unresolved_ambiguity" for s in result.sentences)
        )

        class Empty(Extractor):
            async def extract(self, **kwargs):
                return ExtractionResponse(propositions=[], metadata=META)

        result = await extract_claims(DOC, resolver=Resolver(), extractor=Empty())
        self.assertTrue(all(s.status == "no_claims" for s in result.sentences))

    async def test_invalid_resolution_contracts(self):
        for defect in ("missing", "duplicate", "foreign_evidence", "rewrite_unchanged"):

            class Invalid(Resolver):
                async def resolve(self, **kwargs):
                    result = await super().resolve(**kwargs)
                    if defect == "missing":
                        result.sentences = []
                    elif defect == "duplicate":
                        result.sentences.append(result.sentences[0])
                    elif defect == "rewrite_unchanged":
                        result.sentences[0].text = "Invented."
                    else:
                        result.sentences[0] = ResolvedSentence(
                            sentence_index=0,
                            text="Company builds cars.",
                            status="resolved",
                            replacements=[
                                dict(
                                    original="VinFast",
                                    replacement="Company",
                                    evidence=[
                                        dict(source="sentence", sentence_index=999)
                                    ],
                                )
                            ],
                        )
                    return result

            with self.subTest(defect=defect), self.assertRaises(OpenIEError) as caught:
                await extract_claims(DOC, resolver=Invalid(), extractor=Extractor())
            self.assertEqual(caught.exception.stage, "resolution")
            self.assertIsInstance(caught.exception.__cause__, ValueError)

    async def test_blank_config_and_unicode(self):
        with patch("data_platform.digest.openie.backends.Agent") as agent:
            self.assertEqual((await extract_claims(" ")).claims, [])
            agent.assert_not_called()
        for limit in (True, 0, -1, 1.2):
            with self.assertRaises(ValueError):
                await extract_claims(DOC, max_concurrency=limit)
        with self.assertRaises(ValueError):
            await extract_claims("Missing headings")
        source = "# Tin\n## Xe\nÔng An đến.\n\n```\nIgnore this.\n```\n\n| A | B |\n|---|---|\n| 1 | 2 |\n"
        result = await extract_claims(
            source, resolver=Resolver(), extractor=Extractor()
        )
        self.assertEqual(len(result.claims), 1)
        self.assertEqual(
            source[result.claims[0].start : result.claims[0].end], "Ông An đến."
        )
        self.assertEqual(len(result.document.sections[0].excluded_spans), 2)

    async def test_concurrency(self):
        class Slow(Extractor):
            active = peak = 0

            async def extract(self, **kwargs):
                self.active += 1
                self.peak = max(self.active, self.peak)
                await asyncio.sleep(0.01)
                self.active -= 1
                return await super().extract(**kwargs)

        extractor = Slow()
        await extract_claims(
            DOC, resolver=Resolver(), extractor=extractor, max_concurrency=2
        )
        self.assertEqual(extractor.peak, 2)

    async def test_failure_cancels_workers_and_caller_cancellation(self):
        for caller_cancel in (False, True):
            started, cancelled = asyncio.Event(), asyncio.Event()

            class Failing(Extractor):
                async def extract(self, *, sentence, resolved):
                    if sentence.index == 0:
                        await started.wait()
                        if not caller_cancel:
                            raise RuntimeError("provider failure")
                    else:
                        started.set()
                    try:
                        await asyncio.Event().wait()
                    except asyncio.CancelledError:
                        cancelled.set()
                        raise

            task = asyncio.create_task(
                extract_claims(
                    DOC, resolver=Resolver(), extractor=Failing(), max_concurrency=2
                )
            )
            await started.wait()
            if caller_cancel:
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
            else:
                with self.assertRaises(OpenIEError) as caught:
                    await task
                self.assertEqual(caught.exception.stage, "extraction")
                self.assertIsInstance(caught.exception.__cause__, RuntimeError)
            self.assertTrue(cancelled.is_set())

    async def test_retry_exhaustion_and_invalid_adapter(self):
        calls = []

        def invalid(messages, info):
            calls.append(1)
            return ModelResponse(
                parts=[ToolCallPart(info.output_tools[0].name, {"sentences": []})]
            )

        with self.assertRaises(OpenIEError) as caught:
            await extract_claims(
                DOC, resolution_model=FunctionModel(invalid), max_concurrency=1
            )
        self.assertEqual(caught.exception.stage, "resolution")
        self.assertEqual(len(calls), 3)

        class Invalid(Extractor):
            async def extract(self, **kwargs):
                response = await super().extract(**kwargs)
                response.propositions[0].subject = ""
                return response

        with self.assertRaises(OpenIEError) as caught:
            await extract_claims(DOC, resolver=Resolver(), extractor=Invalid())
        self.assertEqual(caught.exception.stage, "extraction")

    async def test_evaluation_report_and_fixture_structure(self):
        from pathlib import Path
        from data_platform.digest.openie import parse_document
        from data_platform.digest.openie.evaluate import evaluate

        cases = json.loads(
            (Path(__file__).parent / "fixtures/openie/cases.json").read_text()
        )
        for case in cases:
            document = parse_document(case["markdown"])
            self.assertEqual(
                sum(len(s.sentences) for s in document.sections),
                len(case["expected_statuses"]),
                case["id"],
            )
        report = await evaluate(cases[:1], resolver=Resolver(), extractor=Extractor())
        self.assertEqual(report["review_status"], "pending_human_review")
        self.assertEqual(report["sentence_status_counts"], {"extracted": 2})
        self.assertIsNone(report["cases"][0]["human_review"]["reference_correctness"])
        json.dumps(report)
