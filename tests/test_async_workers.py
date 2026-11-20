"""Worker ordering, bounds, and cleanup without provider calls."""

import asyncio
import unittest

from data_platform.digest._async import ordered_map


class WorkerTests(unittest.IsolatedAsyncioTestCase):
    async def test_bounded_ordered_and_empty(self):
        active = peak = 0

        async def work(value):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep((5 - value) * 0.001)
            active -= 1
            return value * 2

        self.assertEqual(
            await ordered_map(work, range(5), max_concurrency=2), [0, 2, 4, 6, 8]
        )
        self.assertEqual(peak, 2)
        self.assertEqual(await ordered_map(work, [], max_concurrency=2), [])
        for limit in (True, 0, -1, 1.5):
            with self.assertRaises(ValueError):
                await ordered_map(work, [], max_concurrency=limit)

    async def test_failure_and_caller_cancellation_cleanup(self):
        for caller_cancel in (False, True):
            started, finished = asyncio.Event(), asyncio.Event()
            failure = RuntimeError("failed")

            async def work(value):
                if value == 0:
                    await started.wait()
                    if not caller_cancel:
                        raise failure
                else:
                    started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    finished.set()

            task = asyncio.create_task(ordered_map(work, [0, 1], max_concurrency=2))
            await started.wait()
            if caller_cancel:
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
            else:
                with self.assertRaises(RuntimeError) as caught:
                    await task
                self.assertIs(caught.exception, failure)
            self.assertTrue(finished.is_set())


class ConcurrencyDefaultsTests(unittest.TestCase):
    def test_public_workflows_default_to_twenty(self):
        import inspect
        from data_platform.digest.claimify import extract_claims as claimify
        from data_platform.digest.openie import extract_claims as openie
        from data_platform.digest.claim_store import ClaimStore
        from data_platform.digest.segmenter.texttiling import (
            detect_document_structure,
            generate_headings,
            segment_with_headings,
            texttiling_segement,
        )

        for workflow in (
            claimify,
            openie,
            ClaimStore,
            detect_document_structure,
            generate_headings,
            segment_with_headings,
            texttiling_segement,
        ):
            with self.subTest(workflow=workflow.__name__):
                self.assertEqual(
                    inspect.signature(workflow).parameters["max_concurrency"].default,
                    20,
                )

    def test_demo_accepts_twenty(self):
        from pydantic import ValidationError
        from data_platform.demo.models import RunRequest

        self.assertEqual(RunRequest(task="segmentation").max_concurrency, 20)
        self.assertEqual(
            RunRequest(task="segmentation", max_concurrency=20).max_concurrency, 20
        )
        with self.assertRaises(ValidationError):
            RunRequest(task="segmentation", max_concurrency=21)
