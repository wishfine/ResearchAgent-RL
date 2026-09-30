from __future__ import annotations
import asyncio
from argparse import Namespace
from contextlib import contextmanager
import unittest

try:
    import slime
    import slime.rollout.sglang_rollout as sr
    from slime.utils.types import Sample
    SLIME_AVAILABLE = True
except ImportError:
    SLIME_AVAILABLE = False


@unittest.skipUnless(SLIME_AVAILABLE, "slime library is not available in current environment")
class TestSlimeGenerateAndRmContract(unittest.TestCase):
    def test_generate_and_rm_plugin_chain(self):
        original_generate_state = sr.GenerateState
        sr.GenerateState = FakeGenerateState
        try:
            asyncio.run(self._run_case())
        finally:
            sr.GenerateState = original_generate_state

    async def _run_case(self):
        args = Namespace(
            custom_generate_function_path="research_agent.adapters.slime.custom_generator.custom_generate",
            custom_rm_path="research_agent.adapters.slime.custom_reward.custom_rm",

            group_rm=False,
            partial_rollout=False,
            mask_offpolicy_in_partial_rollout=False,

            corpus_dir="data/searchqa_debug/corpus",
            max_steps=4,
            reward_type="contains",
            step_penalty=0.01,
            invalid_penalty=0.05,
            actor_model_url="mock",
        )

        sample = Sample(
            index=0,
            group_index=0,
            group_id=0,
            prompt="Which country's flag features a red maple leaf?",
            metadata={
                "task_id": "toy_task_001",
                "user_query": "Which country's flag features a red maple leaf?",
                "ground_truth_answer": "Canada",
                "ground_truth_citations": ["maple_leaf_flag"],
                "reference_docs": [],
            },
        )

        out = await sr.generate_and_rm(
            args,
            sample,
            sampling_params={
                "temperature": 0.0,
                "max_tokens": 256,
            },
            evaluation=False,
        )

        self.assertIsInstance(out, Sample)
        self.assertIsInstance(out.reward, float)
        self.assertEqual(out.metadata.get("done_reason"), "answer_submitted")
        self.assertTrue(out.metadata.get("final_answer"))
        self.assertEqual(len(out.tokens), len(out.loss_mask or []))
        self.assertEqual(out.response_length, sum(out.loss_mask or []))


class FakeGenerateState:
    def __init__(self, args):
        self.args = args
        self.semaphore = asyncio.Semaphore(1)
        self.aborted = False

    @contextmanager
    def dp_rank_context(self):
        yield
