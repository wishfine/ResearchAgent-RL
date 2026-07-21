from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from research_agent.core.schema.parser import ActionParser
from scripts.prepare_hotpotqa_sft import build_examples, write_examples


class TestPrepareHotpotQASFT(unittest.TestCase):
    def test_builds_strict_grounded_four_turn_trajectory(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            tasks = root / "tasks"
            corpus = root / "corpus"
            tasks.mkdir()
            corpus.mkdir()
            task_id = "hotpot_test"
            citation_id = "evidence_hotpot_test"
            (tasks / f"{task_id}.json").write_text(json.dumps({
                "task_id": task_id,
                "task_type": "survey_synthesis",
                "user_query": "Which country has a maple leaf flag?",
                "ground_truth_answer": "Canada",
                "ground_truth_citations": [citation_id],
                "reference_docs": [task_id],
            }), encoding="utf-8")
            (corpus / f"{task_id}.json").write_text(json.dumps({
                "doc_id": task_id,
                "chunks": [{"chunk_id": citation_id, "content": "Canada has a red maple leaf flag."}],
            }), encoding="utf-8")

            examples = build_examples(tasks, corpus)
            self.assertEqual(len(examples), 1)
            assistant_messages = [message["content"] for message in examples[0]["messages"] if message["role"] == "assistant"]
            self.assertEqual(len(assistant_messages), 4)
            self.assertEqual([ActionParser.parse(message).tool for message in assistant_messages], ["SEARCH", "READ", "CITE", "ANSWER"])
            self.assertTrue(all(message.startswith("<action>{") and message.endswith("}</action>") for message in assistant_messages))
            self.assertTrue(all("<reasoning>" not in message and "<think>" not in message for message in assistant_messages))

            output = root / "sft.jsonl"
            manifest = write_examples(examples, output)
            self.assertEqual(manifest["records"], 1)
            self.assertTrue(output.with_suffix(".jsonl.manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
