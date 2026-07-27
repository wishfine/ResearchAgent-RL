from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from scripts.run_llm_baseline import models_endpoint, select_task_paths, wait_for_model_server


class TestRunLLMBaselineCLI(unittest.TestCase):
    def test_help_exposes_three_episode_smoke_controls(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, "scripts/run_llm_baseline.py", "--help"],
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--max_episodes", result.stdout)
        self.assertIn("--model_name", result.stdout)
        self.assertIn("--output_dir", result.stdout)
        self.assertIn("--selection_seed", result.stdout)
        self.assertIn("--api_ready_timeout_sec", result.stdout)

    def test_seeded_task_selection_is_reproducible(self):
        paths = [f"/tasks/{name}.json" for name in ("c", "a", "d", "b")]

        first = select_task_paths(paths, max_episodes=3, selection_seed=2026)
        second = select_task_paths(paths, max_episodes=3, selection_seed=2026)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)
        self.assertNotEqual(first, select_task_paths(paths, max_episodes=3, selection_seed=None))

    def test_models_endpoint_and_readiness_check(self):
        self.assertEqual(
            models_endpoint("http://127.0.0.1:8000/v1/chat/completions"),
            "http://127.0.0.1:8000/v1/models",
        )
        response = MagicMock()
        response.read.return_value = b'{"data": []}'
        mocked_open = MagicMock()
        mocked_open.return_value.__enter__.return_value = response
        with patch("scripts.run_llm_baseline.urllib.request.urlopen", mocked_open):
            endpoint = wait_for_model_server("http://127.0.0.1:8000/v1", timeout_sec=0)
        self.assertEqual(endpoint, "http://127.0.0.1:8000/v1/models")


if __name__ == "__main__":
    unittest.main()
