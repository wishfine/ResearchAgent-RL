from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

from scripts.run_llm_baseline import select_task_paths


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

    def test_seeded_task_selection_is_reproducible(self):
        paths = [f"/tasks/{name}.json" for name in ("c", "a", "d", "b")]

        first = select_task_paths(paths, max_episodes=3, selection_seed=2026)
        second = select_task_paths(paths, max_episodes=3, selection_seed=2026)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)
        self.assertNotEqual(first, select_task_paths(paths, max_episodes=3, selection_seed=None))


if __name__ == "__main__":
    unittest.main()
