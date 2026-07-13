from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


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


if __name__ == "__main__":
    unittest.main()
