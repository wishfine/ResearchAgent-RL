from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


class TestQwenZeroCLI(unittest.TestCase):
    def test_help_exposes_vllm_served_model_name(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, "scripts/qwen_zero_shot_rollout.py", "--help"],
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--model_name", result.stdout)


if __name__ == "__main__":
    unittest.main()
