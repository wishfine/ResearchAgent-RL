"""
src/utils/logging.py
职责: Trajectory JSON 持久化
设计: 序列化 EpisodeResult 和 EvalResult；按 task_id 组织输出目录
"""
from __future__ import annotations
import os
import json
from typing import List
from ..schema.result import EpisodeResult, EvalResult


class TrajectoryLogger:
    """
    轨迹和评测结果的持久化。

    输出结构:
        outputs/
            trajectories/
                {task_id}_step_{step_idx}.json   # 单步记录
                {task_id}_episode.json           # 完整 episode
            eval_results/
                {task_id}_eval.json              # 评测结果
    """

    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = output_dir
        self.trajectory_dir = os.path.join(output_dir, "trajectories")
        self.eval_dir = os.path.join(output_dir, "eval_results")
        os.makedirs(self.trajectory_dir, exist_ok=True)
        os.makedirs(self.eval_dir, exist_ok=True)

    def log_episode(self, result: EpisodeResult) -> None:
        filepath = os.path.join(self.trajectory_dir, f"{result.task_id}_episode.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

    def log_eval(self, result: EvalResult) -> None:
        filepath = os.path.join(self.eval_dir, f"{result.task_id}_eval.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

    def log_step(self, task_id: str, step_idx: int, step_data: dict) -> None:
        filepath = os.path.join(self.trajectory_dir, f"{task_id}_step_{step_idx}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(step_data, f, indent=2, ensure_ascii=False)
