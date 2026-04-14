"""Train module: SFT and RL training for ResearchAgent-RL."""
from .sft.dataset import TrajectoryToSFTConverter, SFTRole
from .sft.trainer import SFTTrainer
from .rl.buffer import TrajectoryBuffer
from .rl.reward import compute_reward_signals, RewardFunction
from .rl.grpo import GRPOTrainer

__all__ = [
    "TrajectoryToSFTConverter",
    "SFTRole",
    "SFTTrainer",
    "TrajectoryBuffer",
    "compute_reward_signals",
    "RewardFunction",
    "GRPOTrainer",
]
