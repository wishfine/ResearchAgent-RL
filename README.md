# ResearchAgent-RL

Document-grounded multi-step tool-use research agent with RL-ready environment and SFT/RL training support.

## 项目目标

将科研任务（survey synthesis、method comparison、experiment design）建模为 document-grounded、finite-horizon、tool-use 的强化学习环境。Agent 在给定论文语料和步数预算内，通过 SEARCH → READ → RERANK → CITE → ANSWER 五个显式原子动作完成 reasoning-aware retrieval、evidence accumulation 和 citation-grounded answer synthesis。

**当前版本**: v0.2.0 (含 SFT + RL 训练支持)

## 核心功能

- **Environment**: `ResearchEnv` - reset/step/finalize_episode 三段式
- **5 个原子工具**: SEARCH, READ, RERANK, CITE, ANSWER
- **Rule-Based Baseline**: 可运行的启发式策略
- **Citation-Aware Eval**: citation precision/recall/F1 评测
- **SFT Training**: trajectory → SFT 格式转换 + HuggingFace Trainer
- **RL Training (GRPO)**: group-level advantage + PPO-style policy update

## 快速开始

```bash
# 安装依赖
pip install -e .

# 1. 收集 rule-based trajectories（准备 SFT 数据）
python scripts/run_training.py --mode collect --corpus_dir data/corpus --tasks_dir data/tasks --output_dir outputs/collected --n_episodes 100

# 2. SFT 训练
python scripts/run_training.py --mode sft --model_name_or_path gpt2 --train_file outputs/collected/collected_data.json --output_dir outputs/sft

# 3. RL (GRPO) 训练
python scripts/run_training.py --mode rl --model_name_or_path gpt2 --corpus_dir data/corpus --tasks_dir data/tasks --output_dir outputs/rl
```

## 完整目录结构

```
ResearchAgent-RL/
├── src/
│   ├── schema/              # 数据模型
│   │   ├── document.py      # Chunk, Document, CandidateChunk, ReadSummary
│   │   ├── task.py          # TaskSample, Rubric, TaskType
│   │   ├── action.py        # Action, TrajectoryStep
│   │   ├── observation.py    # Observation
│   │   └── result.py         # EpisodeResult, EvalResult, RewardSignals
│   ├── environment/         # 核心环境
│   │   ├── env.py           # ResearchEnv
│   │   ├── state.py         # EnvState
│   │   └── corpus.py        # CorpusStore
│   ├── tools/               # 5个原子动作
│   │   ├── base.py          # BaseTool, ToolResult
│   │   ├── search.py
│   │   ├── read.py
│   │   ├── rerank.py
│   │   ├── cite.py
│   │   └── answer.py
│   ├── baselines/
│   │   └── rule_based.py    # RuleBasedPolicy
│   ├── eval/
│   │   ├── evaluator.py      # 主评测器
│   │   └── metrics.py        # 指标计算
│   ├── train/               # SFT + RL 训练
│   │   ├── sft/
│   │   │   ├── dataset.py    # TrajectoryToSFTConverter, SFTDataset
│   │   │   └── trainer.py    # SFTTrainer
│   │   ├── rl/
│   │   │   ├── buffer.py     # TrajectoryBuffer, RolloutBuffer
│   │   │   ├── reward.py      # RewardFunction, AdaptiveKLController
│   │   │   └── grpo.py       # GRPOTrainer
│   │   └── data/
│   │       └── collector.py  # TrajectoryCollector, ExperienceDataset
│   ├── data/
│   │   └── loader.py
│   └── utils/
│       └── logging.py
├── configs/
│   └── default.yaml
├── scripts/
│   ├── run_env_demo.py       # MVP 演示
│   └── run_training.py       # 训练脚本
├── tests/
│   ├── test_env.py
│   ├── test_tools.py
│   └── test_baseline.py
└── pyproject.toml
```

## 核心接口

### 1. Environment

```python
from src.environment.env import ResearchEnv
from src.environment.corpus import CorpusStore
from src.tools.search import SearchTool
from src.tools.read import ReadTool
from src.tools.rerank import RerankTool
from src.tools.cite import CiteTool
from src.tools.answer import AnswerTool
from src.baselines.rule_based import RuleBasedPolicy

env = ResearchEnv(corpus=corpus, max_steps=15)
env.register_tool(SearchTool())
env.register_tool(ReadTool())
env.register_tool(RerankTool())
env.register_tool(CiteTool())
env.register_tool(AnswerTool())

obs = env.reset(task)
for _ in range(15):
    action = policy.decide(obs)
    obs, done, reason = env.step(action)
    if done:
        break
result = env.finalize_episode()
```

### 2. Evaluation

```python
from src.eval.evaluator import Evaluator

evaluator = Evaluator()
eval_result = evaluator.evaluate(episode_result, task_sample)
# eval_result.task_success, eval_result.citation_f1, eval_result.search_success_rate, etc.
```

### 3. SFT Training

```python
from src.train.sft.dataset import TrajectoryToSFTConverter, SFTDatasetWriter
from src.train.sft.trainer import SFTTrainer, SFTConfig
from src.train.data.collector import ExperienceDataset

# 收集数据
collector = TrajectoryCollector(env_factory, policy, task_sampler)
samples = collector.collect(n_episodes=100)

# 转换为 SFT 格式
dataset = ExperienceDataset()
for s in samples: dataset.add(s)
converter = TrajectoryToSFTConverter()
examples = dataset.to_sft_examples(converter)
SFTDatasetWriter("outputs/").write(examples)

# SFT 训练
trainer = SFTTrainer(model, tokenizer, "outputs/sft_data.jsonl", SFTConfig())
trainer.train()
```

### 4. RL Training (GRPO)

```python
from src.train.rl.grpo import GRPOTrainer, GRPOConfig
from src.train.rl.reward import RewardFunction, RewardConfig

reward_fn = RewardFunction(RewardConfig())
trainer = GRPOTrainer(
    actor_model=actor_model,
    ref_model=ref_model,
    tokenizer=tokenizer,
    env_factory=env_factory,
    reward_fn=reward_fn,
    config=GRPOConfig(),
)
trainer.train()
```

## 参考来源

| 模块 | 参考来源 | 说明 |
|------|---------|------|
| SFT Dataset | GAIR-NLP `verl/utils/dataset/sft_dataset.py` | role-based chat format, loss_mask |
| SFT Trainer | GAIR-NLP `verl/trainer/fsdp_sft_trainer.py` | HuggingFace Trainer, loss on response |
| RL Buffer | GAIR-NLP `verl/protocol.py` DataProto | TensorDict, batch 管理 |
| GRPO Trainer | GAIR-NLP `verl/trainer/ppo/ray_trainer.py` | GRPO advantage, PPO policy update |
| Reward | GAIR-NLP `tests/e2e/arithmetic_sequence/rl/main_trainer.py` | token-level reward, KL penalty |
| KL Controller | GAIR-NLP AdaptiveKLController | 论文 https://arxiv.org/pdf/1909.08593.pdf |
| deep-researcher (jackswl) | 轻量 research agent 形态 | 本项目 baseline policy 参考其 workflow |

## 评测指标

| 类型 | 指标 | 公式/说明 |
|------|------|---------|
| Outcome | task_success | answer_quality >= 0.5 AND citation_recall >= 0.3 |
| Outcome | answer_quality | keyword_coverage * 0.4 + structural * 0.3 + no_hallucination * 0.3 |
| Citation | citation_precision | cited ∩ gt / cited |
| Citation | citation_recall | cited ∩ gt / gt |
| Citation | citation_f1 | 2 * P * R / (P + R) |
| Process | search_success_rate | (n_searches - n_failed) / n_searches |
| Process | repeated_query_rate | 重复 query / 总 search |
| Process | tool_diversity | 不同 tool 类型数 / 5 |

## RL Reward 组成

```python
final_answer_reward = answer_quality * 2.0
citation_reward = citation_f1 * 1.5
evidence_gain_reward = sum(evidence_strength of new reads) * 1.0
repeated_action_penalty = -0.1 * n_repeated_actions
invalid_action_penalty = -0.2 * n_invalid_actions
efficiency_bonus = 0.5 if total_steps <= 6 else 0.0
total = sum of above
```

## 下一步

1. **真实数据**: 准备真实论文 corpus 和任务
2. **LLM Actor**: 替换 rule-based policy 为 LLM-based policy
3. **LLM Summarization**: 替换关键词提取为 LLM summarization
4. **BM25 Retrieval**: 替换简单文本匹配为 BM25/dense retrieval
5. **分布式训练**: 接入 DeepSpeed/accelerate 支持大规模训练

## License

MIT
