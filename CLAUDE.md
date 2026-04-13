# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this repository.

## Project Overview

ResearchAgent-RL 是一个基于强化学习的科研任务求解 Agent 项目。它将科研问答建模为有限步长的 tool-use MDP，通过 SEARCH → READ → RERANK → CITE → ANSWER 动作序列完成推理感知检索、证据积累和引用归因。

## Core Architecture

- **MDP Environment**: `env/environment.py` - 定义 Observation/Action/Transition/Reward/Done
- **Tool System**: `tools/base.py` - 工具注册表和 5 个工具实现
- **Policy**: `policy/policy.py` - RandomPolicy, RuleBasedPolicy, LLMPolicy
- **Data Schemas**: `data/schemas.py` - 所有核心 dataclass 定义
- **Evaluator**: `eval/evaluator.py` - 双层评测（结果指标 + 过程指标）

## Tech Stack

- **SFT**: Hugging Face Transformers + PEFT (LoRA)
- **RL**: veRL + Ray + vLLM
- **Retrieval**: Local chunk library (BM25/embedding)

## Training Stages

1. **Stage 0**: Environment + Mock Tools（验证 MDP）
2. **Stage 1**: Rule-based Baseline（生成训练轨迹）
3. **Stage 2**: SFT Warm Start（Transformers + PEFT）
4. **Stage 3**: RL Post-training（veRL + Ray + vLLM）
5. **Stage 4**: Evaluation + Ablation

## MVP Verification

```bash
python scripts/run_mvp.py
```

这会验证环境可运行，rule-based baseline 可执行，评测系统可输出指标。

## Key Files

| File | Purpose |
|------|---------|
| `data/schemas.py` | 所有 dataclass 定义 |
| `env/environment.py` | MDP 环境，step/reset/finalize_episode |
| `tools/base.py` | ToolRegistry + 5 工具实现 |
| `policy/policy.py` | Policy 接口和三种实现 |
| `eval/evaluator.py` | Outcome + Process 双层评测 |
| `scripts/run_mvp.py` | MVP 入口脚本 |

## Action Schema

```python
{
  "tool": "SEARCH|READ|RERANK|CITE|ANSWER",
  "intent": str,
  "params": {...},  # tool-specific parameters
  "reasoning": Optional[str]
}
```

## Reward Formula

```
R_total = 1.0 * R_final + 0.3 * R_citation + 0.1 * R_step_sum
```

- R_final: 答案质量（关键词覆盖 + Section 覆盖 + 参考答案 overlap）
- R_citation: 0.5*recall + 0.5*precision
- R_step_sum: 过程奖励（证据获取、效率奖金、重复 penalty）

## Termination Conditions

- ANSWER 执行
- remaining_steps <= 0
- invalid_action_count >= 3
- no_progress_count >= 5

## No Build System

纯 Python 项目，无 package.json / requirements.txt / Makefile。直接 import 使用。
