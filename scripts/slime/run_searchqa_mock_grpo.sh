#!/bin/bash

# Ensure PYTHONPATH includes both the ResearchAgent directory and the local Slime installation
export PYTHONPATH=.:~/slime:$PYTHONPATH

# 1. Compile the task JSON files into a single JSONL file for the Slime dataset loader
python3 scripts/prepare_slime_dataset.py

# 2. Launch training via Slime's official entrypoint
# We pass the custom generator/reward paths, dataset path, and adapter arguments.
# If SGLang/Megatron server configuration complains about missing parameters,
# append options like: --hf-checkpoint /home/zhangyonglin/models/models/Qwen--Qwen3.5-9B/snapshots/master
python3 -m slime.entrypoints.train \
    --custom-generate-function-path research_agent.adapters.slime.custom_generator.custom_generate \
    --custom-rm-path research_agent.adapters.slime.custom_reward.custom_rm \
    --rollout-global-dataset data/searchqa_debug/rollout_dataset.jsonl \
    --n-samples-per-prompt 2 \
    --rollout-batch-size 2 \
    --over-sampling-batch-size 2 \
    --actor-model-url mock \
    --corpus-dir data/searchqa_debug/corpus \
    --max-steps 4 \
    --reward-type contains \
    --step-penalty 0.01 \
    --invalid-penalty 0.05 \
    --group-rm False
