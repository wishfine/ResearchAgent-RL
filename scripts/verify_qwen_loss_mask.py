#!/usr/bin/env python3
from __future__ import annotations
import os
import sys

# Add project root to python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from research_agent.core.env.rollout_collector import RolloutSegment
from research_agent.core.env.mask_builder import build_loss_mask

def main():
    print("=" * 60)
    print("ResearchAgent-RL: Qwen Tokenizer Loss Mask Verification")
    print("=" * 60)

    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("[ERROR] 'transformers' is not installed in this environment.", file=sys.stderr)
        print("Please run this script on your GPU server where transformers is installed.", file=sys.stderr)
        sys.exit(1)

    import argparse
    parser = argparse.ArgumentParser(description="Qwen Tokenizer Loss Mask Verification")
    parser.add_argument("tokenizer_path", type=str, nargs="?", default="Qwen/Qwen2.5-7B-Instruct", 
                        help="HuggingFace model ID or local directory path containing Qwen tokenizer files")
    args = parser.parse_args()

    model_id = args.tokenizer_path
    print(f"Loading tokenizer from: {model_id} ...")
    
    # Enable local-only loading to prevent trying to hit the HF Hub if network is unreachable
    local_files_only = os.path.exists(model_id)
    if local_files_only:
        print("Detected local directory. Loading in offline mode.")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, 
        trust_remote_code=True, 
        local_files_only=local_files_only
    )
    print("Tokenizer loaded successfully.")

    # 1. Construct a mock multi-turn trajectory
    segments = [
        RolloutSegment(
            role="system",
            text="You are a helpful research assistant.",
            is_trainable=False
        ),
        RolloutSegment(
            role="user",
            text="Question: What is the capital of France?",
            is_trainable=False
        ),
        RolloutSegment(
            role="assistant",
            text="<reasoning>I need to search for France capital.</reasoning>\n<action>{\"tool\": \"SEARCH\", \"params\": {\"query\": \"France capital\"}}</action>",
            is_trainable=True
        ),
        RolloutSegment(
            role="observation",
            text="Tool Executed: SEARCH\nStatus: Success\nResults: Found chunk france_capital with text 'Paris is the capital of France.'",
            is_trainable=False
        ),
        RolloutSegment(
            role="assistant",
            text="<reasoning>I have the answer.</reasoning>\n<action>{\"tool\": \"ANSWER\", \"params\": {\"answer_text\": \"Paris\", \"cited_chunk_ids\": [\"france_capital\"]}}</action>",
            is_trainable=True
        )
    ]

    # 2. Build input_ids and loss_mask using real Qwen tokenizer
    input_ids, loss_mask = build_loss_mask(segments, tokenizer)

    # 3. Print token-by-token mask alignment
    print("\nToken-by-Token Alignment Check:")
    print(f"{'Token ID':<10} | {'Decoded Token':<30} | {'Loss Mask':<10}")
    print("-" * 60)
    
    for idx, (token_id, mask) in enumerate(zip(input_ids, loss_mask)):
        decoded = tokenizer.decode([token_id])
        # Format display string for escape chars
        decoded_rep = repr(decoded)[1:-1]
        
        # Color coding: Green for trainable, Grey for masked
        color_start = "\033[92m" if mask == 1 else "\033[90m"
        color_end = "\033[0m"
        
        print(f"{color_start}{token_id:<10} | {decoded_rep:<30} | {mask:<10}{color_end}")

    print("\n" + "=" * 60)
    print("Verification Script Finished!")
    print("=" * 60)

if __name__ == "__main__":
    main()
