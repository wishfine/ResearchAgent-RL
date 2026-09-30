from __future__ import annotations
from typing import List, Tuple, Any
from .rollout_collector import RolloutSegment

class MockTokenizer:
    """A character-level tokenizer for CPU-only testing of span boundary mapping."""
    def encode(self, text: str) -> List[int]:
        return [ord(char) for char in text]

    def decode(self, ids: List[int]) -> str:
        return "".join(chr(i) for i in ids)


def build_loss_mask(segments: List[RolloutSegment], tokenizer: Any) -> Tuple[List[int], List[int]]:
    """
    Computes input_ids and loss_mask for a sequence of RolloutSegments.
    - Model outputs (Assistant inner thoughts/actions) have loss_mask = 1.
    - Prompts, system text, and environment observations have loss_mask = 0.
    """
    input_ids = []
    loss_mask = []

    for seg in segments:
        if seg.is_trainable:
            # Model turn: Wrap in ChatML markers, but only compute loss on the generated inner text.
            prefix = "<|im_start|>assistant\n"
            suffix = "<|im_end|>\n"
            
            prefix_ids = tokenizer.encode(prefix)
            inner_ids = tokenizer.encode(seg.text)
            suffix_ids = tokenizer.encode(suffix)
            
            input_ids.extend(prefix_ids + inner_ids + suffix_ids)
            loss_mask.extend([0] * len(prefix_ids) + [1] * len(inner_ids) + [0] * len(suffix_ids))
        else:
            # Non-trainable turn (system, user query, or environment observation): entire span has loss = 0.
            seg_text = seg.to_chatml()
            ids = tokenizer.encode(seg_text)
            input_ids.extend(ids)
            loss_mask.extend([0] * len(ids))

    return input_ids, loss_mask


def build_labels(input_ids: List[int], loss_mask: List[int], ignore_index: int = -100) -> List[int]:
    """
    Converts input_ids and loss_mask into standard PyTorch labels
    where non-trainable tokens are replaced by ignore_index (-100).
    """
    labels = []
    for token_id, mask in zip(input_ids, loss_mask):
        if mask == 1:
            labels.append(token_id)
        else:
            labels.append(ignore_index)
    return labels
