"""Ordered SEARCH-observation signatures for shared-prefix experiments.

The key is the ordered sequence of chunk IDs visible in a SEARCH observation.
It is stricter than set/Jaccard similarity, but does not by itself prove full
MDP-state equivalence: raw query text remains in the transcript, and retrieval
scores can affect a later RERANK even when IDs and their order match.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

EffectKey = tuple[str, ...]


@dataclass(frozen=True)
class SearchProposal:
    query: str
    effect: EffectKey


def effect_key(chunk_ids: Iterable[str]) -> EffectKey:
    """Preserve retrieval order; reject malformed or duplicate observations."""
    ids = tuple(chunk_ids)
    if any(not isinstance(chunk_id, str) or not chunk_id for chunk_id in ids):
        raise ValueError("effect chunk IDs must be non-empty strings")
    if len(ids) != len(set(ids)):
        raise ValueError("an exact SEARCH observation cannot repeat a chunk ID")
    return ids


def group_by_effect(proposals: Sequence[SearchProposal]) -> dict[EffectKey, list[int]]:
    """Return all proposal indices in each exact tool-effect class."""
    groups: dict[EffectKey, list[int]] = defaultdict(list)
    for index, proposal in enumerate(proposals):
        groups[proposal.effect].append(index)
    return dict(groups)


def shared_effect_advantages(
    proposals: Sequence[SearchProposal],
    effect_returns: Mapping[EffectKey, Sequence[float]],
    *,
    state_baseline: float,
) -> list[float]:
    """Map independently sampled continuation returns back to query actions.

    The caller must supply a baseline that depends on the shared prefix only,
    not on the sampled query. The estimator is justified only when continuations
    are sampled from an identical post-tool state for every query in a class.
    Current Vime transcripts retain raw query text and candidate scores, so
    this function is an isolated estimator primitive, NOT an enabled training
    path or an unbiased estimator for the current environment.
    """
    if not proposals:
        return []
    values: dict[EffectKey, float] = {}
    for effect in group_by_effect(proposals):
        rewards = effect_returns.get(effect)
        if not rewards:
            raise ValueError(f"missing continuation rewards for effect {effect!r}")
        values[effect] = sum(float(reward) for reward in rewards) / len(rewards)
    return [values[proposal.effect] - state_baseline for proposal in proposals]


def jaccard(left: EffectKey, right: EffectKey) -> float:
    """Diagnostic only: overlap does not establish effect equivalence."""
    left_set, right_set = set(left), set(right)
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 1.0


def finite_effect_kl_decomposition(
    old_policy: Mapping[str, float],
    new_policy: Mapping[str, float],
    action_effects: Mapping[str, EffectKey],
) -> tuple[float, float, float]:
    """Exact KL chain rule on an explicitly enumerated finite action set.

    Returns (full_action_kl, between_effect_kl, within_effect_kl). This is a
    mathematical diagnostic, not an estimator of an LLM's full action-space KL.
    Both policies must be normalized and the new policy must have support under
    the old policy so all returned terms are finite.
    """
    actions = set(old_policy)
    if actions != set(new_policy) or actions != set(action_effects) or not actions:
        raise ValueError("Policies and effects must share a non-empty action set")
    for policy in (old_policy, new_policy):
        if any(not math.isfinite(value) or value < 0 for value in policy.values()):
            raise ValueError("Policy probabilities must be finite and non-negative")
        if not math.isclose(sum(policy.values()), 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("Policy probabilities must sum to one")
    if any(new_policy[action] > 0 and old_policy[action] == 0 for action in actions):
        raise ValueError("New policy must be absolutely continuous under old policy")

    old_effects: dict[EffectKey, float] = defaultdict(float)
    new_effects: dict[EffectKey, float] = defaultdict(float)
    for action in actions:
        effect = action_effects[action]
        old_effects[effect] += old_policy[action]
        new_effects[effect] += new_policy[action]

    full = sum(new_policy[action] * math.log(new_policy[action] / old_policy[action])
               for action in actions if new_policy[action] > 0)
    between = sum(new_mass * math.log(new_mass / old_effects[effect])
                  for effect, new_mass in new_effects.items() if new_mass > 0)
    within = sum(
        new_policy[action] * math.log(
            (new_policy[action] / new_effects[action_effects[action]])
            / (old_policy[action] / old_effects[action_effects[action]])
        )
        for action in actions if new_policy[action] > 0
    )
    return full, between, within
