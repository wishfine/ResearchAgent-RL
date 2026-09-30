from __future__ import annotations

import json
import re

from .action import Action, VALID_TOOLS


class ActionParser:
    """Parse the single, machine-readable action emitted by an agent turn.

    An action is deliberately a narrow contract: exactly one ``<action>`` block
    containing one JSON object.  Natural-language, Markdown, and key-value
    fallbacks are not accepted because accepting them makes a rollout's action
    space ambiguous and unsuitable for later SFT/RL supervision.
    """

    @staticmethod
    def parse(text: str) -> Action:
        reasoning = ActionParser._extract_reasoning(text)
        action_blocks = re.findall(r"<action>(.*?)</action>", text, re.DOTALL)

        if len(action_blocks) != 1:
            return ActionParser._invalid(
                "Expected exactly one <action>...</action> JSON block", reasoning, text
            )

        try:
            payload = json.loads(action_blocks[0].strip())
        except json.JSONDecodeError as exc:
            return ActionParser._invalid(f"Invalid action JSON: {exc.msg}", reasoning, text)

        if not isinstance(payload, dict):
            return ActionParser._invalid("Action JSON must be an object", reasoning, text)

        tool = payload.get("tool")
        params = payload.get("params")
        if not isinstance(tool, str) or tool.upper() not in VALID_TOOLS:
            return ActionParser._invalid(
                f"Invalid tool: {tool!r}. Must be one of {sorted(VALID_TOOLS)}", reasoning, text
            )
        if not isinstance(params, dict):
            return ActionParser._invalid("Action 'params' must be an object", reasoning, text)

        intent = payload.get("intent", f"Execute {tool.upper()}")
        if not isinstance(intent, str):
            return ActionParser._invalid("Action 'intent' must be a string", reasoning, text)

        return Action(tool=tool.upper(), intent=intent, params=params, reasoning=reasoning)

    @staticmethod
    def _extract_reasoning(text: str) -> str:
        match = re.search(r"<reasoning>(.*?)</reasoning>", text, re.DOTALL)
        if match:
            return match.group(1).strip()

        action_start = text.find("<action>")
        return text[:action_start].strip() if action_start >= 0 else ""

    @staticmethod
    def _invalid(message: str, reasoning: str, raw_text: str) -> Action:
        return Action(
            tool="INVALID",
            intent=message,
            params={"raw_text": raw_text},
            reasoning=reasoning,
        )
