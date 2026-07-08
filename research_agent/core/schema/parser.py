import re
import json
from typing import Optional
from .action import Action

class ActionParser:
    @staticmethod
    def parse(text: str) -> Action:
        """
        Parses text to extract reasoning and Action.
        Supports standard JSON, markdown-wrapped JSON, and fallback key-value parsing.
        """
        # 1. Extract reasoning if present
        reasoning = None
        reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", text, re.DOTALL)
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()
        else:
            # Fallback reasoning: everything before <action>
            action_start = text.find("<action>")
            if action_start != -1:
                reasoning = text[:action_start].strip()
            if not reasoning:
                reasoning = ""

        # 2. Extract action block
        action_content = text
        action_match = re.search(r"<action>(.*?)</action>", text, re.DOTALL)
        if action_match:
            action_content = action_match.group(1).strip()

        # 3. Try to find json block inside ```json ... ```
        json_block_match = re.search(r"```json(.*?)```", action_content, re.DOTALL)
        if json_block_match:
            json_str = json_block_match.group(1).strip()
        else:
            # Try to find first { and last }
            brace_start = action_content.find("{")
            brace_end = action_content.rfind("}")
            if brace_start != -1 and brace_end != -1 and brace_start < brace_end:
                json_str = action_content[brace_start:brace_end+1].strip()
            else:
                json_str = action_content.strip()

        # 4. Try JSON parsing
        try:
            data = json.loads(json_str)
            tool = data.get("tool")
            params = data.get("params", {})
            intent = data.get("intent", f"Execute {tool}")
            
            if not tool:
                return Action(
                    tool="INVALID",
                    intent="Action missing 'tool' field",
                    params={},
                    reasoning=reasoning
                )
            
            return Action(
                tool=str(tool).upper(),
                intent=str(intent),
                params=params,
                reasoning=reasoning
            )
        except Exception as e:
            # 5. Fallback Key-Value parsing
            kv_dict = {}
            for line in action_content.split("\n"):
                line = line.strip()
                if not line or ":" not in line:
                    continue
                k, v = line.split(":", 1)
                k = k.strip().lower()
                v = v.strip()
                kv_dict[k] = v

            tool = kv_dict.get("tool", "").upper()
            if tool in {"SEARCH", "READ", "ANSWER"}:
                params = {}
                if tool == "SEARCH":
                    params["query"] = kv_dict.get("query", kv_dict.get("params", ""))
                    if "topk" in kv_dict:
                        try:
                            params["topk"] = int(kv_dict["topk"])
                        except ValueError:
                            pass
                elif tool == "READ":
                    cids_str = kv_dict.get("chunk_ids", "")
                    if cids_str.startswith("[") and cids_str.endswith("]"):
                        try:
                            params["chunk_ids"] = json.loads(cids_str.replace("'", '"'))
                        except Exception:
                            params["chunk_ids"] = [c.strip().strip('"').strip("'") for c in cids_str[1:-1].split(",") if c.strip()]
                    else:
                        params["chunk_ids"] = [c.strip() for c in cids_str.split(",") if c.strip()]
                elif tool == "ANSWER":
                    params["answer_text"] = kv_dict.get("answer_text", kv_dict.get("answer", ""))
                    ccids_str = kv_dict.get("cited_chunk_ids", "")
                    if ccids_str.startswith("[") and ccids_str.endswith("]"):
                        try:
                            params["cited_chunk_ids"] = json.loads(ccids_str.replace("'", '"'))
                        except Exception:
                            params["cited_chunk_ids"] = [c.strip().strip('"').strip("'") for c in ccids_str[1:-1].split(",") if c.strip()]
                    else:
                        params["cited_chunk_ids"] = [c.strip() for c in ccids_str.split(",") if c.strip()]
                
                return Action(
                    tool=tool,
                    intent=kv_dict.get("intent", f"Execute {tool}"),
                    params=params,
                    reasoning=reasoning
                )

            return Action(
                tool="INVALID",
                intent=f"Failed to parse action JSON: {e}",
                params={"raw_content": action_content},
                reasoning=reasoning
            )
