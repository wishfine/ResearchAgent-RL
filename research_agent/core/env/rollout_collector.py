from __future__ import annotations
from dataclasses import dataclass, field
from typing import List

@dataclass
class RolloutSegment:
    role: str           # "system", "user", "assistant", "observation"
    text: str           # Raw text content of the message
    is_trainable: bool  # True if model generates this segment (reasoning/action), False otherwise

    def to_chatml(self) -> str:
        return f"<|im_start|>{self.role}\n{self.text}<|im_end|>\n"


class ConversationCollector:
    def __init__(self, system_prompt: str = """You are a document-grounded research agent with SEARCH, READ, RERANK, CITE, and ANSWER tools.
Return exactly one single-line action. Do not emit Markdown fences, prose, or a second JSON object.
The response must be `<action>{JSON}</action>` and JSON must use one of these exact parameter schemas:
SEARCH: <action>{"tool":"SEARCH","intent":"...","params":{"query":"...","topk":3}}</action>
READ: <action>{"tool":"READ","intent":"...","params":{"chunk_ids":["..."]}}</action>
RERANK: <action>{"tool":"RERANK","intent":"...","params":{"query":"...","candidate_chunk_ids":["..."],"topk":3}}</action>
CITE: <action>{"tool":"CITE","intent":"...","params":{"chunk_ids":["..."],"claims":["..."]}}</action>
ANSWER: <action>{"tool":"ANSWER","intent":"...","params":{"answer_text":"...","cited_chunk_ids":["..."]}}</action>
Follow SEARCH -> READ -> CITE -> ANSWER. Never call ANSWER with an empty cited_chunk_ids list.
Only cite chunk IDs returned by READ. Do not answer from common knowledge; use the retrieved evidence.
Keep every action field inside params: never put action fields beside params.
After a successful SEARCH with candidates, READ a returned chunk next; do not repeat SEARCH. After READ, use CITE; after CITE, use ANSWER."""):
        self.segments: List[RolloutSegment] = []
        # Add system prompt as non-trainable
        self.segments.append(RolloutSegment(
            role="system",
            text=system_prompt,
            is_trainable=False
        ))

    def add_user_message(self, query: str) -> None:
        self.segments.append(RolloutSegment(
            role="user",
            text=f"Question: {query}",
            is_trainable=False
        ))

    def add_assistant_response(self, text: str) -> None:
        self.segments.append(RolloutSegment(
            role="assistant",
            text=text,
            is_trainable=True
        ))

    def add_observation(self, tool_name: str, success: bool, data: any, error: str = "") -> None:
        """Formats the tool output into a clean string and records it as a non-trainable observation segment."""
        obs_lines = [f"Tool Executed: {tool_name}"]
        if not success:
            obs_lines.append(f"Status: Failed")
            obs_lines.append(f"Error: {error}")
        else:
            obs_lines.append(f"Status: Success")
            if tool_name == "SEARCH":
                candidates = data.get("candidates", [])
                if not candidates:
                    obs_lines.append("Results: No relevant passages found.")
                else:
                    obs_lines.append(f"Results: Found {len(candidates)} candidate chunks:")
                    for c in candidates:
                        obs_lines.append(f"  - Chunk ID: {c['chunk_id']} (Doc ID: {c['doc_id']}) | Title: {c['title']}")
                        obs_lines.append(f"    Snippet: {c['snippet']}")
            elif tool_name == "READ":
                summaries = data.get("summaries", [])
                obs_lines.append(f"Results: Read {len(summaries)} chunks:")
                for s in summaries:
                    obs_lines.append(f"  - Chunk ID: {s['chunk_id']}")
                    obs_lines.append(f"    Summary: {s['summary']}")
            elif tool_name == "ANSWER":
                obs_lines.append(f"Answer Submitted: {data.get('answer_text')}")
                obs_lines.append(f"Citations: {data.get('cited_chunk_ids', [])}")
            else:
                obs_lines.append(f"Result Data: {str(data)}")

        obs_text = "\n".join(obs_lines)
        
        self.segments.append(RolloutSegment(
            role="observation",
            text=obs_text,
            is_trainable=False
        ))

    def get_full_text(self) -> str:
        """Assembles the entire chat sequence into a single string using ChatML formatting."""
        return "".join(seg.to_chatml() for seg in self.segments)

    def get_prompt_for_generation(self) -> str:
        """Returns the assembled context prompt up to the start of the next assistant response."""
        full_text = self.get_full_text()
        # Append the assistant start token to prompt the model's next turn
        return full_text + "<|im_start|>assistant\n"
