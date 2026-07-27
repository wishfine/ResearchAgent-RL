from __future__ import annotations
import urllib.request
import urllib.error
import json
import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class LLMResponse:
    """Content and accounting information returned by an OpenAI-compatible API."""

    content: str
    reasoning: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0

class LLMClient:
    def __init__(
        self,
        api_url: str = "http://localhost:8000/v1",
        api_key: Optional[str] = None,
        model: str = "default",
    ):
        """
        OpenAI-compatible HTTP completion client using standard python libraries.
        - Supports /v1/completions (preferred for agent prefix injection).
        - Supports /v1/chat/completions (fallback for chat APIs).
        """
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.model = model

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: Optional[float] = None,
        model: Optional[str] = None,
        stop_tokens: List[str] = None
    ) -> str:
        return self.generate_response(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            model=model,
            stop_tokens=stop_tokens,
        ).content

    def generate_response(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: Optional[float] = None,
        model: Optional[str] = None,
        stop_tokens: List[str] = None,
    ) -> LLMResponse:
        """Generate one turn and retain server-provided usage and reasoning fields."""
        # 1. Determine if we are using chat or raw completion
        is_chat = "chat/completions" in self.api_url or "/chat" in self.api_url
        
        if is_chat:
            target_url = self.api_url
            if not target_url.endswith("/chat/completions"):
                if target_url.endswith("/v1"):
                    target_url += "/chat/completions"
                else:
                    target_url += "/v1/chat/completions"
            messages = self._prompt_to_chat_messages(prompt)
            data = {
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            if top_p is not None:
                data["top_p"] = top_p
            if stop_tokens:
                data["stop"] = stop_tokens
        else:
            target_url = self.api_url
            if not target_url.endswith("/completions"):
                if target_url.endswith("/v1"):
                    target_url += "/completions"
                else:
                    target_url += "/v1/completions"
            data = {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            if top_p is not None:
                data["top_p"] = top_p
            if stop_tokens:
                data["stop"] = stop_tokens

        data["model"] = model or self.model

        req = urllib.request.Request(
            target_url,
            data=json.dumps(data).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")

        try:
            # We set a reasonable timeout of 60 seconds for generation
            with urllib.request.urlopen(req, timeout=60) as response:
                resp_data = json.loads(response.read().decode("utf-8"))
                if is_chat:
                    message = resp_data["choices"][0]["message"]
                    content = message.get("content") or ""
                    reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
                else:
                    content = resp_data["choices"][0]["text"]
                    reasoning = ""

                usage = resp_data.get("usage") or {}
                return LLMResponse(
                    content=content,
                    reasoning=reasoning,
                    prompt_tokens=int(usage.get("prompt_tokens") or 0),
                    completion_tokens=int(usage.get("completion_tokens") or 0),
                )
        except urllib.error.HTTPError as e:
            err_content = e.read().decode("utf-8") if e else str(e)
            raise RuntimeError(f"LLM API HTTP Error {e.code}: {err_content}")
        except Exception as e:
            raise RuntimeError(f"Failed to connect to LLM API: {e}")

    def _prompt_to_chat_messages(self, prompt: str) -> List[dict]:
        """Converts ChatML raw prompt back into a standard list of chat messages, mapping observation to user."""
        messages = []
        # Match <|im_start|>role\ncontent<|im_end|>
        pattern = re.compile(r"<\|im_start\|>(\w+)\n(.*?)(?:<\|im_end\|>|$)", re.DOTALL)
        matches = pattern.findall(prompt)
        for role, content in matches:
            content = content.strip()
            # Standard APIs like OpenAI do not support 'observation' role; map to 'user' instead.
            if role == "observation":
                role = "user"
            messages.append({"role": role, "content": content})
        return messages
