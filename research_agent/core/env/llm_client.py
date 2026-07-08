from __future__ import annotations
import urllib.request
import urllib.error
import json
import re
from typing import List, Optional

class LLMClient:
    def __init__(self, api_url: str = "http://localhost:8000/v1", api_key: Optional[str] = None):
        """
        OpenAI-compatible HTTP completion client using standard python libraries.
        - Supports /v1/completions (preferred for agent prefix injection).
        - Supports /v1/chat/completions (fallback for chat APIs).
        """
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        stop_tokens: List[str] = None
    ) -> str:
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
            if stop_tokens:
                data["stop"] = stop_tokens

        # Set default model key (ignored by most local servers, needed for strict API endpoints)
        data["model"] = "default"

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
                    return resp_data["choices"][0]["message"]["content"]
                else:
                    return resp_data["choices"][0]["text"]
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
