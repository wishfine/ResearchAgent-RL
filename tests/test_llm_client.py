from __future__ import annotations
import unittest
from unittest.mock import patch, MagicMock
import json
import io
from urllib.error import HTTPError

from research_agent.core.env.llm_client import LLMClient

class TestLLMClient(unittest.TestCase):
    def test_prompt_to_chat_messages_conversion(self):
        client = LLMClient()
        prompt = (
            "<|im_start|>system\nYou are an agent.<|im_end|>\n"
            "<|im_start|>user\nWho are you?<|im_end|>\n"
            "<|im_start|>observation\nResult: 42<|im_end|>\n"
        )
        messages = client._prompt_to_chat_messages(prompt)
        expected = [
            {"role": "system", "content": "You are an agent."},
            {"role": "user", "content": "Who are you?"},
            {"role": "user", "content": "Result: 42"},  # mapped from observation to user
        ]
        self.assertEqual(messages, expected)

    @patch("urllib.request.urlopen")
    def test_generate_completions_endpoint(self, mock_urlopen):
        # Setup mock response for raw completions
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "choices": [{"text": "Hello, world!"}]
        }).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_response

        client = LLMClient(api_url="http://localhost:8000/v1", model="Qwen3.5-9B")
        response = client.generate(
            "test prompt",
            max_tokens=10,
            temperature=0.5,
            top_p=0.9,
            model="per-request-model",
            stop_tokens=["\n"],
        )
        
        self.assertEqual(response, "Hello, world!")
        
        # Verify urlopen arguments
        mock_urlopen.assert_called_once()
        req_arg = mock_urlopen.call_args[0][0]
        self.assertEqual(req_arg.full_url, "http://localhost:8000/v1/completions")
        self.assertEqual(req_arg.headers["Content-type"], "application/json")
        
        req_body = json.loads(req_arg.data.decode("utf-8"))
        self.assertEqual(req_body["prompt"], "test prompt")
        self.assertEqual(req_body["max_tokens"], 10)
        self.assertEqual(req_body["temperature"], 0.5)
        self.assertEqual(req_body["top_p"], 0.9)
        self.assertEqual(req_body["model"], "per-request-model")
        self.assertEqual(req_body["stop"], ["\n"])

    @patch("urllib.request.urlopen")
    def test_generate_chat_endpoint(self, mock_urlopen):
        # Setup mock response for chat completions
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({
            "choices": [{"message": {"content": "I am Qwen."}}]
        }).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_response

        # Use an explicit /chat endpoint URL
        client = LLMClient(api_url="http://localhost:8000/v1/chat/completions")
        prompt = "<|im_start|>user\nWho are you?<|im_end|>\n"
        response = client.generate(prompt)
        
        self.assertEqual(response, "I am Qwen.")
        
        mock_urlopen.assert_called_once()
        req_arg = mock_urlopen.call_args[0][0]
        self.assertEqual(req_arg.full_url, "http://localhost:8000/v1/chat/completions")
        
        req_body = json.loads(req_arg.data.decode("utf-8"))
        self.assertEqual(req_body["messages"], [{"role": "user", "content": "Who are you?"}])

    @patch("urllib.request.urlopen")
    def test_generate_api_error_handling(self, mock_urlopen):
        # Mock HTTP Error 500
        mock_error = HTTPError(
            url="http://localhost:8000/v1/completions",
            code=500,
            msg="Internal Server Error",
            hdrs=None,
            fp=io.BytesIO(b"Out of memory")
        )
        mock_urlopen.side_effect = mock_error

        client = LLMClient()
        with self.assertRaises(RuntimeError) as context:
            client.generate("test")
        self.assertIn("LLM API HTTP Error 500: Out of memory", str(context.exception))

if __name__ == "__main__":
    unittest.main()
