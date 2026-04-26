import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import requests

from provider import genai


class _TokenManager:
    def get_token(self):
        return "test-token"

    def force_refresh(self):
        return "refreshed-token"


class GenAITimeoutTests(unittest.TestCase):
    def test_read_timeout_uses_configured_timeout_and_returns_clean_error_chunk(self):
        calls = []

        def fake_post(*args, **kwargs):
            calls.append((args, kwargs))
            raise requests.exceptions.ReadTimeout(
                "HTTPSConnectionPool(host='genai.shanghaitech.edu.cn', port=443): "
                "Read timed out. (read timeout=60)"
            )

        config = SimpleNamespace(token_manager=_TokenManager())

        with patch.object(genai.model_registry, "get_root_ai_type", return_value="xinference"), \
                patch.object(genai.requests, "post", side_effect=fake_post):
            chunks = list(genai.stream_genai_response(
                chat_info="hello",
                messages=[{"role": "user", "content": "hello"}],
                model="deepseek-pro",
                max_tokens=None,
                config=config,
            ))

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1]["timeout"], genai.GENAI_REQUEST_TIMEOUT)
        self.assertEqual(len(chunks), 1)
        self.assertIn("data: [DONE]", chunks[0])

        data_line = chunks[0].split("\n\n", 1)[0]
        payload = json.loads(data_line.removeprefix("data: "))
        choice = payload["choices"][0]

        self.assertEqual(choice["finish_reason"], "error")
        message = choice["delta"]["content"]
        self.assertIn("Upstream GenAI read timed out after", message)
        self.assertIn("request may be too large", message)
        self.assertNotIn("HTTPSConnectionPool", message)


if __name__ == "__main__":
    unittest.main()
