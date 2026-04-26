import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app import create_app
from config import Config
from provider import genai


class _TokenManager:
    def get_token(self):
        return "test-token"

    def force_refresh(self):
        return "refreshed-token"


class _FakeResponse:
    status_code = 200
    text = ""

    def __init__(self, lines):
        self._lines = lines

    def iter_lines(self):
        return iter(self._lines)


class UsageAccountingTests(unittest.TestCase):
    def test_extract_usage_normalizes_genai_camel_case_fields(self):
        usage = genai.extract_usage_from_genai({
            "other": json.dumps({
                "inputTokens": "12",
                "outputTokens": 34,
                "reasoningTokens": 5,
                "cacheTokens": 6,
                "totalTokens": 51,
            })
        })

        self.assertEqual(usage["prompt_tokens"], 12)
        self.assertEqual(usage["completion_tokens"], 34)
        self.assertEqual(usage["total_tokens"], 51)
        self.assertEqual(
            usage["completion_tokens_details"]["reasoning_tokens"],
            5,
        )
        self.assertEqual(usage["prompt_tokens_details"]["cached_tokens"], 6)

    def test_stream_sends_prompt_token_estimate_and_preserves_upstream_usage(self):
        calls = []

        def fake_post(*args, **kwargs):
            calls.append((args, kwargs))
            return _FakeResponse([
                b'data: {"choices":[{"delta":{"content":"hello"},"finish_reason":null}]}',
                b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
                b'data: {"other":"{\\"promptTokens\\":11}"}',
            ])

        config = SimpleNamespace(token_manager=_TokenManager())
        messages = [{"role": "user", "content": "你好，请介绍 GenAI"}]

        with patch.object(genai.model_registry, "get_root_ai_type", return_value="xinference"), \
                patch.object(genai.requests, "post", side_effect=fake_post):
            chunks = list(genai.stream_genai_response(
                chat_info="你好，请介绍 GenAI",
                messages=messages,
                model="deepseek-pro",
                max_tokens=None,
                config=config,
            ))

        self.assertGreater(calls[0][1]["json"]["promptTokens"], 0)

        final_payload = json.loads(chunks[-2].removeprefix("data: "))
        self.assertEqual(final_payload["choices"][0]["finish_reason"], "stop")
        self.assertEqual(final_payload["usage"]["prompt_tokens"], 11)
        self.assertGreater(final_payload["usage"]["completion_tokens"], 0)
        self.assertEqual(
            final_payload["usage"]["total_tokens"],
            final_payload["usage"]["prompt_tokens"] + final_payload["usage"]["completion_tokens"],
        )
        self.assertEqual(chunks[-1], "data: [DONE]\n\n")

    def test_non_stream_chat_response_uses_stream_usage(self):
        def fake_stream(*_args, **_kwargs):
            yield 'data: {"choices":[{"delta":{"content":"hello"},"finish_reason":null}]}\n\n'
            yield (
                'data: {"choices":[{"delta":{},"finish_reason":"stop"}],'
                '"usage":{"prompt_tokens":9,"completion_tokens":4,"total_tokens":13}}\n\n'
            )
            yield "data: [DONE]\n\n"

        app = create_app(Config(
            token_manager=_TokenManager(),
            port=5000,
            api_key=None,
            debug=False,
        ))

        with patch("api.chat.stream_genai_response", side_effect=fake_stream):
            response = app.test_client().post("/v1/chat/completions", json={
                "model": "deepseek-pro",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": False,
            })

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["choices"][0]["message"]["content"], "hello")
        self.assertEqual(payload["usage"], {
            "prompt_tokens": 9,
            "completion_tokens": 4,
            "total_tokens": 13,
        })


if __name__ == "__main__":
    unittest.main()
