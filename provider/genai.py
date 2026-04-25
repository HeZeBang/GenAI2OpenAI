import json
import logging
import uuid
from datetime import datetime

import requests

from config import GENAI_URL, build_genai_headers, model_registry
from errors import make_error_chunk
from tools.parsing import extract_tool_calls, _tag_prefix_len

logger = logging.getLogger(__name__)


def convert_messages_to_genai_format(messages):
    chat_info = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            chat_info = msg.get("content", "")
            break
    return chat_info


def extract_content_from_genai(response_data):
    try:
        if "choices" in response_data and len(response_data["choices"]) > 0:
            delta = response_data["choices"][0].get("delta", {})
            content = delta.get("content") or None
            reasoning = delta.get("reasoning_content") or None
            return content, reasoning
    except (KeyError, IndexError, TypeError):
        pass
    return None, None


def stream_genai_response(chat_info, messages, model, max_tokens, config):
    token = config.token_manager.get_token()
    root_ai_type = model_registry.get_root_ai_type(model, token)
    headers = build_genai_headers(token)

    genai_data = {
        "chatInfo": "",
        "messages": messages,
        "type": "3",
        "stream": True,
        "aiType": model,
        "aiSecType": "1",
        "promptTokens": 0,
        "rootAiType": root_ai_type,
        "maxToken": max_tokens or 30000
    }

    logger.debug("=== GenAI Request ===")
    logger.debug("Model: %s, rootAiType: %s", model, root_ai_type)
    logger.debug("Messages count: %d", len(messages))
    for i, msg in enumerate(messages):
        role = msg.get('role', '?')
        content = msg.get('content', '')
        preview = (content[:200] + '...') if content and len(content) > 200 else content
        logger.debug("  [%d] role=%s, content=%s", i, role, preview)

    try:
        response = requests.post(
            GENAI_URL,
            headers=headers,
            json=genai_data,
            stream=True,
            timeout=60
        )

        logger.debug("GenAI Response Status: %d", response.status_code)

        if response.status_code == 401:
            new_token = config.token_manager.force_refresh()
            if new_token:
                logger.info("Token refreshed after 401, retrying request")
                headers = build_genai_headers(new_token)
                response = requests.post(
                    GENAI_URL, headers=headers, json=genai_data,
                    stream=True, timeout=60
                )

        if response.status_code != 200:
            logger.warning("GenAI API error %d: %s", response.status_code, response.text[:500])
            if response.status_code == 401:
                yield make_error_chunk("Upstream authentication failed", model)
            elif response.status_code == 429:
                yield make_error_chunk("Upstream rate limit exceeded", model)
            else:
                yield make_error_chunk(f"Upstream API error: {response.status_code}", model)
            return

        finished = False
        line_count = 0
        for line in response.iter_lines():
            if finished:
                break

            if line:
                try:
                    line_str = line.decode('utf-8') if isinstance(line, bytes) else line

                    if line_count < 5:
                        logger.debug("Raw line [%d]: %s", line_count, line_str[:300])
                    line_count += 1

                    if line_str.startswith('data:'):
                        line_str = line_str[5:].strip()

                    if line_str:
                        genai_json = json.loads(line_str)

                        if isinstance(genai_json, dict) and genai_json.get("success") is False:
                            err_msg = genai_json.get("message", "Unknown upstream error")
                            err_code = genai_json.get("code", 500)
                            logger.warning("GenAI business error (code=%s): %s", err_code, err_msg)
                            yield make_error_chunk(f"Upstream error: {err_msg}", model)
                            return

                        if "choices" in genai_json and len(genai_json["choices"]) > 0:
                            choice = genai_json["choices"][0]
                            if choice.get("finish_reason") is not None:
                                finished = True

                        if finished:
                            final_response = {
                                "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                                "object": "chat.completion.chunk",
                                "created": int(datetime.now().timestamp()),
                                "model": model,
                                "choices": [{
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": "stop"
                                }]
                            }
                            yield f"data: {json.dumps(final_response)}\n\n"
                            yield "data: [DONE]\n\n"
                            break

                        content, reasoning = extract_content_from_genai(genai_json)

                        delta = {}
                        if content:
                            delta["content"] = content
                        if reasoning:
                            delta["reasoning_content"] = reasoning

                        if delta:
                            openai_response = {
                                "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                                "object": "chat.completion.chunk",
                                "created": int(datetime.now().timestamp()),
                                "model": model,
                                "choices": [{
                                    "index": 0,
                                    "delta": delta,
                                    "finish_reason": None
                                }]
                            }
                            yield f"data: {json.dumps(openai_response)}\n\n"

                except json.JSONDecodeError as e:
                    logger.debug("JSON decode error: %s, line: %s", e, line_str[:200])

        logger.debug("Total lines received: %d, finished: %s", line_count, finished)

        final_response = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion.chunk",
            "created": int(datetime.now().timestamp()),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }]
        }
        yield f"data: {json.dumps(final_response)}\n\n"
        yield "data: [DONE]\n\n"

    except Exception as e:
        logger.exception("Error in stream_genai_response")
        yield make_error_chunk(str(e), model)


def stream_genai_response_with_tools(chat_info, messages, model, max_tokens, config):
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(datetime.now().timestamp())

    OPEN_TAG = "<tool_call>"

    buffer = ""
    tool_buffer = ""
    sent_role = False
    tool_detected = False

    def make_chunk(delta, finish_reason=None):
        chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason
            }]
        }
        return f"data: {json.dumps(chunk)}\n\n"

    def emit_text(text):
        nonlocal sent_role
        delta = {"content": text}
        if not sent_role:
            delta["role"] = "assistant"
            sent_role = True
        return make_chunk(delta)

    for line in stream_genai_response(chat_info, messages, model, max_tokens, config):
        if not line.startswith("data: "):
            continue
        data_str = line[6:].strip()
        if data_str == "[DONE]":
            continue
        try:
            data = json.loads(data_str)
        except json.JSONDecodeError:
            continue
        if "choices" not in data or not data["choices"]:
            continue
        chunk_delta = data["choices"][0].get("delta", {})
        content = chunk_delta.get("content", "")
        if not content:
            continue

        if tool_detected:
            tool_buffer += content
            continue

        buffer += content

        tag_pos = buffer.find(OPEN_TAG)
        if tag_pos >= 0:
            pre = buffer[:tag_pos]
            if pre.strip():
                yield emit_text(pre)

            tool_detected = True
            tool_buffer = buffer[tag_pos:]
            buffer = ""
            continue

        plen = _tag_prefix_len(buffer, OPEN_TAG)
        if plen > 0:
            safe = buffer[:-plen]
            if safe:
                yield emit_text(safe)
            buffer = buffer[-plen:]
        else:
            if buffer:
                yield emit_text(buffer)
            buffer = ""

    if tool_detected:
        tool_calls, remaining = extract_tool_calls(tool_buffer)

        if tool_calls:
            logger.debug("Streaming tool calling: detected %d tool_call(s)", len(tool_calls))

            if remaining and remaining.strip():
                yield emit_text(remaining.strip())

            if not sent_role:
                yield make_chunk({"role": "assistant"})
                sent_role = True

            for i, tc in enumerate(tool_calls):
                yield make_chunk({
                    "tool_calls": [{
                        "index": i,
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["function"]["name"],
                            "arguments": tc["function"]["arguments"]
                        }
                    }]
                })

            yield make_chunk({}, finish_reason="tool_calls")
            yield "data: [DONE]\n\n"
        else:
            logger.warning("Tool tag detected but parsing failed — emitting as text")
            yield emit_text(tool_buffer)
            yield make_chunk({}, finish_reason="stop")
            yield "data: [DONE]\n\n"
    else:
        if buffer:
            yield emit_text(buffer)

        if not sent_role:
            yield make_chunk({"role": "assistant", "content": ""})

        yield make_chunk({}, finish_reason="stop")
        yield "data: [DONE]\n\n"
