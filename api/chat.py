import json
import logging
import time
import uuid
from datetime import datetime

from flask import Blueprint, current_app, request, jsonify, stream_with_context, Response

from config import TokenExpiredError
from errors import openai_error
from tools.prompts import inject_tool_prompt
from tools.parsing import extract_tool_calls
from provider.genai import (
    convert_messages_to_genai_format,
    stream_genai_response,
    stream_genai_response_with_tools,
)

logger = logging.getLogger(__name__)

chat_bp = Blueprint('chat', __name__)


@chat_bp.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    config = current_app.config["APP_CONFIG"]
    request_id = f"req_{uuid.uuid4().hex[:16]}"
    start_time = time.monotonic()

    try:
        req_data = request.get_json()

        if not req_data or 'messages' not in req_data:
            return openai_error("Missing 'messages' field in request body")

        messages = req_data.get('messages', [])
        model = req_data.get('model', 'gpt-3.5-turbo')
        stream = req_data.get('stream', False)
        max_tokens = req_data.get('max_tokens', 30000)
        tools = req_data.get('tools', None)
        tool_choice = req_data.get('tool_choice', None)

        has_tools = tools and len(tools) > 0

        logger.info("[%s] model=%s stream=%s tools=%s messages=%d",
                     request_id, model, stream, bool(has_tools), len(messages))

        if has_tools:
            messages = inject_tool_prompt(messages, tools, tool_choice)

        chat_info = convert_messages_to_genai_format(messages)

        if not chat_info:
            return openai_error("No user message found in 'messages'")

        if stream:
            if has_tools:
                gen = stream_genai_response_with_tools(
                    chat_info, messages, model, max_tokens, config
                )
            else:
                gen = stream_genai_response(
                    chat_info, messages, model, max_tokens, config
                )
            return Response(
                stream_with_context(gen),
                mimetype='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'Connection': 'keep-alive',
                    'Content-Type': 'text/event-stream',
                }
            )

        else:
            complete_content = ""
            for line in stream_genai_response(chat_info, messages, model, max_tokens, config):
                if line.startswith('data: '):
                    data_str = line[6:].strip()
                    if data_str == '[DONE]':
                        continue
                    try:
                        data = json.loads(data_str)
                        if 'choices' in data and data['choices']:
                            delta = data['choices'][0].get('delta', {})
                            content = delta.get('content', '')
                            if content:
                                complete_content += content
                    except json.JSONDecodeError:
                        pass

            completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

            if has_tools:
                tool_calls, remaining_text = extract_tool_calls(complete_content)
            else:
                tool_calls, remaining_text = None, complete_content

            if tool_calls:
                message_obj = {
                    "role": "assistant",
                    "content": remaining_text,
                    "tool_calls": tool_calls
                }
                finish_reason = "tool_calls"
            else:
                message_obj = {
                    "role": "assistant",
                    "content": complete_content
                }
                finish_reason = "stop"

            response = {
                "id": completion_id,
                "object": "chat.completion",
                "created": int(datetime.now().timestamp()),
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": message_obj,
                    "finish_reason": finish_reason
                }],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": len(complete_content),
                    "total_tokens": len(complete_content)
                }
            }
            return jsonify(response)

    except TokenExpiredError as e:
        return openai_error(
            f"Upstream token expired: {e}",
            error_type="authentication_error",
            code="token_expired",
            status=401,
        )
    except Exception as e:
        logger.exception("[%s] Unhandled error", request_id)
        return openai_error(
            str(e),
            error_type="server_error",
            code="internal_error",
            status=500
        )
    finally:
        elapsed = time.monotonic() - start_time
        logger.info("[%s] completed in %.2fs", request_id, elapsed)
