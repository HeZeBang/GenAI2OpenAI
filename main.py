from flask import Flask, request, jsonify, stream_with_context, Response
from flask_cors import CORS
import requests
import json
import uuid
from datetime import datetime
import re
import argparse
import logging
import time
import os

app = Flask(__name__)
CORS(app)

# 解析命令行参数
parser = argparse.ArgumentParser(description='GenAI Flask API Server')
parser.add_argument('--token', type=str, default='eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJleHAiOjE3NjMzNjA2MzgsInVzZXJuYW1lIjoiMjAyNDEzNDAyMiJ9.b4E5VzUxkn0Kc1pxkKVipybRFCw47NcppBognTD39e8',
                    help='GenAI API Access Token')
parser.add_argument('--port', type=int, default=5000,
                    help='Flask server port (default: 5000)')
parser.add_argument('--debug', action='store_true',
                    help='Enable debug logging')
parser.add_argument('--api-key', type=str, default=None,
                    help='API key for client authentication (or set API_KEY env var)')
args = parser.parse_args()

# ============================================================
# Logging 配置
# ============================================================
logging.basicConfig(
    level=logging.DEBUG if args.debug else logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# API Key：命令行参数优先，其次环境变量
API_KEY = args.api_key or os.environ.get("API_KEY")
if API_KEY:
    logger.info("API key authentication enabled")
else:
    logger.info("No API key set — running in open mode (no auth)")


# ============================================================
# OpenAI 兼容错误响应
# ============================================================
def openai_error(message, error_type="invalid_request_error", code=None, status=400):
    """返回 OpenAI 格式的错误响应"""
    return jsonify({
        "error": {
            "message": message,
            "type": error_type,
            "code": code
        }
    }), status


def make_error_chunk(message, model="unknown", completion_id=None):
    """生成流式错误 chunk（带 finish_reason: 'error'），用于 SSE"""
    cid = completion_id or f"chatcmpl-{uuid.uuid4().hex[:24]}"
    error_chunk = {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": int(datetime.now().timestamp()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {"content": f"[Error] {message}"},
            "finish_reason": "error"
        }]
    }
    return f"data: {json.dumps(error_chunk)}\n\ndata: [DONE]\n\n"


# ============================================================
# API Key 认证中间件
# ============================================================
@app.before_request
def check_api_key():
    """校验 Bearer token（仅在设置了 API_KEY 时生效）"""
    if not API_KEY:
        return  # 开发模式，跳过认证

    # 健康检查不需要认证
    if request.path == '/health':
        return

    # 只保护 /v1/ 路径
    if not request.path.startswith('/v1/'):
        return

    auth_header = request.headers.get('Authorization', '')
    if not auth_header.startswith('Bearer '):
        return openai_error(
            "Missing Authorization header with Bearer token",
            error_type="invalid_request_error",
            code="invalid_api_key",
            status=401
        )

    token = auth_header[7:]  # len('Bearer ') == 7
    if token != API_KEY:
        return openai_error(
            "Incorrect API key provided",
            error_type="invalid_request_error",
            code="invalid_api_key",
            status=401
        )

# GenAI API 配置
GENAI_URL = "https://genai.shanghaitech.edu.cn/htk/chat/start/chat"
GENAI_HEADERS = {
    "Accept": "*/*, text/event-stream",
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "Content-Type": "application/json",
    "Origin": "https://genai.shanghaitech.edu.cn",
    "Referer": "https://genai.shanghaitech.edu.cn/dialogue",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36",
    "X-Access-Token": args.token,
    "sec-ch-ua": '"Chromium";v="142", "Google Chrome";v="142", "Not_A Brand";v="99"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
}


# ============================================================
# Tool Calling 支持
# ============================================================

TOOL_SYSTEM_PROMPT = """\
You have access to the following tools:

<tools>
{tool_definitions}
</tools>

When you need to call a tool, you MUST use the following XML format. Do NOT use markdown code blocks.

<tool_call>
{{"name": "<function-name>", "arguments": {{<arguments-as-json>}}}}
</tool_call>

Rules:
1. You can call multiple tools by using multiple <tool_call> blocks.
2. If you don't need any tool, just respond normally in plain text without any <tool_call> tags.
3. After receiving tool results, analyze them and either call more tools or give a final answer in plain text.
4. The "arguments" field MUST be a valid JSON object matching the tool's parameter schema.
5. NEVER wrap <tool_call> in markdown code blocks like ```xml or ```json."""

TOOL_CHOICE_REQUIRED_PROMPT = "\nYou MUST call at least one tool in your response. Do NOT respond with plain text only."
TOOL_CHOICE_SPECIFIC_PROMPT = '\nYou MUST call the tool named "{name}" in your response.'


def format_tool_definitions(tools):
    """把 OpenAI tools 格式转为 prompt 中的 XML 描述"""
    definitions = []
    for tool in tools:
        if tool.get("type") != "function":
            continue
        func = tool["function"]
        params = func.get("parameters", {})
        params_json = json.dumps(params, ensure_ascii=False, indent=2)
        definitions.append(
            f'<tool_definition>\n'
            f'  <name>{func["name"]}</name>\n'
            f'  <description>{func.get("description", "")}</description>\n'
            f'  <parameters>\n{params_json}\n  </parameters>\n'
            f'</tool_definition>'
        )
    return "\n".join(definitions)


def inject_tool_prompt(messages, tools, tool_choice=None):
    """将 tool 定义注入到 messages 的 system prompt 中，并处理 tool/assistant 历史消息"""
    tool_defs = format_tool_definitions(tools)
    tool_prompt = TOOL_SYSTEM_PROMPT.format(tool_definitions=tool_defs)

    # 处理 tool_choice
    if tool_choice == "required":
        tool_prompt += TOOL_CHOICE_REQUIRED_PROMPT
    elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        name = tool_choice["function"]["name"]
        tool_prompt += TOOL_CHOICE_SPECIFIC_PROMPT.format(name=name)

    # 构建新的 messages 列表
    new_messages = []
    has_system = False

    for msg in messages:
        role = msg.get("role")

        if role == "system":
            # 追加 tool prompt 到已有 system 消息
            new_messages.append({
                "role": "system",
                "content": msg.get("content", "") + "\n\n" + tool_prompt
            })
            has_system = True

        elif role == "tool":
            # 把 tool result 转为 user 消息，模型才能理解
            tool_call_id = msg.get("tool_call_id", "unknown")
            new_messages.append({
                "role": "user",
                "content": (
                    f'<tool_result>\n'
                    f'  <tool_call_id>{tool_call_id}</tool_call_id>\n'
                    f'  <result>\n{msg.get("content", "")}\n  </result>\n'
                    f'</tool_result>'
                )
            })

        elif role == "assistant" and msg.get("tool_calls"):
            # 把 assistant 的 tool_calls 还原为文本
            tc_text = msg.get("content") or ""
            for tc in msg["tool_calls"]:
                func = tc.get("function", {})
                call_obj = {
                    "name": func.get("name", ""),
                    "arguments": json.loads(func.get("arguments", "{}"))
                }
                tc_text += f'\n<tool_call>\n{json.dumps(call_obj, ensure_ascii=False)}\n</tool_call>'
            new_messages.append({
                "role": "assistant",
                "content": tc_text.strip()
            })

        else:
            new_messages.append(msg)

    # 如果没有 system 消息，插入一条
    if not has_system:
        new_messages.insert(0, {
            "role": "system",
            "content": tool_prompt
        })

    return new_messages


def strip_think_blocks(content):
    """剥离 <think>...</think> 推理块（DeepSeek-R1 等模型会输出）"""
    return re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()


def _parse_tool_call_body(raw):
    """解析 <tool_call>...</tool_call> 内部的内容。

    支持两种格式：
    1. JSON: {"name": "func", "arguments": {...}}
    2. XML:  <name>func</name><arguments>{"key": "val"}</arguments>
       或    <name>func</name><arguments>\n{"key": "val"}\n</arguments>
    返回 {"name": ..., "arguments": ...} 或 None。
    """
    raw = raw.strip()

    # 尝试 JSON 格式
    try:
        call = json.loads(raw)
        if "name" in call:
            return call
    except (json.JSONDecodeError, ValueError):
        pass

    # 尝试 XML 格式: <name>...</name> ... <arguments>...</arguments>
    name_m = re.search(r'<name>\s*(.*?)\s*</name>', raw, re.DOTALL)
    args_m = re.search(r'<arguments>\s*(.*?)\s*</arguments>', raw, re.DOTALL)
    if name_m:
        name = name_m.group(1).strip()
        arguments = {}
        if args_m:
            args_str = args_m.group(1).strip()
            try:
                arguments = json.loads(args_str)
            except (json.JSONDecodeError, ValueError):
                # arguments 不是合法 JSON，当作字符串
                arguments = {"raw": args_str}
        return {"name": name, "arguments": arguments}

    return None


def extract_tool_calls(content):
    """从模型文本中提取 <tool_call>...</tool_call> 块，返回 (tool_calls, remaining_text)"""
    # 先剥离 <think> 块，避免误匹配
    cleaned = strip_think_blocks(content)

    # 剥离包裹 <tool_call> 的 markdown 代码块（模型可能不听话）
    # 匹配 ```xml、```json、``` 等包裹
    cleaned = re.sub(
        r'```(?:xml|json|plaintext|text)?\s*\n?\s*(<tool_call>.*?</tool_call>)\s*\n?\s*```',
        r'\1',
        cleaned,
        flags=re.DOTALL
    )

    pattern = r'<tool_call>\s*(.*?)\s*</tool_call>'
    matches = re.findall(pattern, cleaned, re.DOTALL)

    if not matches:
        logger.debug("No <tool_call> tags found in content (%d chars): %s",
                      len(content), content[:500])
        return None, content

    logger.debug("Found %d <tool_call> match(es)", len(matches))

    tool_calls = []
    for i, match in enumerate(matches):
        call = _parse_tool_call_body(match)
        if call:
            tool_calls.append({
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {
                    "name": call["name"],
                    "arguments": json.dumps(
                        call.get("arguments", {}),
                        ensure_ascii=False
                    )
                }
            })
        else:
            logger.warning("Failed to parse tool_call[%d] — raw: %s", i, match[:300])
            continue

    if not tool_calls:
        return None, content

    # 移除 tool_call 和 think 块后的剩余文本
    remaining = re.sub(r'<tool_call>.*?</tool_call>', '', cleaned, flags=re.DOTALL).strip()
    return tool_calls, remaining or None


def _tag_prefix_len(text, tag):
    """返回 text 末尾与 tag 前缀的最长匹配长度。

    用于流式检测中识别可能被截断的标签前缀，避免将
    "<tool_call>" 的前几个字符（如 "<to"）误输出为普通文本。
    例如 _tag_prefix_len("hello <to", "<tool_call>") → 3
    """
    max_len = min(len(tag) - 1, len(text))
    for length in range(max_len, 0, -1):
        if text[-length:] == tag[:length]:
            return length
    return 0


# ============================================================
# 原有函数（略有修改）
# ============================================================

def convert_messages_to_genai_format(messages):
    """将OpenAI格式的消息转换为GenAI格式"""
    # 提取最后一条用户消息作为 chatInfo
    chat_info = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            chat_info = msg.get("content", "")
            break

    return chat_info

def extract_content_from_genai(response_data):
    """从GenAI API响应中提取 (content, reasoning_content)。

    GenAI 上游对 DeepSeek 模型的 chunk 格式：
    - V3: {"content": "你好", "reasoning_content": null}
    - R1 思考阶段: {"reasoning_content": "嗯...", "content": ""}
    - R1 回复阶段: {"content": "你好", "reasoning_content": null}

    返回 (content, reasoning_content) 元组，均可能为 None。
    """
    try:
        if "choices" in response_data and len(response_data["choices"]) > 0:
            delta = response_data["choices"][0].get("delta", {})
            content = delta.get("content") or None
            reasoning = delta.get("reasoning_content") or None
            return content, reasoning
    except (KeyError, IndexError, TypeError):
        pass
    return None, None

def stream_genai_response(chat_info, messages, model, max_tokens):
    """流式调用GenAI API并转换为OpenAI格式"""

    # 确定 rootAiType
    azure_models = {"GPT-5", "o4-mini", "GPT-4.1", "o3", "GPT-4.1-mini"}
    root_ai_type = "azure" if model in azure_models else "xinference"

    # 构建GenAI请求数据
    genai_data = {
        # "chatInfo": chat_info,
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

        # 调用GenAI API
        response = requests.post(
            GENAI_URL,
            headers=GENAI_HEADERS,
            json=genai_data,
            stream=True,
            timeout=60
        )

        logger.debug("GenAI Response Status: %d", response.status_code)

        if response.status_code != 200:
            logger.warning("GenAI API error %d: %s", response.status_code, response.text[:500])
            # 映射上游错误码
            if response.status_code == 401:
                yield make_error_chunk("Upstream authentication failed", model)
            elif response.status_code == 429:
                yield make_error_chunk("Upstream rate limit exceeded", model)
            else:
                yield make_error_chunk(f"Upstream API error: {response.status_code}", model)
            return

        # 处理流式响应
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

                    # 处理SSE格式
                    if line_str.startswith('data:'):
                        line_str = line_str[5:].strip()

                    if line_str:
                        genai_json = json.loads(line_str)

                        # 检查上游是否返回业务级错误（HTTP 200 但 success=false）
                        if isinstance(genai_json, dict) and genai_json.get("success") is False:
                            err_msg = genai_json.get("message", "Unknown upstream error")
                            err_code = genai_json.get("code", 500)
                            logger.warning("GenAI business error (code=%s): %s", err_code, err_msg)
                            yield make_error_chunk(f"Upstream error: {err_msg}", model)
                            return

                        # 检查是否已经完成
                        if "choices" in genai_json and len(genai_json["choices"]) > 0:
                            choice = genai_json["choices"][0]
                            if choice.get("finish_reason") is not None:
                                finished = True

                        if finished:
                            # 发送完成信号后跳出循环
                            final_response = {
                                "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                                "object": "chat.completion.chunk",
                                "created": int(datetime.now().timestamp()),
                                "model": model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {},
                                        "finish_reason": "stop"
                                    }
                                ]
                            }
                            yield f"data: {json.dumps(final_response)}\n\n"
                            yield "data: [DONE]\n\n"
                            break

                        content, reasoning = extract_content_from_genai(genai_json)

                        # 构建 delta：content 和 reasoning_content 分开传递
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
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": delta,
                                        "finish_reason": None
                                    }
                                ]
                            }
                            yield f"data: {json.dumps(openai_response)}\n\n"

                except json.JSONDecodeError as e:
                    logger.debug("JSON decode error: %s, line: %s", e, line_str[:200])

        logger.debug("Total lines received: %d, finished: %s", line_count, finished)

        # 发送完成信号
        final_response = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion.chunk",
            "created": int(datetime.now().timestamp()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }
            ]
        }
        yield f"data: {json.dumps(final_response)}\n\n"
        yield "data: [DONE]\n\n"

    except Exception as e:
        logger.exception("Error in stream_genai_response")
        yield make_error_chunk(str(e), model)


# ============================================================
# 流式 Tool Calling：类 Ollama 状态机 + 标签前缀检测
# ============================================================

def stream_genai_response_with_tools(chat_info, messages, model, max_tokens):
    """流式调用 GenAI，支持 tool call 检测 — 文本部分实时流式输出

    采用类 Ollama 的两阶段方案：
    1. STREAMING: 正常流式输出文本，同时检测 <tool_call> 标签
       - 标签前的文本实时输出，无额外延迟
       - 缓冲区末尾若匹配标签前缀（如 "<to"）则暂不输出
    2. BUFFERING: 检测到 <tool_call> 后缓冲剩余内容
       - 上游结束后统一解析 tool call 并输出结构化 chunk
    """
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(datetime.now().timestamp())

    OPEN_TAG = "<tool_call>"

    buffer = ""              # 待检测缓冲区（可能含标签前缀）
    tool_buffer = ""         # tool call 内容缓冲区
    sent_role = False        # 是否已发送 role: assistant
    tool_detected = False    # 是否已进入 tool call 缓冲模式

    def make_chunk(delta, finish_reason=None):
        """构造 OpenAI 格式的 SSE chunk"""
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
        """发送文本 chunk，首次自动附加 role: assistant"""
        nonlocal sent_role
        delta = {"content": text}
        if not sent_role:
            delta["role"] = "assistant"
            sent_role = True
        return make_chunk(delta)

    # ── 消费上游流式响应 ──
    for line in stream_genai_response(chat_info, messages, model, max_tokens):
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

        # ── 已进入 tool call 缓冲模式，直接追加 ──
        if tool_detected:
            tool_buffer += content
            continue

        # ── 正常流式模式：追加并检测标签 ──
        buffer += content

        # 1) 检查完整的 <tool_call> 标签
        tag_pos = buffer.find(OPEN_TAG)
        if tag_pos >= 0:
            # 标签前的文本立即流式输出
            pre = buffer[:tag_pos]
            if pre.strip():
                yield emit_text(pre)

            # 切换到缓冲模式
            tool_detected = True
            tool_buffer = buffer[tag_pos:]
            buffer = ""
            continue

        # 2) 检查末尾是否可能是标签前缀（如 "<to"）
        plen = _tag_prefix_len(buffer, OPEN_TAG)
        if plen > 0:
            # 前缀之前的部分安全输出，前缀保留等待后续确认
            safe = buffer[:-plen]
            if safe:
                yield emit_text(safe)
            buffer = buffer[-plen:]
        else:
            # 全部安全，直接输出
            if buffer:
                yield emit_text(buffer)
            buffer = ""

    # ── 上游响应结束，处理剩余内容 ──
    if tool_detected:
        # 解析所有 tool calls（含 think 块剥离、markdown 解包）
        tool_calls, remaining = extract_tool_calls(tool_buffer)

        if tool_calls:
            logger.debug("Streaming tool calling: detected %d tool_call(s)", len(tool_calls))

            # tool call 之间的剩余文本
            if remaining and remaining.strip():
                yield emit_text(remaining.strip())

            # 确保已发送 role chunk（OpenAI 协议要求首 chunk 带 role）
            if not sent_role:
                yield make_chunk({"role": "assistant"})
                sent_role = True

            # 发送 tool_calls chunks
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
            # 标签检测到了但解析失败 — 当作普通文本输出
            logger.warning("Tool tag detected but parsing failed — emitting as text")
            yield emit_text(tool_buffer)
            yield make_chunk({}, finish_reason="stop")
            yield "data: [DONE]\n\n"
    else:
        # 未检测到 tool call — 刷出缓冲区中残留的前缀文本
        if buffer:
            yield emit_text(buffer)

        if not sent_role:
            yield make_chunk({"role": "assistant", "content": ""})

        yield make_chunk({}, finish_reason="stop")
        yield "data: [DONE]\n\n"


# ============================================================
# API 路由
# ============================================================

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    """OpenAI兼容的聊天完成端点"""
    request_id = f"req_{uuid.uuid4().hex[:16]}"
    start_time = time.monotonic()

    try:
        req_data = request.get_json()

        # 验证必要字段
        if not req_data or 'messages' not in req_data:
            return openai_error("Missing 'messages' field in request body")

        messages = req_data.get('messages', [])
        model = req_data.get('model', 'gpt-3.5-turbo')
        stream = req_data.get('stream', False)
        max_tokens = req_data.get('max_tokens', 30000)
        tools = req_data.get('tools', None)
        tool_choice = req_data.get('tool_choice', None)

        # 如果有 tools，注入 tool prompt 并转换消息
        has_tools = tools and len(tools) > 0

        logger.info("[%s] model=%s stream=%s tools=%s messages=%d",
                     request_id, model, stream, bool(has_tools), len(messages))

        if has_tools:
            messages = inject_tool_prompt(messages, tools, tool_choice)

        # 转换消息格式
        chat_info = convert_messages_to_genai_format(messages)

        if not chat_info:
            return openai_error("No user message found in 'messages'")

        # 流式响应
        if stream:
            if has_tools:
                gen = stream_genai_response_with_tools(
                    chat_info, messages, model, max_tokens
                )
            else:
                gen = stream_genai_response(
                    chat_info, messages, model, max_tokens
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

        # 非流式响应（收集所有内容后返回）
        else:
            complete_content = ""
            for line in stream_genai_response(chat_info, messages, model, max_tokens):
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

            # 检测 tool calls
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
                "choices": [
                    {
                        "index": 0,
                        "message": message_obj,
                        "finish_reason": finish_reason
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": len(complete_content),
                    "total_tokens": len(complete_content)
                }
            }
            return jsonify(response)

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

@app.route('/v1/models', methods=['GET'])
def list_models():
    """列出可用模型"""
    available_models = [
        "deepseek-v3:671b",
        "deepseek-r1:671b",
        "GPT-5",
        "o4-mini",
        "GPT-4.1",
        "o3",
        "GPT-4.1-mini",
        "qwen-instruct",
        "qwen-think"
    ]

    models = []
    for model_id in available_models:
        models.append({
            "id": model_id,
            "object": "model",
            "owned_by": "genai",
            "permission": []
        })

    return jsonify({"object": "list", "data": models})

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查端点"""
    return jsonify({"status": "ok"}), 200

if __name__ == '__main__':
    logger.info("Starting GenAI2OpenAI proxy on port %d", args.port)
    logger.info("Debug: %s, Auth: %s", args.debug, "enabled" if API_KEY else "disabled")
    app.run(host='0.0.0.0', port=args.port, debug=False)
