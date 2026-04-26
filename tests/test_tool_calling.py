"""
Tool Calling 测试脚本
用法: uv run test_tool_calling.py [--base-url http://localhost:5000] [--model GPT-4.1]

测试场景:
  1. 单次 tool call (天气查询)
  2. 多次 tool call (并行查天气)
  3. 多轮对话 (tool call -> tool result -> 最终回答)
  4. 无 tool 调用 (普通问题不应触发 tool)
  5. 流式 tool call
"""

import argparse
import json
import requests
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--base-url', default='http://localhost:5000')
parser.add_argument('--model', default='GPT-4.1')
args, _unknown_args = parser.parse_known_args()

BASE_URL = args.base_url
MODEL = args.model

# 定义测试用的 tools
WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a given location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name, e.g. 'Shanghai' or 'New York'"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit"
                }
            },
            "required": ["location"]
        }
    }
}

CALCULATOR_TOOL = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "Evaluate a mathematical expression and return the result.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A math expression, e.g. '2 + 3 * 4'"
                }
            },
            "required": ["expression"]
        }
    }
}

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for information about a topic.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                }
            },
            "required": ["query"]
        }
    }
}


def print_separator(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def test_list_models():
    """测试 0: 获取模型列表"""
    print_separator("Test 0: List Models")

    resp = requests.get(f"{BASE_URL}/v1/models")
    data = resp.json()
    print(json.dumps(data, indent=2, ensure_ascii=False))

    if data.get("object") != "list":
        print("\n[FAIL] Response object is not 'list'")
        return False

    models = data.get("data", [])
    if not models:
        print("\n[FAIL] No models returned")
        return False

    model_ids = [m["id"] for m in models]
    print(f"\n[PASS] {len(models)} models available: {model_ids}")
    return True


def test_single_tool_call():
    """测试 1: 单次 tool call"""
    print_separator("Test 1: Single Tool Call (Weather)")

    resp = requests.post(f"{BASE_URL}/v1/chat/completions", json={
        "model": MODEL,
        "messages": [
            {"role": "user", "content": "What's the weather in Shanghai?"}
        ],
        "tools": [WEATHER_TOOL],
        "stream": False
    })

    data = resp.json()
    print(json.dumps(data, indent=2, ensure_ascii=False))

    choice = data.get("choices", [{}])[0]
    msg = choice.get("message", {})
    if msg.get("tool_calls"):
        print("\n[PASS] Tool calls detected:")
        for tc in msg["tool_calls"]:
            print(f"  - {tc['function']['name']}({tc['function']['arguments']})")
        return True
    else:
        print("\n[FAIL] No tool calls in response")
        print(f"  Content: {msg.get('content', '')[:200]}")
        return False


def test_multiple_tools():
    """测试 2: 提供多个 tools，看模型是否选对"""
    print_separator("Test 2: Multiple Tools Available")

    resp = requests.post(f"{BASE_URL}/v1/chat/completions", json={
        "model": MODEL,
        "messages": [
            {"role": "user", "content": "What is 123 * 456 + 789?"}
        ],
        "tools": [WEATHER_TOOL, CALCULATOR_TOOL, SEARCH_TOOL],
        "stream": False
    })

    data = resp.json()
    print(json.dumps(data, indent=2, ensure_ascii=False))

    choice = data.get("choices", [{}])[0]
    msg = choice.get("message", {})
    if msg.get("tool_calls"):
        names = [tc['function']['name'] for tc in msg["tool_calls"]]
        print(f"\n[INFO] Tools called: {names}")
        if "calculate" in names:
            print("[PASS] Correctly chose 'calculate' tool")
            return True
        else:
            print("[WARN] Did not choose 'calculate' - chose different tool")
            return False
    else:
        print("\n[FAIL] No tool calls in response")
        return False


def test_multi_turn():
    """测试 3: 多轮 tool calling (call -> result -> final answer)"""
    print_separator("Test 3: Multi-turn Tool Calling")

    # 第一轮：用户提问
    print("--- Round 1: User asks about weather ---")
    resp1 = requests.post(f"{BASE_URL}/v1/chat/completions", json={
        "model": MODEL,
        "messages": [
            {"role": "user", "content": "What's the weather like in Beijing right now?"}
        ],
        "tools": [WEATHER_TOOL],
        "stream": False
    })

    data1 = resp1.json()
    choice1 = data1.get("choices", [{}])[0]
    msg1 = choice1.get("message", {})

    if not msg1.get("tool_calls"):
        print("[FAIL] Round 1 did not produce tool calls")
        print(json.dumps(data1, indent=2, ensure_ascii=False))
        return False

    tc = msg1["tool_calls"][0]
    print(f"  Tool called: {tc['function']['name']}({tc['function']['arguments']})")

    # 第二轮：把 tool 结果传回去
    print("\n--- Round 2: Sending tool result back ---")
    resp2 = requests.post(f"{BASE_URL}/v1/chat/completions", json={
        "model": MODEL,
        "messages": [
            {"role": "user", "content": "What's the weather like in Beijing right now?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": msg1["tool_calls"]
            },
            {
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": json.dumps({
                    "location": "Beijing",
                    "temperature": 22,
                    "unit": "celsius",
                    "condition": "Partly cloudy",
                    "humidity": 45
                })
            }
        ],
        "tools": [WEATHER_TOOL],
        "stream": False
    })

    data2 = resp2.json()
    print(json.dumps(data2, indent=2, ensure_ascii=False))

    choice2 = data2.get("choices", [{}])[0]
    msg2 = choice2.get("message", {})
    if msg2.get("content") and not msg2.get("tool_calls"):
        print(f"\n[PASS] Final answer: {msg2['content'][:200]}")
        return True
    elif msg2.get("tool_calls"):
        print("\n[WARN] Model called another tool instead of answering")
        return False
    else:
        print("\n[FAIL] Empty response")
        return False


def test_no_tool_needed():
    """测试 4: 普通问题不应触发 tool call"""
    print_separator("Test 4: No Tool Needed")

    resp = requests.post(f"{BASE_URL}/v1/chat/completions", json={
        "model": MODEL,
        "messages": [
            {"role": "user", "content": "What is the capital of France?"}
        ],
        "tools": [WEATHER_TOOL, CALCULATOR_TOOL],
        "stream": False
    })

    data = resp.json()
    choice = data.get("choices", [{}])[0]
    msg = choice.get("message", {})

    if not msg.get("tool_calls"):
        print(f"[PASS] No tool called. Answer: {msg.get('content', '')[:200]}")
        return True
    else:
        names = [tc['function']['name'] for tc in msg["tool_calls"]]
        print(f"[WARN] Unexpected tool call: {names}")
        return False


def test_stream_tool_call():
    """测试 5: 流式 tool calling"""
    print_separator("Test 5: Streaming Tool Call")

    resp = requests.post(f"{BASE_URL}/v1/chat/completions", json={
        "model": MODEL,
        "messages": [
            {"role": "user", "content": "What's the weather in Tokyo?"}
        ],
        "tools": [WEATHER_TOOL],
        "stream": True
    }, stream=True)

    full_content = ""
    tool_calls_found = []

    for line in resp.iter_lines():
        if not line:
            continue
        line_str = line.decode('utf-8') if isinstance(line, bytes) else line
        if not line_str.startswith('data: '):
            continue
        data_str = line_str[6:].strip()
        if data_str == '[DONE]':
            print("  [DONE]")
            break

        try:
            chunk = json.loads(data_str)
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            finish = chunk.get("choices", [{}])[0].get("finish_reason")

            if delta.get("content"):
                full_content += delta["content"]
                print(f"  content: {delta['content'][:80]}")

            if delta.get("tool_calls"):
                for tc in delta["tool_calls"]:
                    tool_calls_found.append(tc)
                    print(f"  tool_call: {tc['function']['name']}({tc['function']['arguments']})")

            if finish:
                print(f"  finish_reason: {finish}")

        except json.JSONDecodeError:
            pass

    if tool_calls_found:
        print(f"\n[PASS] Stream tool calls detected: {len(tool_calls_found)}")
        return True
    elif full_content and '<tool_call>' in full_content:
        print(f"\n[WARN] Tool call in raw text but not parsed")
        return False
    else:
        print(f"\n[FAIL] No tool calls in stream")
        print(f"  Content: {full_content[:200]}")
        return False


def test_stream_no_tool_needed():
    """测试 6: 流式 tool calling — 普通问题应真正流式输出（不缓冲）"""
    print_separator("Test 6: Streaming With Tools (No Tool Needed)")

    import time
    resp = requests.post(f"{BASE_URL}/v1/chat/completions", json={
        "model": MODEL,
        "messages": [
            {"role": "user", "content": "What is the capital of France? Answer in one sentence."}
        ],
        "tools": [WEATHER_TOOL],
        "stream": True
    }, stream=True)

    chunks = []
    first_chunk_time = None
    start_time = time.monotonic()

    for line in resp.iter_lines():
        if not line:
            continue
        line_str = line.decode('utf-8') if isinstance(line, bytes) else line
        if not line_str.startswith('data: '):
            continue
        data_str = line_str[6:].strip()
        if data_str == '[DONE]':
            break

        try:
            chunk = json.loads(data_str)
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            finish = chunk.get("choices", [{}])[0].get("finish_reason")

            if delta.get("content"):
                if first_chunk_time is None:
                    first_chunk_time = time.monotonic()
                chunks.append(delta["content"])

        except json.JSONDecodeError:
            pass

    total_time = time.monotonic() - start_time
    ttfb = (first_chunk_time - start_time) if first_chunk_time else total_time
    full_content = "".join(chunks)

    print(f"  Chunks: {len(chunks)}")
    print(f"  TTFB: {ttfb:.2f}s")
    print(f"  Total: {total_time:.2f}s")
    print(f"  Content: {full_content[:200]}")

    if len(chunks) > 1:
        print(f"\n[PASS] True streaming: {len(chunks)} chunks (not buffered into 1)")
        return True
    elif len(chunks) == 1 and full_content:
        print(f"\n[WARN] Only 1 chunk — may be buffered")
        return False
    else:
        print(f"\n[FAIL] No content received")
        return False


def test_tag_prefix_detection():
    """测试 7: _tag_prefix_len 单元测试（离线，不需要服务器）"""
    print_separator("Test 7: Tag Prefix Detection (Unit Test)")

    # 可从 tools.parsing 导入，此处内联以便独立运行
    def _tag_prefix_len(text, tag):
        max_len = min(len(tag) - 1, len(text))
        for length in range(max_len, 0, -1):
            if text[-length:] == tag[:length]:
                return length
        return 0

    TAG = "<tool_call>"
    cases = [
        # (text, expected_len, description)
        ("hello world", 0, "no prefix"),
        ("hello <", 1, "just <"),
        ("hello <t", 2, "<t prefix"),
        ("hello <to", 3, "<to prefix"),
        ("hello <too", 4, "<too prefix"),
        ("hello <tool", 5, "<tool prefix"),
        ("hello <tool_", 6, "<tool_ prefix"),
        ("hello <tool_c", 7, "<tool_c prefix"),
        ("hello <tool_ca", 8, "<tool_ca prefix"),
        ("hello <tool_cal", 9, "<tool_cal prefix"),
        ("hello <tool_call", 10, "<tool_call prefix (missing >)"),
        ("hello <toast", 0, "<toast — not a prefix after <to"),
        ("hello <div>", 0, "HTML tag — not a prefix"),
        ("<", 1, "buffer is just <"),
        ("<tool_call>", 0, "full tag — should be caught by .find(), not prefix"),
        ("", 0, "empty string"),
        ("hello <ta", 0, "<ta — diverges at 3rd char"),
    ]

    all_pass = True
    for text, expected, desc in cases:
        result = _tag_prefix_len(text, TAG)
        status = "ok" if result == expected else "FAIL"
        if result != expected:
            all_pass = False
        print(f"  [{status}] {desc}: _tag_prefix_len({text!r}) = {result} (expected {expected})")

    if all_pass:
        print(f"\n[PASS] All {len(cases)} cases passed")
    else:
        print(f"\n[FAIL] Some cases failed")
    return all_pass


if __name__ == '__main__':
    print(f"Testing against: {BASE_URL}")
    print(f"Model: {MODEL}")

    results = {}
    tests = [
        ("tag_prefix_detection", test_tag_prefix_detection),
        ("list_models", test_list_models),
        ("single_tool_call", test_single_tool_call),
        ("multiple_tools", test_multiple_tools),
        ("multi_turn", test_multi_turn),
        ("no_tool_needed", test_no_tool_needed),
        ("stream_tool_call", test_stream_tool_call),
        ("stream_no_tool_needed", test_stream_no_tool_needed),
    ]

    for name, fn in tests:
        try:
            results[name] = fn()
        except Exception as e:
            print(f"\n[ERROR] {name}: {e}")
            results[name] = False

    print_separator("Summary")
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")

    total = len(results)
    passed = sum(1 for v in results.values() if v)
    print(f"\n  {passed}/{total} tests passed")

    sys.exit(0 if passed == total else 1)
