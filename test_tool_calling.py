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
args = parser.parse_args()

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


if __name__ == '__main__':
    print(f"Testing against: {BASE_URL}")
    print(f"Model: {MODEL}")

    results = {}
    tests = [
        ("list_models", test_list_models),
        ("single_tool_call", test_single_tool_call),
        ("multiple_tools", test_multiple_tools),
        ("multi_turn", test_multi_turn),
        ("no_tool_needed", test_no_tool_needed),
        ("stream_tool_call", test_stream_tool_call),
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
