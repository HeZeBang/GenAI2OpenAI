#!/bin/bash
# 快速 curl 测试 tool calling
# 用法: bash test_curl.sh

BASE_URL="${1:-http://localhost:5000}"
echo "Testing: $BASE_URL"

echo ""
echo "=========================================="
echo "  Test: 获取模型列表"
echo "=========================================="
curl -s "$BASE_URL/v1/models" | python3 -m json.tool

echo ""
echo "=========================================="
echo "  Test: Tool Call (非流式)"
echo "=========================================="
curl -s "$BASE_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "GPT-4.1",
    "stream": false,
    "messages": [
      {"role": "user", "content": "What is the weather in Shanghai?"}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "Get current weather for a location",
          "parameters": {
            "type": "object",
            "properties": {
              "location": {"type": "string", "description": "City name"},
              "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["location"]
          }
        }
      }
    ]
  }' | python3 -m json.tool

echo ""
echo "=========================================="
echo "  Test: Tool Call (流式)"
echo "=========================================="
curl -sN "$BASE_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "GPT-4.1",
    "stream": true,
    "messages": [
      {"role": "user", "content": "What is 42 * 58?"}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "calculate",
          "description": "Evaluate a math expression",
          "parameters": {
            "type": "object",
            "properties": {
              "expression": {"type": "string", "description": "Math expression"}
            },
            "required": ["expression"]
          }
        }
      }
    ]
  }'
