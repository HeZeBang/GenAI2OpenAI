from flask import Flask, request, jsonify, stream_with_context, Response
from flask_cors import CORS
import requests
import json
import uuid
from datetime import datetime
import re
import argparse
from icecream import ic

app = Flask(__name__)
CORS(app)

# 解析命令行参数
parser = argparse.ArgumentParser(description='GenAI Flask API Server')
parser.add_argument('--token', type=str, default='eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJleHAiOjE3NjMzNjA2MzgsInVzZXJuYW1lIjoiMjAyNDEzNDAyMiJ9.b4E5VzUxkn0Kc1pxkKVipybRFCw47NcppBognTD39e8',
                    help='GenAI API Access Token')
parser.add_argument('--port', type=int, default=5000,
                    help='Flask server port (default: 5000)')
args = parser.parse_args()

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
    """从GenAI API响应中提取内容"""
    try:
        if "choices" in response_data and len(response_data["choices"]) > 0:
            delta = response_data["choices"][0].get("delta", {})
            if "reasoning_content" in delta:
                return delta["reasoning_content"]
            content = delta.get("content", "")
            return content
    except (KeyError, IndexError, TypeError):
        pass
    return None

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
    
    try:
        
        # 调用GenAI API
        response = requests.post(
            GENAI_URL,
            headers=GENAI_HEADERS,
            json=genai_data,
            stream=True,
            timeout=60
        )
        
        # 打印原始响应状态
        # ic(f"DEBUG: GenAI API Response Status: {response.status_code}")
        
        if response.status_code != 200:
            yield f"data: {json.dumps({'error': f'GenAI API error: {response.status_code}'})}\n\n"
            return
        
        # 处理流式响应
        finished = False
        for line in response.iter_lines():
            if finished:
                break
                
            if line:
                try:
                    line_str = line.decode('utf-8') if isinstance(line, bytes) else line
                    
                    # 处理SSE格式
                    if line_str.startswith('data:'):
                        line_str = line_str[5:].strip()
                    
                    if line_str:
                        # ic(f"DEBUG: Raw response line: {line_str}")  # 打印原始响应行
                        genai_json = json.loads(line_str)
                        
                        # 检查是否已经完成
                        if "choices" in genai_json and len(genai_json["choices"]) > 0:
                            choice = genai_json["choices"][0]
                            if choice.get("finish_reason") is not None:
                                finished = True
                        
                        if finished:
                            # 发送完成信号后跳出循环
                            final_response = {
                                "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                                "object": "text_completion.chunk",
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
                        
                        content = extract_content_from_genai(genai_json)
                        
                        if content is not None:
                            # 转换为OpenAI格式
                            openai_response = {
                                "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                                "object": "text_completion.chunk",
                                "created": int(datetime.now().timestamp()),
                                "model": model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"content": content},
                                        "finish_reason": None
                                    }
                                ]
                            }
                            yield f"data: {json.dumps(openai_response)}\n\n"
                
                except json.JSONDecodeError:
                    pass
        
        # 发送完成信号
        final_response = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "text_completion.chunk",
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
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    """OpenAI兼容的聊天完成端点"""
    try:
        req_data = request.get_json()
        
        # 验证必要字段
        if not req_data or 'messages' not in req_data:
            return jsonify({'error': 'Missing messages field'}), 400
        
        messages = req_data.get('messages', [])
        model = req_data.get('model', 'gpt-3.5-turbo')
        stream = req_data.get('stream', False)
        max_tokens = req_data.get('max_tokens', 30000)
        
        # 转换消息格式
        chat_info = convert_messages_to_genai_format(messages)
        
        if not chat_info:
            return jsonify({'error': 'No user message found'}), 400
        
        # 流式响应
        if stream:
            return Response(
                stream_with_context(stream_genai_response(
                    chat_info, 
                    messages, 
                    model, 
                    max_tokens
                )),
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
                    try:
                        data = json.loads(line[6:])
                        if 'choices' in data and data['choices']:
                            delta = data['choices'][0].get('delta', {})
                            content = delta.get('content', '')
                            if content:
                                complete_content += content
                    except json.JSONDecodeError:
                        pass
            
            response = {
                "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                "object": "text_completion",
                "created": int(datetime.now().timestamp()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": complete_content
                        },
                        "finish_reason": "stop"
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
        return jsonify({'error': str(e)}), 500

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
    # 运行Flask应用
    app.run(host='0.0.0.0', port=args.port, debug=False)
