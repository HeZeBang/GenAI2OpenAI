import json
import uuid
from datetime import datetime

from flask import jsonify


def openai_error(message, error_type="invalid_request_error", code=None, status=400):
    """Return an OpenAI-formatted JSON error response."""
    return jsonify({
        "error": {
            "message": message,
            "type": error_type,
            "code": code
        }
    }), status


def make_error_chunk(message, model="unknown", completion_id=None):
    """Generate a streaming error chunk (with finish_reason: 'error') for SSE."""
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
