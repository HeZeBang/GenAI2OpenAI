from flask import Blueprint, jsonify, g

from config import model_registry, TokenExpiredError
from errors import openai_error

models_bp = Blueprint('models', __name__)


@models_bp.route('/v1/models', methods=['GET'])
def list_models():
    token = g.get("token", "")
    try:
        models_map = model_registry.get_models(token)
    except TokenExpiredError as e:
        return openai_error(
            f"Upstream token expired: {e}",
            error_type="authentication_error",
            code="token_expired",
            status=401,
        )

    models = []
    for model_id, info in models_map.items():
        models.append({
            "id": model_id,
            "object": "model",
            "owned_by": info.root_ai_type,
            "permission": []
        })
    return jsonify({"object": "list", "data": models})
