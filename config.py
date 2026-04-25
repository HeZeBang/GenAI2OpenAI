import logging
import time
from dataclasses import dataclass, field

import requests

logger = logging.getLogger(__name__)


class TokenExpiredError(Exception):
    pass


@dataclass
class Config:
    token: str
    port: int
    api_key: str | None
    debug: bool


GENAI_URL = "https://genai.shanghaitech.edu.cn/htk/chat/start/chat"
GENAI_MODELS_URL = "https://genai.shanghaitech.edu.cn/htk/ai/aiModel/list"


@dataclass
class ModelInfo:
    id: str
    name: str
    root_ai_type: str
    max_tokens: int | None
    description: str | None


@dataclass
class ModelRegistry:
    _models: dict[str, ModelInfo] = field(default_factory=dict)
    _last_fetched: float = 0
    _cache_ttl: float = 300  # 5 minutes

    def fetch(self, token: str) -> None:
        headers = build_genai_headers(token)
        params = {
            "_t": int(time.time() * 1000),
            "pageNo": 1,
            "pageSize": 999,
            "showStatusList": "2,3",
        }
        try:
            resp = requests.get(GENAI_MODELS_URL, headers=headers, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            logger.exception("Failed to fetch model list from GenAI")
            return

        if not data.get("success"):
            msg = data.get("message", "Unknown error")
            logger.warning("GenAI model list API returned failure: %s", msg)
            raise TokenExpiredError(msg)

        records = data.get("result", {}).get("records", [])
        models: dict[str, ModelInfo] = {}
        for rec in records:
            ai_type = rec.get("aiType")
            if not ai_type:
                continue
            models[ai_type] = ModelInfo(
                id=ai_type,
                name=rec.get("aiName", ai_type),
                root_ai_type=rec.get("rootAiType", "xinference"),
                max_tokens=rec.get("maxToken"),
                description=rec.get("descInfo"),
            )

        self._models = models
        self._last_fetched = time.time()
        logger.info("Fetched %d models from GenAI platform", len(models))

    def get_models(self, token: str) -> dict[str, ModelInfo]:
        if not self._models or (time.time() - self._last_fetched > self._cache_ttl):
            self.fetch(token)
        return self._models

    def get_root_ai_type(self, model: str, token: str) -> str:
        models = self.get_models(token)
        info = models.get(model)
        if info:
            return info.root_ai_type
        return "xinference"


model_registry = ModelRegistry()


def build_genai_headers(token: str) -> dict:
    return {
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
        "X-Access-Token": token,
        "sec-ch-ua": '"Chromium";v="142", "Google Chrome";v="142", "Not_A Brand";v="99"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"Windows"',
    }
