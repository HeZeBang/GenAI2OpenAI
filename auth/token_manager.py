import base64
import json
import logging
import threading
import time

from auth.cas_login import LoginError, login_genai

logger = logging.getLogger(__name__)


class TokenExpiredError(Exception):
    pass


class TokenManager:
    def __init__(self, token_input: str):
        self._lock = threading.Lock()
        self._jwt: str | None = None

        # JWT format: 3 dot-separated segments
        if token_input.count(".") == 2:
            self._mode = "static"
            self._jwt = token_input
            self._student_id = None
            self._password = None
            logger.info("Token mode: static JWT")
        else:
            # credential format: student_id@password (split on first @)
            parts = token_input.split("@", 1)
            if len(parts) != 2 or not parts[0] or not parts[1]:
                raise ValueError(
                    "Invalid --token format. Use a JWT token (eyJ...) or student_id@password"
                )
            self._mode = "credential"
            self._student_id = parts[0]
            self._password = parts[1]
            logger.info("Token mode: credential (student_id=%s)", self._student_id)

    @property
    def mode(self) -> str:
        return self._mode

    def _decode_exp(self, token: str) -> float | None:
        try:
            payload = token.split(".")[1]
            # Fix base64 padding
            padding = 4 - len(payload) % 4
            if padding != 4:
                payload += "=" * padding
            decoded = base64.urlsafe_b64decode(payload)
            data = json.loads(decoded)
            return data.get("exp")
        except Exception:
            return None

    def _is_expired(self, token: str, margin: int = 60) -> bool:
        exp = self._decode_exp(token)
        if exp is None:
            return False  # Can't determine, assume valid
        return time.time() >= (exp - margin)

    def get_token(self) -> str:
        with self._lock:
            if self._mode == "static":
                if self._jwt and self._is_expired(self._jwt):
                    raise TokenExpiredError(
                        "Static JWT has expired. Please provide a new token."
                    )
                return self._jwt

            # Credential mode: auto-refresh if expired or no token yet
            if self._jwt is None or self._is_expired(self._jwt):
                logger.info("Token expired or missing, refreshing via CAS login...")
                self._jwt = login_genai(self._student_id, self._password)
                logger.info("Token refreshed successfully")

            return self._jwt

    def force_refresh(self) -> str | None:
        with self._lock:
            if self._mode == "static":
                return None
            logger.info("Force refreshing token via CAS login...")
            self._jwt = login_genai(self._student_id, self._password)
            logger.info("Token force-refreshed successfully")
            return self._jwt

    def initial_login(self) -> None:
        """For credential mode, perform login at startup to fail fast."""
        if self._mode == "credential":
            self.get_token()
