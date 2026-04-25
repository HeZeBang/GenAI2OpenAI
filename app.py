import logging

from flask import Flask, g
from flask_cors import CORS

from config import Config, TokenExpiredError, LoginError
from auth.apikey import register_auth
from api import register_routes
from errors import openai_error


def create_app(config: Config) -> Flask:
    app = Flask(__name__)
    app.config["APP_CONFIG"] = config

    CORS(app)

    @app.before_request
    def set_token():
        g.token = app.config["APP_CONFIG"].token_manager.get_token()

    @app.errorhandler(TokenExpiredError)
    def handle_token_expired(e):
        return openai_error(
            f"Upstream token expired: {e}",
            error_type="authentication_error",
            code="token_expired",
            status=401,
        )

    @app.errorhandler(LoginError)
    def handle_login_error(e):
        return openai_error(
            f"CAS login failed: {e}",
            error_type="authentication_error",
            code="login_failed",
            status=401,
        )

    register_auth(app)
    register_routes(app)

    return app
