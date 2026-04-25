import logging

from flask import Flask, g
from flask_cors import CORS

from config import Config
from auth.apikey import register_auth
from api import register_routes


def create_app(config: Config) -> Flask:
    app = Flask(__name__)
    app.config["APP_CONFIG"] = config

    CORS(app)

    logging.basicConfig(
        level=logging.DEBUG if config.debug else logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    @app.before_request
    def set_token():
        g.token = app.config["APP_CONFIG"].token

    register_auth(app)
    register_routes(app)

    return app
