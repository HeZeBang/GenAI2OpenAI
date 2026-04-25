import argparse
import logging
import os

from config import Config
from app import create_app

parser = argparse.ArgumentParser(description='GenAI Flask API Server')
parser.add_argument('--token', type=str,
                    default='eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJleHAiOjE3NjMzNjA2MzgsInVzZXJuYW1lIjoiMjAyNDEzNDAyMiJ9.b4E5VzUxkn0Kc1pxkKVipybRFCw47NcppBognTD39e8',
                    help='GenAI API Access Token')
parser.add_argument('--port', type=int, default=5000,
                    help='Flask server port (default: 5000)')
parser.add_argument('--debug', action='store_true',
                    help='Enable debug logging')
parser.add_argument('--api-key', type=str, default=None,
                    help='API key for client authentication (or set API_KEY env var)')
args = parser.parse_args()

config = Config(
    token=args.token,
    port=args.port,
    api_key=args.api_key or os.environ.get("API_KEY"),
    debug=args.debug,
)

app = create_app(config)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    logger.info("Starting GenAI2OpenAI proxy on port %d", config.port)
    logger.info("Debug: %s, Auth: %s", config.debug, "enabled" if config.api_key else "disabled")
    app.run(host='0.0.0.0', port=config.port, debug=False)
