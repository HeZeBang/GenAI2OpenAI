import argparse
import logging
import os
import sys

from auth.cas_login import LoginError
from auth.token_manager import TokenManager
from config import Config
from app import create_app

parser = argparse.ArgumentParser(description='GenAI Flask API Server')
parser.add_argument('--token', type=str, required=True,
                    help='JWT token (eyJ...) or student_id@password for auto-login')
parser.add_argument('--port', type=int, default=5000,
                    help='Flask server port (default: 5000)')
parser.add_argument('--debug', action='store_true',
                    help='Enable debug logging')
parser.add_argument('--api-key', type=str, default=None,
                    help='API key for client authentication (or set API_KEY env var)')
args = parser.parse_args()

logging.basicConfig(
    level=logging.DEBUG if args.debug else logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

try:
    token_manager = TokenManager(args.token)
    token_manager.initial_login()
except (LoginError, ValueError) as e:
    logger.error("Failed to initialize token: %s", e)
    sys.exit(1)

config = Config(
    token_manager=token_manager,
    port=args.port,
    api_key=args.api_key or os.environ.get("API_KEY"),
    debug=args.debug,
)

app = create_app(config)

if __name__ == '__main__':
    logger.info("Starting GenAI2OpenAI proxy on port %d", config.port)
    logger.info("Debug: %s, Auth: %s, Token mode: %s",
                config.debug, "enabled" if config.api_key else "disabled",
                token_manager.mode)
    app.run(host='0.0.0.0', port=config.port, debug=False)
