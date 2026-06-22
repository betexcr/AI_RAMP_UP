import logging
import os

from dotenv import load_dotenv
from flask import Flask

from config import config_by_name
from errors import register_error_handlers
from routes.demo import demo_bp
from routes.health import health_bp
from routes.search import search_bp
from routes.weather import weather_bp
from services.data_store import init_data_store
from services.openai_client import init_openai_client

logger = logging.getLogger(__name__)


def create_app(config_name: str | None = None) -> Flask:
    load_dotenv()

    config_name = config_name or os.environ.get("FLASK_CONFIG", "default")
    config_class = config_by_name[config_name]

    app = Flask(__name__)
    app.config.from_object(config_class)

    if not app.config["OPENAI_API_KEY"] and not app.config.get("TESTING"):
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to your .env file.")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    init_openai_client(app.config["OPENAI_API_KEY"])
    init_data_store(
        app.config["REVIEWS_PARQUET"],
        app.config["WW2_PARQUET"],
        app.config["INSTRUCTIONS_PATH"],
        require_reviews=not app.config.get("TESTING"),
    )

    register_error_handlers(app)

    app.register_blueprint(health_bp)
    app.register_blueprint(demo_bp)
    app.register_blueprint(weather_bp)
    app.register_blueprint(search_bp)

    logger.info("Application initialized with config '%s'", config_name)
    return app
