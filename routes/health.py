from flask import Blueprint, current_app, jsonify

from services.data_store import get_data_store

health_bp = Blueprint("health", __name__)

API_DOCS = {
    "endpoints": [
        {"method": "GET", "path": "/health", "description": "Health check"},
        {"method": "GET", "path": "/docs", "description": "API index"},
        {"method": "GET", "path": "/poet", "params": ["prompt", "raw"]},
        {"method": "GET,POST", "path": "/instructions", "params": ["question"]},
        {"method": "GET,POST", "path": "/calendar", "params": ["text"]},
        {"method": "GET,POST", "path": "/math", "params": ["text"]},
        {
            "method": "GET",
            "path": "/weather",
            "params": ["location", "lat", "lon", "units"],
        },
        {"method": "GET,POST", "path": "/search_reviews", "params": ["search_query"]},
        {"method": "GET,POST", "path": "/ask_ww2_history", "params": ["question"]},
    ]
}


@health_bp.route("/")
def index():
    return jsonify({"message": "AI Ramp Up API", "docs": "/docs"})


@health_bp.route("/health")
def health():
    store = get_data_store()
    return jsonify(
        {
            "status": "ok",
            "reviews_loaded": store.reviews is not None,
            "ww2_loaded": store.ww2 is not None,
        }
    )


@health_bp.route("/docs")
def docs():
    return jsonify(API_DOCS)
