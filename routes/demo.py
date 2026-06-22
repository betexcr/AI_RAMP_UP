from flask import Blueprint, current_app, request

from errors import APIError
from request_helpers import get_json_or_query_text, success
from schemas import CalendarEvent, MathReasoning
from services.data_store import get_data_store
from services.openai_client import get_openai_client
from services.openai_helpers import parse_structured

demo_bp = Blueprint("demo", __name__)


@demo_bp.route("/poet", methods=["GET"])
def poet():
    config = current_app.config
    prompt = request.args.get("prompt", config["DEFAULT_POET_PROMPT"])
    raw = request.args.get("raw", "").lower() in ("1", "true", "yes")

    client = get_openai_client()
    response = client.responses.create(
        model=config["GPT_MODEL"],
        instructions="Write it like a poet",
        input=prompt,
        prompt_cache_retention=config["PROMPT_CACHE_RETENTION"],
    )

    if raw:
        return success(response.model_dump())
    return success({"text": response.output_text})


@demo_bp.route("/instructions", methods=["GET", "POST"])
def instructions():
    config = current_app.config
    question = get_json_or_query_text("question", config["DEFAULT_INSTRUCTIONS_QUESTION"])
    if not question:
        raise APIError("Missing question parameter", code="missing_parameter", status_code=400)

    store = get_data_store()
    client = get_openai_client()
    response = client.responses.create(
        model=config["GPT_MODEL"],
        instructions=store.instructions_prompt,
        input=question,
        prompt_cache_retention=config["PROMPT_CACHE_RETENTION"],
    )
    return success({"result": response.output_text})


@demo_bp.route("/calendar", methods=["GET", "POST"])
def calendar():
    config = current_app.config
    text = get_json_or_query_text("text", config["DEFAULT_CALENDAR_TEXT"])
    if not text:
        raise APIError("Missing text parameter", code="missing_parameter", status_code=400)

    parsed = parse_structured(
        [
            {"role": "system", "content": "Extract the event information."},
            {"role": "user", "content": text},
        ],
        CalendarEvent,
        model=config["GPT_MODEL"],
        prompt_cache_retention=config["PROMPT_CACHE_RETENTION"],
    )
    return success(parsed.model_dump())


@demo_bp.route("/math", methods=["GET", "POST"])
def math():
    config = current_app.config
    text = get_json_or_query_text("text", config["DEFAULT_MATH_TEXT"])
    if not text:
        raise APIError("Missing text parameter", code="missing_parameter", status_code=400)

    parsed = parse_structured(
        [
            {
                "role": "system",
                "content": (
                    "You are a helpful math tutor. Guide the user through "
                    "the solution step by step."
                ),
            },
            {"role": "user", "content": text},
        ],
        MathReasoning,
        model=config["GPT_MODEL"],
        prompt_cache_retention=config["PROMPT_CACHE_RETENTION"],
    )
    return success(parsed.model_dump())
