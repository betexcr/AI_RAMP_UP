import logging

from flask import jsonify
from openai import APIConnectionError, APITimeoutError, BadRequestError, OpenAIError
from pydantic import ValidationError as PydanticValidationError
from werkzeug.exceptions import HTTPException

logger = logging.getLogger(__name__)


class APIError(Exception):
    def __init__(self, message: str, code: str, status_code: int):
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code


class LocationNotFoundError(APIError):
    def __init__(self, location: str):
        super().__init__(
            f"Location not found: {location}",
            code="location_not_found",
            status_code=404,
        )


class KnowledgeBaseNotLoadedError(APIError):
    def __init__(self, name: str):
        super().__init__(
            f"{name} knowledge base is not initialized.",
            code="knowledge_base_unavailable",
            status_code=503,
        )


class ParseFailedError(APIError):
    def __init__(self, detail: str = "Failed to parse structured response"):
        super().__init__(detail, code="parse_failed", status_code=502)


def error_response(code: str, message: str, status_code: int):
    return jsonify({"error": {"code": code, "message": message}}), status_code


def register_error_handlers(app):
    @app.errorhandler(APIError)
    def handle_api_error(exc: APIError):
        return error_response(exc.code, exc.message, exc.status_code)

    @app.errorhandler(PydanticValidationError)
    def handle_validation_error(exc: PydanticValidationError):
        return error_response(
            "validation_error",
            exc.errors()[0]["msg"] if exc.errors() else "Invalid request",
            400,
        )

    @app.errorhandler(BadRequestError)
    def handle_openai_bad_request(exc: BadRequestError):
        logger.warning("OpenAI bad request: %s", exc.message)
        return error_response("openai_bad_request", exc.message, 400)

    @app.errorhandler(APITimeoutError)
    def handle_openai_timeout(exc: APITimeoutError):
        logger.warning("OpenAI timeout: %s", exc)
        return error_response("openai_timeout", "OpenAI request timed out", 504)

    @app.errorhandler(APIConnectionError)
    def handle_openai_connection(exc: APIConnectionError):
        logger.warning("OpenAI connection error: %s", exc)
        return error_response("openai_connection_error", "Could not reach OpenAI", 502)

    @app.errorhandler(OpenAIError)
    def handle_openai_error(exc: OpenAIError):
        logger.exception("OpenAI error")
        return error_response("openai_error", str(exc), 502)

    @app.errorhandler(HTTPException)
    def handle_http_exception(exc: HTTPException):
        return error_response(
            exc.name.lower().replace(" ", "_"),
            exc.description or exc.name,
            exc.code or 500,
        )

    @app.errorhandler(Exception)
    def handle_unexpected_error(exc: Exception):
        logger.exception("Unhandled error")
        return error_response("internal_error", "An unexpected error occurred", 500)
