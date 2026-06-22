from flask import Blueprint, current_app, request

from errors import APIError, KnowledgeBaseNotLoadedError
from request_helpers import get_json_or_query_text, parse_body, success
from schemas import QuestionRequest, SearchRequest
from services.data_store import get_data_store
from services.embedding_search import search_reviews
from services.rag import ask_ww2_history

search_bp = Blueprint("search", __name__)


def _get_search_query() -> str:
    if request.method == "POST" and request.is_json:
        return parse_body(SearchRequest).search_query
    query = get_json_or_query_text("search_query")
    if not query:
        raise APIError(
            "Missing search_query parameter",
            code="missing_parameter",
            status_code=400,
        )
    return query


def _get_question() -> str:
    if request.method == "POST" and request.is_json:
        return parse_body(QuestionRequest).question
    question = get_json_or_query_text("question")
    if not question:
        raise APIError(
            "Missing question parameter",
            code="missing_parameter",
            status_code=400,
        )
    return question


@search_bp.route("/search_reviews", methods=["GET", "POST"])
def search_reviews_route():
    config = current_app.config
    store = get_data_store()
    if store.reviews is None:
        raise KnowledgeBaseNotLoadedError("Reviews")

    query = _get_search_query()
    results = search_reviews(
        query,
        store.reviews,
        embedding_model=config["EMBEDDING_MODEL"],
        top_k=config["TOP_K"],
    )
    return success({"query": query, "results": results})


@search_bp.route("/ask_ww2_history", methods=["GET", "POST"])
def ask_ww2_history_route():
    config = current_app.config
    store = get_data_store()
    if store.ww2 is None:
        raise KnowledgeBaseNotLoadedError("WW2")

    question = _get_question()
    result = ask_ww2_history(question, store.ww2, config)
    return success(result)
