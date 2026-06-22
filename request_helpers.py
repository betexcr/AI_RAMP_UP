from flask import jsonify, request
from pydantic import BaseModel, Field


class TextRequest(BaseModel):
    text: str = Field(min_length=1)


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1)


class SearchRequest(BaseModel):
    search_query: str = Field(min_length=1)


def success(data):
    return jsonify({"data": data})


def get_json_or_query_text(param_name: str = "text", default: str | None = None) -> str:
    if request.is_json and request.json:
        value = request.json.get(param_name) or request.json.get("question") or request.json.get("search_query")
        if value:
            return str(value).strip()
    value = request.args.get(param_name, default or "").strip()
    if not value and param_name == "text":
        value = request.args.get("question", "").strip()
    return value


def parse_body(model: type[BaseModel]) -> BaseModel:
    payload = request.get_json(silent=True) or {}
    return model.model_validate(payload)
