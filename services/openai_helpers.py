from typing import TypeVar

from pydantic import BaseModel

from services.openai_client import get_openai_client

T = TypeVar("T", bound=BaseModel)


def parse_structured(
    messages: list[dict],
    text_format: type[T],
    *,
    model: str,
    prompt_cache_retention: str,
    tools: list | None = None,
    tool_choice: dict | None = None,
) -> T:
    client = get_openai_client()
    kwargs: dict = {
        "model": model,
        "input": messages,
        "prompt_cache_retention": prompt_cache_retention,
        "text_format": text_format,
    }
    if tools is not None:
        kwargs["tools"] = tools
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice

    response = client.responses.parse(**kwargs)
    if response.output_parsed is None:
        from errors import ParseFailedError

        raise ParseFailedError()
    return response.output_parsed
