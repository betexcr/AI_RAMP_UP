import json

from errors import LocationNotFoundError, ParseFailedError
from schemas import Weather
from services.openai_client import get_openai_client
from services.openai_helpers import parse_structured
from tools.tools import tools
from utils.weather import (
    DEFAULT_LATITUDE,
    DEFAULT_LONGITUDE,
    DEFAULT_UNITS,
    fetch_live_weather,
)

VALID_UNITS = frozenset({"celsius", "fahrenheit"})


def validate_units(units: str) -> str:
    if units not in VALID_UNITS:
        from errors import APIError

        raise APIError(
            f"units must be one of: {', '.join(sorted(VALID_UNITS))}",
            code="invalid_units",
            status_code=400,
        )
    return units


def build_weather_user_message(
    location: str | None,
    latitude: float | None,
    longitude: float | None,
    units: str,
) -> str:
    if location:
        return f"What is the weather in {location}? Use {units}."
    if latitude is not None and longitude is not None:
        return (
            f"What is the weather at coordinates {latitude}, {longitude}? "
            f"Use {units}."
        )
    return (
        f"What is the weather at coordinates {DEFAULT_LATITUDE}, "
        f"{DEFAULT_LONGITUDE}? Use {units}."
    )


def _merge_weather_args(tool_args: dict, request_defaults: dict) -> dict:
    merged = {**request_defaults, **tool_args}
    for key in ("location", "latitude", "longitude"):
        if merged.get(key) in (None, ""):
            merged[key] = request_defaults.get(key)
    if not merged.get("units"):
        merged["units"] = request_defaults["units"]
    return merged


def _make_weather_handler(request_defaults: dict):
    def get_weather(**kwargs):
        try:
            return fetch_live_weather(**_merge_weather_args(kwargs, request_defaults))
        except ValueError as exc:
            raise LocationNotFoundError(str(exc).removeprefix("Location not found: ")) from exc

    return get_weather


def get_weather_forecast(
    config,
    *,
    location: str | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
    units: str = DEFAULT_UNITS,
) -> Weather:
    units = validate_units(units)
    request_defaults = {
        "location": location,
        "latitude": latitude,
        "longitude": longitude,
        "units": units,
    }
    tool_handlers = {"get_weather": _make_weather_handler(request_defaults)}

    input_messages = [
        {"role": "system", "content": "Extract the weather information."},
        {
            "role": "user",
            "content": build_weather_user_message(location, latitude, longitude, units),
        },
    ]

    client = get_openai_client()
    response = client.responses.parse(
        model=config["GPT_MODEL"],
        tools=tools,
        tool_choice={"type": "function", "name": "get_weather"},
        input=input_messages,
        prompt_cache_retention=config["PROMPT_CACHE_RETENTION"],
        text_format=Weather,
    )

    for _ in range(config["MAX_TOOL_LOOP_ITERATIONS"]):
        function_calls = [
            item
            for item in response.output
            if item.type == "function_call" and item.name in tool_handlers
        ]
        if not function_calls:
            break

        tool_outputs = []
        for item in function_calls:
            args = json.loads(item.arguments)
            result = tool_handlers[item.name](**args)
            tool_outputs.append(
                {
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": json.dumps(result),
                }
            )

        input_messages = input_messages + list(response.output) + tool_outputs
        response = client.responses.parse(
            model=config["GPT_MODEL"],
            tools=tools,
            input=input_messages,
            prompt_cache_retention=config["PROMPT_CACHE_RETENTION"],
            text_format=Weather,
        )
    else:
        raise ParseFailedError("Tool loop exceeded maximum iterations")

    if response.output_parsed is None:
        raise ParseFailedError("Failed to parse weather data")

    return response.output_parsed
