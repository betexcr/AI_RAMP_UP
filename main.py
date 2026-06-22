import json
import os
import numpy as np
from flask import Flask, jsonify, request
from dotenv import load_dotenv
from openai import OpenAI
from schemas import CalendarEvent, MathReasoning, Weather
from tools.tools import tools 
import pandas as pd 
from utils.weather import (
    DEFAULT_LATITUDE,
    DEFAULT_LONGITUDE,
    DEFAULT_UNITS,
    fetch_live_weather,
)
# from utils.serializers import CustomEncoder

load_dotenv()
GPT_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
app = Flask(__name__)
api_key = os.environ.get("OPENAI_API_KEY")
df = pd.read_parquet("./data/embedded_1k_reviews.parquet")  # <-- This defines 'df'
embeddings_matrix = np.array(df['ada_embedding'].tolist()).astype('float32')
client = OpenAI(api_key=api_key)


def _build_weather_user_message(
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
        return fetch_live_weather(**_merge_weather_args(kwargs, request_defaults))

    return get_weather
@app.route("/poet", methods=["GET"])
def poet():
    response = client.responses.create(
        model=GPT_MODEL,
        # reasoning={"effort": "low"}, NOT SUPPORTED FOR GPT-4O-MINI
        instructions="Write it like a poet",
        input="Write a one-sentence story about a robot.",
        prompt_cache_retention="24h",
    )

    # json_data = json.dumps(response.__dict__, cls=CustomEncoder)
    # return Response(json_data, mimetype="application/json", status=200)
    json_data = response.model_dump()
    return jsonify(json_data)


@app.route("/instructions", methods=["GET"])
def instructions():
    with open("instructions/prompt.txt", "r", encoding="utf-8") as f:
        instructions = f.read()

        response = client.responses.create(
            model=GPT_MODEL,
            instructions=instructions,
            input="How would I declare a variable for a last name?",
            prompt_cache_retention="24h",
        )

        clean_text = response.output_text
        return jsonify({"status": "success", "result": clean_text})


@app.route("/calendar", methods=["GET"])
def calendar():
    response = client.responses.parse(
        model=GPT_MODEL,
        input=[
            {"role": "system", "content": "Extract the event information."},
            {
                "role": "user",
                "content": "Alice and Bob are going to a science fair on Friday.",
            },
        ],
        prompt_cache_retention="24h",
        text_format=CalendarEvent,
    )
    parsed_event = response.output_parsed
    clean_json = parsed_event.model_dump()
    return jsonify(clean_json)

@app.route("/math", methods=["GET"])
def math():
    response = client.responses.parse(
        model=GPT_MODEL,
        input=[
            {
                "role": "system",
                "content": "You are a helpful math tutor. Guide the user through the solution step by step.",
            },
            {"role": "user", "content": "how can I solve 8x + 7 = -23"},
        ],
        prompt_cache_retention="24h",
        text_format=MathReasoning,
    )

    parsed_event = response.output_parsed
    clean_json = parsed_event.model_dump()
    return jsonify(clean_json)

@app.route("/weather", methods=["GET"])
def weather():
    location = request.args.get("location")
    latitude = request.args.get("lat", type=float)
    longitude = request.args.get("lon", type=float)
    units = request.args.get("units", DEFAULT_UNITS)

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
            "content": _build_weather_user_message(
                location, latitude, longitude, units
            ),
        },
    ]

    response = client.responses.parse(
        model=GPT_MODEL,
        tools=tools,
        tool_choice={"type": "function", "name": "get_weather"},
        input=input_messages,
        prompt_cache_retention="24h",
        text_format=Weather,
    )

    while True:
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
            model=GPT_MODEL,
            tools=tools,
            input=input_messages,
            text_format=Weather,
        )

    if response.output_parsed is None:
        return jsonify({"error": "Failed to parse weather data"}), 502

    return jsonify(response.output_parsed.model_dump())

@app.route("/search_reviews", methods=["GET"])
def search_reviews():
    # 1. Grab the search text from the URL parameters
    search_query = request.args.get("search_query", "").strip()
    if not search_query:
        return jsonify({"error": "Missing search_query parameter"}), 400
    
    try:
        # 2. Clean text and get the raw vector array from OpenAI
        text = search_query.replace("\n", " ")
        # Note: Access .data[0].embedding (matching your structure)
        query_vector = client.embeddings.create(input=[text], model=EMBEDDING_MODEL).data[0].embedding
        
        # 3. Convert query vector to a float32 NumPy array
        query_vector_np = np.array(query_vector).astype('float32')
        
        # 4. PERFORM NP.DOT (Calculates score for all 1,000 reviews instantly)
        # embeddings_matrix is the global matrix loaded from your Parquet file
        similarities = np.dot(embeddings_matrix, query_vector_np)
        
        # 5. Extract top 3 highest scores
        top_k = 3
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # 6. Build the response payload
        results = []
        for idx in top_indices:
            results.append({
                "score": float(similarities[idx]), # Convert numpy float to standard python float
                "summary": str(df.iloc[idx]['Summary']),
                "review_text": str(df.iloc[idx]['Text'])
            })
            
        return jsonify({
            "query": search_query,
            "results": results
        })

    except Exception as e:
        return jsonify({"error": f"Search execution failed: {str(e)}"}), 500
