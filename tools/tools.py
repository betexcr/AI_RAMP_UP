# Define a list of callable tools for the model
tools = [
    {
        "type": "function",
        "name": "get_horoscope",
        "description": "Get today's horoscope for an astrological sign.",
        "parameters": {
            "type": "object",
            "properties": {
                "sign": {
                    "type": "string",
                    "description": "An astrological sign like Taurus or Aquarius",
                },
            },
            "required": ["sign"],
        },
    },
    {
        "type": "function",
        "name": "get_weather",
        "description": (
            "Retrieves current weather. Use location for a place name, "
            "or latitude/longitude for coordinates. "
            "If neither is provided, default coordinates are used."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": ["string", "null"],
                    "description": (
                        "Comma-separated place hierarchy, smallest to largest, "
                        "ending with country. Examples: 'Costa Rica'; "
                        "'Cartago, Costa Rica'; 'Turrialba, Cartago, Costa Rica'; "
                        "'Santa Cruz, Turrialba, Cartago, Costa Rica'. "
                        "Use null when querying by coordinates."
                    ),
                },
                "latitude": {
                    "type": ["number", "null"],
                    "description": (
                        "Latitude when querying by coordinates. "
                        "Use null when querying by location name."
                    ),
                },
                "longitude": {
                    "type": ["number", "null"],
                    "description": (
                        "Longitude when querying by coordinates. "
                        "Use null when querying by location name."
                    ),
                },
                "units": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Units the temperature will be returned in.",
                },
            },
            "required": ["location", "latitude", "longitude", "units"],
            "additionalProperties": False,
        },        "strict": True,
    },
]
