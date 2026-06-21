import re

import requests

DEFAULT_LATITUDE = 9.948375
DEFAULT_LONGITUDE = -83.699654
DEFAULT_UNITS = "celsius"

WMO_DESCRIPTIONS = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    71: "Slight snow",
    80: "Rain showers",
    95: "Thunderstorm",
}


def _degrees_to_compass(degrees: float) -> str:
    directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    return directions[round(degrees / 45) % 8]


def _parse_coords(text: str) -> tuple[float, float] | None:
    match = re.match(r"^\s*(-?\d+(?:\.\d+)?)\s*,?\s*(-?\d+(?:\.\d+)?)\s*$", text)
    if not match:
        return None
    return float(match.group(1)), float(match.group(2))


def _geocode_by_name(location: str) -> tuple[float, float, str]:
    geo = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": location, "count": 1, "language": "en", "format": "json"},
        timeout=10,
    )
    geo.raise_for_status()
    results = geo.json().get("results")
    if not results:
        raise ValueError(f"Location not found: {location}")

    place = results[0]
    return place["latitude"], place["longitude"], place.get("name", location)


def _format_coordinates(latitude: float, longitude: float) -> str:
    return f"{latitude},{longitude}"


def _resolve_coordinates(
    location: str | None,
    latitude: float | None,
    longitude: float | None,
) -> tuple[float, float, str]:
    if location and location.strip():
        coords = _parse_coords(location)
        if coords:
            lat, lon = coords
            return lat, lon, _format_coordinates(lat, lon)
        lat, lon, name = _geocode_by_name(location.strip())
        return lat, lon, name

    if latitude is not None and longitude is not None:
        return latitude, longitude, _format_coordinates(latitude, longitude)

    return (
        DEFAULT_LATITUDE,
        DEFAULT_LONGITUDE,
        _format_coordinates(DEFAULT_LATITUDE, DEFAULT_LONGITUDE),
    )


def _fetch_forecast(lat: float, lon: float, units: str) -> dict:
    temp_unit = "fahrenheit" if units == "fahrenheit" else "celsius"
    wind_unit = "mph" if units == "fahrenheit" else "kmh"

    weather = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat,
            "longitude": lon,
            "current": (
                "temperature_2m,weather_code,wind_speed_10m,"
                "wind_direction_10m,wind_gusts_10m"
            ),
            "temperature_unit": temp_unit,
            "wind_speed_unit": wind_unit,
            "timezone": "UTC",
        },
        timeout=10,
    )
    weather.raise_for_status()
    return weather.json()["current"]


def fetch_live_weather(
    units: str = DEFAULT_UNITS,
    location: str | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
) -> dict:
    lat, lon, resolved_name = _resolve_coordinates(location, latitude, longitude)
    current = _fetch_forecast(lat, lon, units)

    code = current["weather_code"]
    wind_dir = _degrees_to_compass(current["wind_direction_10m"])
    gust = current.get("wind_gusts_10m") or 0.0
    timestamp = current["time"]
    if not timestamp.endswith("Z"):
        timestamp = f"{timestamp}Z"

    return {
        "location": resolved_name,
        "units": units,
        "temperature": current["temperature_2m"],
        "description": WMO_DESCRIPTIONS.get(code, f"Weather code {code}"),
        "timestamp": timestamp,
        "weather_code": code,
        "wind_speed": current["wind_speed_10m"],
        "wind_direction": wind_dir,
        "wind_gust": gust,
        "wind_gust_direction": wind_dir,
        "wind_gust_speed": gust,
    }
