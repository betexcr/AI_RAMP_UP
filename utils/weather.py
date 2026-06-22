import re
from dataclasses import dataclass

import requests

DEFAULT_LATITUDE = 9.948375
DEFAULT_LONGITUDE = -83.699654
DEFAULT_UNITS = "celsius"
DEFAULT_TIMEZONE = "Etc/GMT+6"  # GMT-6 / UTC-6
NOMINATIM_USER_AGENT = "AI_RAMP_UP/1.0"

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

COUNTRY_CODES = {
    "costa rica": "CR",
    "colombia": "CO",
    "mexico": "MX",
    "united states": "US",
    "usa": "US",
    "canada": "CA",
    "spain": "ES",
    "france": "FR",
    "germany": "DE",
    "united kingdom": "GB",
    "uk": "GB",
    "brazil": "BR",
    "argentina": "AR",
    "chile": "CL",
    "peru": "PE",
    "panama": "PA",
    "nicaragua": "NI",
    "honduras": "HN",
    "guatemala": "GT",
    "el salvador": "SV",
}


@dataclass(frozen=True)
class GeocodedLocation:
    latitude: float
    longitude: float
    name: str
    geocoder: str


def _degrees_to_compass(degrees: float) -> str:
    directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    return directions[round(degrees / 45) % 8]


def _parse_coords(text: str) -> tuple[float, float] | None:
    match = re.match(r"^\s*(-?\d+(?:\.\d+)?)\s*,?\s*(-?\d+(?:\.\d+)?)\s*$", text)
    if not match:
        return None
    return float(match.group(1)), float(match.group(2))


def _country_to_code(country: str | None) -> str | None:
    if not country:
        return None
    return COUNTRY_CODES.get(country.lower())


def _parse_hierarchical_location(location: str) -> dict:
    """Parse comma-separated locations from country down to district.

    Supported shapes:
    - Costa Rica
    - Cartago, Costa Rica
    - Turrialba, Cartago, Costa Rica
    - Santa Cruz, Turrialba, Cartago, Costa Rica
    """
    parts = [part.strip() for part in location.split(",") if part.strip()]
    if not parts:
        return {
            "place": location.strip(),
            "admins": [],
            "country": None,
            "country_only": False,
        }

    if len(parts) == 1 and _country_to_code(parts[0]):
        return {
            "place": parts[0],
            "admins": [],
            "country": parts[0],
            "country_only": True,
        }

    country = None
    if len(parts) > 1 and _country_to_code(parts[-1]):
        country = parts[-1]
        parts = parts[:-1]

    if not parts and country:
        return {
            "place": country,
            "admins": [],
            "country": country,
            "country_only": True,
        }

    return {
        "place": parts[0],
        "admins": parts[1:],
        "country": country,
        "country_only": False,
    }


def _admin_fields(place: dict) -> str:
    return " ".join(
        filter(
            None,
            [
                place.get("admin4"),
                place.get("admin3"),
                place.get("admin2"),
                place.get("admin1"),
            ],
        )
    ).lower()


def _admin_matches(place: dict, admin_terms: list[str]) -> bool:
    if not admin_terms:
        return True
    haystack = _admin_fields(place)
    return all(term.lower() in haystack for term in admin_terms)


def _pick_best_open_meteo_result(
    results: list[dict],
    country: str | None,
    admin_terms: list[str],
) -> dict | None:
    if not results:
        return None

    country_code = _country_to_code(country)
    filtered = results
    if country_code:
        filtered = [place for place in results if place.get("country_code") == country_code]
    elif country:
        country_lower = country.lower()
        filtered = [
            place
            for place in results
            if country_lower in (place.get("country") or "").lower()
        ]

    if not filtered:
        filtered = results

    if admin_terms:
        admin_matches = [place for place in filtered if _admin_matches(place, admin_terms)]
        if admin_matches:
            return admin_matches[0]

    return filtered[0]


def _search_open_meteo(params: dict) -> list[dict]:
    response = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={
            "count": 100,
            "language": "en",
            "format": "json",
            **params,
        },
        timeout=10,
    )
    response.raise_for_status()
    return response.json().get("results") or []


def _format_open_meteo_name(
    place: dict, admin_terms: list[str], country: str | None
) -> str:
    name = place.get("name", "")
    country_label = place.get("country") or country or ""

    if admin_terms:
        return ", ".join(part for part in [name, *admin_terms, country_label] if part)

    admin1 = (place.get("admin1") or "").removesuffix(" Province")
    if admin1 and admin1.lower() not in name.lower() and not admin_terms:
        return ", ".join(part for part in [name, admin1, country_label] if part)

    return ", ".join(part for part in [name, country_label] if part)


def _geocode_country_only(country: str) -> GeocodedLocation | None:
    country_code = _country_to_code(country)
    params: dict = {"q": country, "format": "json", "limit": 1}
    if country_code:
        params["countrycodes"] = country_code.lower()

    response = requests.get(
        "https://nominatim.openstreetmap.org/search",
        params=params,
        headers={"User-Agent": NOMINATIM_USER_AGENT},
        timeout=10,
    )
    response.raise_for_status()
    results = response.json()
    if not results:
        return None

    place = results[0]
    return GeocodedLocation(
        latitude=float(place["lat"]),
        longitude=float(place["lon"]),
        name=country,
        geocoder="nominatim",
    )


def _geocode_open_meteo(location: str, parsed: dict | None = None) -> GeocodedLocation | None:
    parsed = parsed or _parse_hierarchical_location(location)
    if parsed["country_only"]:
        return None

    place = parsed["place"]
    admin_terms = parsed["admins"]
    country = parsed["country"]
    country_code = _country_to_code(country)

    search_attempts: list[dict] = []
    if country_code:
        search_attempts.append({"name": place, "countryCode": country_code})
    if admin_terms:
        search_attempts.append({"name": " ".join([place, *admin_terms])})
    if country:
        search_attempts.append({"name": f"{place} {country}"})
    search_attempts.append({"name": location.replace(",", " ")})
    search_attempts.append({"name": place})

    for params in search_attempts:
        results = _search_open_meteo(params)
        match = _pick_best_open_meteo_result(results, country, admin_terms)
        if match is None:
            continue
        if admin_terms and not _admin_matches(match, admin_terms):
            continue
        return GeocodedLocation(
            latitude=match["latitude"],
            longitude=match["longitude"],
            name=_format_open_meteo_name(match, admin_terms, country),
            geocoder="open-meteo",
        )

    return None


def _geocode_nominatim(location: str) -> GeocodedLocation | None:
    response = requests.get(
        "https://nominatim.openstreetmap.org/search",
        params={"q": location, "format": "json", "limit": 1},
        headers={"User-Agent": NOMINATIM_USER_AGENT},
        timeout=10,
    )
    response.raise_for_status()
    results = response.json()
    if not results:
        return None

    place = results[0]
    display_name = place.get("display_name", location)
    short_name = ", ".join(
        part.strip() for part in display_name.split(",")[:4] if part.strip()
    )
    return GeocodedLocation(
        latitude=float(place["lat"]),
        longitude=float(place["lon"]),
        name=short_name,
        geocoder="nominatim",
    )


def _geocode_by_name(location: str) -> GeocodedLocation:
    parsed = _parse_hierarchical_location(location)

    if parsed["country_only"]:
        country_result = _geocode_country_only(parsed["country"])
        if country_result is not None:
            return country_result

    open_meteo_result = _geocode_open_meteo(location, parsed)
    if open_meteo_result is not None:
        return open_meteo_result

    nominatim_result = _geocode_nominatim(location)
    if nominatim_result is not None:
        return nominatim_result

    raise ValueError(f"Location not found: {location}")


def _format_coordinates(latitude: float, longitude: float) -> str:
    return f"{latitude},{longitude}"


def _resolve_location(
    location: str | None,
    latitude: float | None,
    longitude: float | None,
) -> GeocodedLocation:
    if location and location.strip():
        coords = _parse_coords(location)
        if coords:
            lat, lon = coords
            return GeocodedLocation(
                latitude=lat,
                longitude=lon,
                name=_format_coordinates(lat, lon),
                geocoder="coordinates",
            )
        return _geocode_by_name(location.strip())

    if latitude is not None and longitude is not None:
        return GeocodedLocation(
            latitude=latitude,
            longitude=longitude,
            name=_format_coordinates(latitude, longitude),
            geocoder="coordinates",
        )

    return GeocodedLocation(
        latitude=DEFAULT_LATITUDE,
        longitude=DEFAULT_LONGITUDE,
        name=_format_coordinates(DEFAULT_LATITUDE, DEFAULT_LONGITUDE),
        geocoder="default",
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
            "timezone": DEFAULT_TIMEZONE,
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
    geocoded = _resolve_location(location, latitude, longitude)
    current = _fetch_forecast(geocoded.latitude, geocoded.longitude, units)

    code = current["weather_code"]
    wind_dir = _degrees_to_compass(current["wind_direction_10m"])
    gust = current.get("wind_gusts_10m") or 0.0
    timestamp = current["time"]
    if not timestamp.endswith(("Z", "+", "-")):
        timestamp = f"{timestamp}-06:00"

    return {
        "location": geocoded.name,
        "latitude": geocoded.latitude,
        "longitude": geocoded.longitude,
        "geocoder": geocoded.geocoder,
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
