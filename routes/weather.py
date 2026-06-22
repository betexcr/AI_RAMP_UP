from flask import Blueprint, current_app, request

from request_helpers import success
from services.weather_service import get_weather_forecast
from utils.weather import DEFAULT_UNITS

weather_bp = Blueprint("weather", __name__)


@weather_bp.route("/weather", methods=["GET"])
def weather():
    config = current_app.config
    location = request.args.get("location")
    latitude = request.args.get("lat", type=float)
    longitude = request.args.get("lon", type=float)
    units = request.args.get("units", DEFAULT_UNITS)

    if (latitude is None) ^ (longitude is None):
        from errors import APIError

        raise APIError(
            "Both lat and lon are required for coordinate lookup",
            code="invalid_coordinates",
            status_code=400,
        )

    forecast = get_weather_forecast(
        config,
        location=location,
        latitude=latitude,
        longitude=longitude,
        units=units,
    )
    return success(forecast.model_dump())
