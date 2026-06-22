from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from schemas import Weather


def test_weather_success(client):
    mock_weather = Weather(
        location="San José",
        latitude=9.93,
        longitude=-84.08,
        geocoder="open-meteo",
        units="celsius",
        temperature=20.0,
        description="Clear",
        timestamp="2026-06-20T12:00-06:00",
        weather_code=0,
        wind_speed=10.0,
        wind_direction="N",
        wind_gust=12.0,
        wind_gust_direction="N",
        wind_gust_speed=12.0,
    )

    mock_response = MagicMock()
    mock_response.output = []
    mock_response.output_parsed = mock_weather

    with patch("services.weather_service.get_openai_client") as mock_get_client:
        mock_get_client.return_value.responses.parse.return_value = mock_response
        response = client.get("/weather?location=San Jose,Costa Rica")

    assert response.status_code == 200
    data = response.get_json()["data"]
    assert data["location"] == "San José"
    assert data["units"] == "celsius"


def test_weather_invalid_units(client):
    response = client.get("/weather?units=kelvin")
    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "invalid_units"


def test_weather_partial_coordinates(client):
    response = client.get("/weather?lat=9.9")
    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "invalid_coordinates"


def test_weather_location_not_found(client):
    function_call = SimpleNamespace(
        type="function_call",
        name="get_weather",
        arguments='{"location":"Nowhere","latitude":null,"longitude":null,"units":"celsius"}',
        call_id="call_1",
    )
    mock_response = SimpleNamespace(
        output=[function_call],
        output_parsed=None,
    )

    with patch("services.weather_service.get_openai_client") as mock_get_client:
        mock_get_client.return_value.responses.parse.return_value = mock_response
        with patch(
            "services.weather_service.fetch_live_weather",
            side_effect=ValueError("Location not found: Nowhere"),
        ):
            response = client.get("/weather?location=Nowhere")

    assert response.status_code == 404
    assert response.get_json()["error"]["code"] == "location_not_found"
