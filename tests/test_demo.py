from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from schemas import CalendarEvent


def test_calendar_with_text_param(client):
    mock_event = CalendarEvent(
        name="Science fair",
        date="Friday",
        participants=["Alice", "Bob"],
    )
    mock_response = MagicMock()
    mock_response.output_parsed = mock_event

    with patch("services.openai_helpers.get_openai_client") as mock_get_client:
        mock_get_client.return_value.responses.parse.return_value = mock_response
        response = client.get("/calendar?text=Alice+and+Bob+science+fair+Friday")

    assert response.status_code == 200
    data = response.get_json()["data"]
    assert data["name"] == "Science fair"


def test_poet_default_response(client):
    mock_response = SimpleNamespace(output_text="A robot dreamed in verse.")

    with patch("routes.demo.get_openai_client") as mock_get_client:
        mock_get_client.return_value.responses.create.return_value = mock_response
        response = client.get("/poet")

    assert response.status_code == 200
    assert response.get_json()["data"]["text"] == "A robot dreamed in verse."
