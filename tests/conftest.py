import pytest
from unittest.mock import MagicMock, patch

from schemas import Weather


@pytest.fixture
def mock_openai_client():
    client = MagicMock()
    with patch("services.openai_client.init_openai_client", return_value=client):
        with patch("services.openai_client.get_openai_client", return_value=client):
            yield client


@pytest.fixture
def app(mock_openai_client):
    from factory import create_app

    application = create_app("testing")
    yield application


@pytest.fixture
def client(app):
    return app.test_client()
