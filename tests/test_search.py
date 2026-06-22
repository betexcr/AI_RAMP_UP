from unittest.mock import patch

import numpy as np


def test_search_reviews(client):
    fake_embedding = np.ones(1536, dtype="float32").tolist()
    mock_embed_response = type("R", (), {"data": [type("E", (), {"embedding": fake_embedding})()]})()

    with patch("services.embedding_search.get_openai_client") as mock_get_client:
        mock_get_client.return_value.embeddings.create.return_value = mock_embed_response
        response = client.get("/search_reviews?search_query=spicy+noodles")

    assert response.status_code == 200
    payload = response.get_json()["data"]
    assert payload["query"] == "spicy noodles"
    assert len(payload["results"]) == 3
    assert "score" in payload["results"][0]


def test_search_reviews_post(client):
    fake_embedding = np.ones(1536, dtype="float32").tolist()
    mock_embed_response = type("R", (), {"data": [type("E", (), {"embedding": fake_embedding})()]})()

    with patch("services.embedding_search.get_openai_client") as mock_get_client:
        mock_get_client.return_value.embeddings.create.return_value = mock_embed_response
        response = client.post(
            "/search_reviews",
            json={"search_query": "sweet dessert"},
        )

    assert response.status_code == 200
    assert response.get_json()["data"]["query"] == "sweet dessert"


def test_search_reviews_missing_query(client):
    response = client.get("/search_reviews")
    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "missing_parameter"
