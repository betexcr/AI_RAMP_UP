def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "ok"
    assert "reviews_loaded" in payload
    assert "ww2_loaded" in payload


def test_docs(client):
    response = client.get("/docs")
    assert response.status_code == 200
    payload = response.get_json()
    assert "endpoints" in payload
    assert len(payload["endpoints"]) >= 7


def test_index(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.get_json()["docs"] == "/docs"
