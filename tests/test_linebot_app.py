import pytest

from fraud_detection.linebot_app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "ok"
    assert "model_loaded" in data


def test_callback_missing_signature(client):
    response = client.post("/callback", data="test", content_type="text/plain")
    assert response.status_code == 400
