from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_returns_ok():
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["app"] == "IntelliSum"
    assert body["device"] in {"cpu", "cuda"}


def test_unknown_route_returns_404_json():
    response = client.get("/api/does-not-exist")
    assert response.status_code == 404
    assert "detail" in response.json()
