import torch
from fastapi.testclient import TestClient

from llm_inference_optimization.api import main


client = TestClient(main.app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_generate_validates_request_before_loading_model():
    response = client.post("/generate", json={"prompt": "hello", "variant": "unknown"})

    assert response.status_code == 422


def test_generate_rejects_zero_new_tokens():
    response = client.post("/generate", json={"prompt": "hello", "max_new_tokens": 0})

    assert response.status_code == 422


def test_generate_returns_service_result(monkeypatch):
    result = {
        "variant": "baseline",
        "text": "hello",
        "metrics": {},
        "baseline_metrics": None,
        "difference_vs_baseline": None,
    }
    monkeypatch.setattr(main, "generate_response", lambda request: result)

    response = client.post("/generate", json={"prompt": "hello"})

    assert response.status_code == 200
    assert response.json() == result


def test_generate_returns_bad_request_for_value_error(monkeypatch):
    def raise_value_error(request):
        raise ValueError("invalid request")

    monkeypatch.setattr(main, "generate_response", raise_value_error)

    response = client.post("/generate", json={"prompt": "hello"})

    assert response.status_code == 400
    assert response.json()["detail"] == "invalid request"


def test_generate_returns_service_unavailable_for_gpu_oom(monkeypatch):
    def raise_oom(request):
        raise torch.OutOfMemoryError("out of memory")

    monkeypatch.setattr(main, "generate_response", raise_oom)

    response = client.post("/generate", json={"prompt": "hello"})

    assert response.status_code == 503
    assert "GPU out of memory" in response.json()["detail"]


def test_generate_hides_internal_error(monkeypatch):
    def raise_error(request):
        raise RuntimeError("internal error")

    monkeypatch.setattr(main, "generate_response", raise_error)

    response = client.post("/generate", json={"prompt": "hello"})

    assert response.status_code == 500
    assert response.json()["detail"] == "Generation failed. Check the server logs for details."
