from fastapi.testclient import TestClient


def test_health(client: TestClient) -> None:
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] == "xgb_n300_d5_balanced"


def test_predict_bajo_riesgo(client: TestClient, solicitud_bajo_riesgo: dict) -> None:
    response = client.post("/api/v1/predict", json=solicitud_bajo_riesgo)

    assert response.status_code == 200
    data = response.json()
    assert 0.0 <= data["probabilidad_incumplimiento"] <= 1.0
    assert data["clasificacion"] == "Riesgo Bajo"
    assert len(data["explicacion"]["variables"]) == 4


def test_predict_alto_riesgo(client: TestClient, solicitud_alto_riesgo: dict) -> None:
    response = client.post("/api/v1/predict", json=solicitud_alto_riesgo)

    assert response.status_code == 200
    data = response.json()
    assert data["clasificacion"] == "Riesgo Alto"
    # una solicitud con moras severas debe pesar en la explicacion
    variables = {v["variable"] for v in data["explicacion"]["variables"]}
    assert variables & {"moras_90", "moras_30_59", "moras_60_89", "utilizacion"}


def test_predict_rechaza_input_invalido(client: TestClient) -> None:
    response = client.post("/api/v1/predict", json={"utilizacion": "no-es-un-numero"})

    assert response.status_code == 422
