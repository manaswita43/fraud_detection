def test_health_check(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_predict_endpoint(client, sample_transaction):
    response = client.post("/predict", json=sample_transaction)
    assert response.status_code == 200

    body = response.json()
    assert "prediction" in body
    assert "probability" in body

    assert body["prediction"] in [0, 1]
    assert 0.0 <= body["probability"] <= 1.0

def test_predict_invalid_payload(client):
    response = client.post("/predict", json={"bad": "payload"})
    assert response.status_code == 422  # validation error
