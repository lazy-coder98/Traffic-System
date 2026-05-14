import pytest
import json
from app import app

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

def test_health_endpoint(client):
    rv = client.get("/health")
    assert rv.status_code == 200
    assert json.loads(rv.data) == {"status": "ok"}

def test_predict_invalid_input(client):
    # Test missing payload
    rv = client.post("/api/predict", json=None)
    assert rv.status_code == 400
    
    # Test invalid junction type
    rv = client.post("/api/predict", json={"junction": "invalid", "hour": 12, "day": 1})
    assert rv.status_code == 400
    
    # Test invalid hour range
    rv = client.post("/api/predict", json={"junction": 1, "hour": 25, "day": 1})
    assert rv.status_code == 400

def test_predict_valid_input(client):
    rv = client.post("/api/predict", json={"junction": 1, "hour": 12, "day": 1})
    assert rv.status_code == 200
    data = json.loads(rv.data)
    assert "predicted_vehicles" in data
    assert "congestion_level" in data
    assert "signal_time_seconds" in data
    assert "rl_recommendation" in data
    
    rl_rec = data["rl_recommendation"]
    assert "optimal_signal_seconds" in rl_rec
    assert "phase_allocation" in rl_rec

def test_stats_endpoint(client):
    rv = client.get("/api/stats")
    assert rv.status_code == 200
    data = json.loads(rv.data)
    assert "total_records" in data
    assert "model_metrics" in data

def test_management_status(client):
    rv = client.get("/api/management/status")
    assert rv.status_code == 200
    data = json.loads(rv.data)
    assert "junctions" in data
    assert len(data["junctions"]) == 4
    assert "training_metrics" in data
