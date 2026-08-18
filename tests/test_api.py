import pytest
from fastapi.testclient import TestClient
import numpy as np
from src.main import app

client = TestClient(app)

def create_test_data(n_entities=50, n_periods=5):
    np.random.seed(42)
    entities = [f"entity_{i}" for i in range(n_entities)]
    time_periods = list(range(n_periods))
    
    x1 = np.random.randn(n_entities, n_periods)
    x2 = np.random.randn(n_entities, n_periods)
    
    entity_effects = np.random.randn(n_entities, 1)
    
    y = 1.0 + 2.0 * x1 - 0.5 * x2 + entity_effects + np.random.randn(n_entities, n_periods) * 0.1
    
    data = {
        "entities": entities,
        "time_periods": time_periods,
        "dependent_var": "y",
        "independent_vars": ["x1", "x2"],
        "data": {
            "y": y.tolist(),
            "x1": x1.tolist(),
            "x2": x2.tolist()
        }
    }
    
    return data

def test_estimate_model_success():
    data = create_test_data()
    request = {
        "data": data,
        "params":{
            "entity_effects": True,
            "time_effects": False,
            "cov_type": "robust"
        }
    }
    
    response = client.post("/api/v1/estimate", json=request)

    print(f"status: {response.status_code}")
    print(f"body: {response.json()}")
    assert response.status_code == 200
    result = response.json()
    
    assert "coefficients" in result
    assert "metrics" in result
    assert "fitted_values" in result
    assert "residuals" in result
    assert "model_serialized_base64" in result
    
    assert len(result["coefficients"]) == 2
    
    assert result["metrics"]["r_squared"] > 0.9
    assert result["metrics"]["nobs"] == 250 

def test_estimate_model_insufficient_data():
    data = create_test_data(n_entities=2, n_periods=2)
    request = {
        "data": data,
        "params": {
            "entity_effects": True,
            "time_effects": True,
            "cov_type": "robust"
        }
    }
    
    response = client.post("/api/v1/estimate", json=request)
    
    assert response.status_code == 422
    
def test_estimate_model_invalid_data():
    data = create_test_data()
    data["data"]["y"][0][0] = None
    
    request = {
        "data": data,
        "params": {}
    }
    
    response = client.post("/api/v1/estimate", json=request)
    assert response.status_code == 422

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}