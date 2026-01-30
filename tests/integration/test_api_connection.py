import pytest
import requests
import time

BASE_URL = "http://localhost:8000"

def test_backend_health():
    """Test backend health endpoint"""
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_upload_endpoint_exists():
    """Test upload endpoint is accessible"""
    response = requests.options(f"{BASE_URL}/api/upload")
    assert response.status_code in [200, 405]

def test_query_endpoint_exists():
    """Test query endpoint is accessible"""
    response = requests.options(f"{BASE_URL}/api/query")
    assert response.status_code in [200, 405]

def test_cors_headers():
    """Test CORS headers are set"""
    response = requests.options(f"{BASE_URL}/api/upload")
    assert "access-control-allow-origin" in response.headers

if __name__ == "__main__":
    pytest.main([__file__, "-v"])