import pytest
import requests
import io

BASE_URL = "http://localhost:8000"

def test_upload_text_file():
    """Test uploading a text file"""
    files = {
        'files': ('test.txt', io.BytesIO(b"This is a test file"), 'text/plain')
    }
    response = requests.post(f"{BASE_URL}/api/upload", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert len(data["files"]) == 1

def test_upload_multiple_files():
    """Test uploading multiple files"""
    files = [
        ('files', ('test1.txt', io.BytesIO(b"Test 1"), 'text/plain')),
        ('files', ('test2.txt', io.BytesIO(b"Test 2"), 'text/plain'))
    ]
    response = requests.post(f"{BASE_URL}/api/upload", files=files)
    assert response.status_code == 200
    data = response.json()
    assert len(data["files"]) == 2

if __name__ == "__main__":
    pytest.main([__file__, "-v"])