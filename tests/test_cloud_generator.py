# tests/test_cloud_generator.py
import pytest
from unittest.mock import patch, MagicMock
from generation.cloud_generator import CloudGenerator
from generation.lmstudio_generator import GeneratedResponse


def _mock_config():
    return {
        "api_url": "https://api.example.com/v1",
        "api_key": "test-key",
        "model": "test-model",
        "max_tokens": 512,
        "temperature": 0.7,
        "timeout": 30,
    }


def test_cloud_generator_returns_generated_response():
    gen = CloudGenerator(_mock_config())
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": "Test answer"}}],
        "model": "test-model",
        "usage": {"total_tokens": 100},
    }
    mock_resp.raise_for_status = MagicMock()
    with patch("generation.cloud_generator.requests.post", return_value=mock_resp):
        result = gen.generate_grounded_response(
            query="What is Python?",
            context=[{"content": "Python is a language.", "file_path": "test.txt"}],
        )
    assert isinstance(result, GeneratedResponse)
    assert result.response_text == "Test answer"
    assert result.model_used == "test-model"


def test_cloud_generator_missing_config_raises():
    with pytest.raises(ValueError, match="NEURAX_CLOUD_API_URL"):
        CloudGenerator({"api_url": "", "api_key": "", "model": ""})
