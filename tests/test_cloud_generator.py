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


def _sse_response(lines: list[str]) -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()
    mock_resp.iter_lines.return_value = iter(lines)
    mock_resp.close = MagicMock()
    return mock_resp


def test_cloud_generator_stream_yields_deltas_and_returns_response():
    gen = CloudGenerator(_mock_config())
    lines = [
        'data: {"model": "test-model", "choices": [{"delta": {"content": "Hel"}}]}',
        'data: {"choices": [{"delta": {"content": "lo"}}]}',
        "data: [DONE]",
    ]
    with patch("generation.cloud_generator.requests.post",
               return_value=_sse_response(lines)) as post:
        stream = gen.generate_grounded_response_stream(
            query="What is Python?",
            context=[{"content": "Python is a language.", "file_path": "test.txt"}],
        )
        deltas = []
        while True:
            try:
                deltas.append(next(stream))
            except StopIteration as stop:
                final = stop.value
                break

    assert deltas == ["Hel", "lo"]
    assert final.response_text == "Hello"
    assert final.model_used == "test-model"
    assert final.context_used[0]["file_path"] == "test.txt"
    # streaming must be requested over HTTP
    assert post.call_args.kwargs["json"]["stream"] is True


def test_cloud_generator_stream_skips_malformed_lines():
    gen = CloudGenerator(_mock_config())
    lines = [
        "",
        "event: ping",
        "data: not-json",
        'data: {"choices": [{"delta": {"content": "ok"}}]}',
        "data: [DONE]",
    ]
    with patch("generation.cloud_generator.requests.post",
               return_value=_sse_response(lines)):
        stream = gen.generate_grounded_response_stream(
            query="q", context=[{"content": "c", "file_path": "f"}]
        )
        deltas = []
        while True:
            try:
                deltas.append(next(stream))
            except StopIteration as stop:
                final = stop.value
                break

    assert deltas == ["ok"]
    assert final.response_text == "ok"
