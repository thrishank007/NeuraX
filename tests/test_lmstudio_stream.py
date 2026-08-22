"""Token-streaming tests for LMStudioGenerator (HTTP mocked — no server)."""
from unittest.mock import MagicMock, patch

import requests

from generation.lmstudio_generator import GeneratedResponse, LMStudioGenerator


def _make_generator() -> LMStudioGenerator:
    gen = LMStudioGenerator.__new__(LMStudioGenerator)
    gen.base_url = "http://localhost:1234/v1"
    gen.timeout = 30
    gen.gemma_model = "google/gemma-3n"
    gen.qwen_model = "qwen/qwen3-4b"
    gen.current_model = "qwen/qwen3-4b"
    gen.max_tokens = 512
    gen.temperature = 0.7
    gen.top_p = 0.9
    gen.max_context_length = 4096
    return gen


def _models_response() -> MagicMock:
    mock = MagicMock()
    mock.status_code = 200
    mock.json.return_value = {"data": [{"id": "qwen/qwen3-4b"}]}
    return mock


def _sse_response(lines: list[str]) -> MagicMock:
    mock = MagicMock()
    mock.status_code = 200
    mock.iter_lines.return_value = iter(lines)
    mock.close = MagicMock()
    return mock


def test_stream_yields_deltas_and_returns_response():
    gen = _make_generator()
    lines = [
        'data: {"choices": [{"delta": {"content": "An"}}]}',
        'data: {"choices": [{"delta": {"content": "swer"}}]}',
        "data: [DONE]",
    ]
    with patch("generation.lmstudio_generator.requests.get",
               return_value=_models_response()), \
         patch("generation.lmstudio_generator.requests.post",
               return_value=_sse_response(lines)) as post:
        stream = gen.generate_grounded_response_stream(
            query="q",
            context=[{"content": "some context", "file_path": "f.txt"}],
        )
        deltas = []
        while True:
            try:
                deltas.append(next(stream))
            except StopIteration as stop:
                final = stop.value
                break

    assert deltas == ["An", "swer"]
    assert isinstance(final, GeneratedResponse)
    assert final.response_text == "Answer"
    assert final.model_used == "qwen/qwen3-4b"
    assert post.call_args.kwargs["json"]["stream"] is True


def test_stream_connection_error_yields_guidance_delta():
    gen = _make_generator()
    with patch("generation.lmstudio_generator.requests.get",
               return_value=_models_response()), \
         patch("generation.lmstudio_generator.requests.post",
               side_effect=requests.exceptions.ConnectionError):
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

    assert len(deltas) == 1
    assert "unreachable" in deltas[0]
    assert "unreachable" in final.response_text


def test_stream_skips_malformed_sse_lines():
    gen = _make_generator()
    lines = [
        "",
        "event: ping",
        "data: not-json",
        'data: {"choices": [{"delta": {}}]}',
        'data: {"choices": [{"delta": {"content": "ok"}}]}',
        "data: [DONE]",
    ]
    with patch("generation.lmstudio_generator.requests.get",
               return_value=_models_response()), \
         patch("generation.lmstudio_generator.requests.post",
               return_value=_sse_response(lines)):
        stream = gen.generate_grounded_response_stream(query="q", context=[])
        deltas = []
        while True:
            try:
                deltas.append(next(stream))
            except StopIteration as stop:
                final = stop.value
                break

    assert deltas == ["ok"]
    assert final.response_text == "ok"
    assert final.confidence_score == 0.0  # no context -> ungrounded
