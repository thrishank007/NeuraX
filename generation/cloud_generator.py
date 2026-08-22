"""
Cloud LLM generator using OpenAI-compatible API.
Same interface as LMStudioGenerator — returns GeneratedResponse.
"""
import json
import time
from typing import Dict, Iterator, List, Optional

import requests
from loguru import logger

from generation.lmstudio_generator import GeneratedResponse


class CloudGenerator:
    """Generates responses via a remote OpenAI-compatible API endpoint."""

    def __init__(self, config: Dict):
        self.api_url = config.get("api_url", "")
        self.api_key = config.get("api_key", "")
        self.model = config.get("model", "")
        self.max_tokens = config.get("max_tokens", 1024)
        self.temperature = config.get("temperature", 0.7)
        self.timeout = config.get("timeout", 60)

        if not self.api_url:
            raise ValueError(
                "NEURAX_CLOUD_API_URL is required for cloud mode. "
                "Set it in your .env or environment."
            )
        if not self.model:
            raise ValueError(
                "NEURAX_CLOUD_MODEL is required for cloud mode. "
                "Set it in your .env or environment."
            )

        # Normalize base URL
        self.api_url = self.api_url.rstrip("/")
        if not self.api_url.endswith("/v1"):
            # ponytail: assume user gives base URL, append /v1 if missing
            if "/v1" not in self.api_url:
                self.api_url += "/v1"

        logger.info(f"Cloud generator ready: {self.api_url} model={self.model}")

    def generate_grounded_response(
        self,
        query: str,
        context: List[Dict],
        max_length: Optional[int] = None,
    ) -> GeneratedResponse:
        start = time.time()

        context_text = "\n\n".join(
            f"[Source: {doc.get('file_path', 'unknown')}]\n{doc.get('content', '')}"
            for doc in context
            if doc.get("content")
        )

        if context_text:
            system_prompt = (
                "You are a precise, grounded assistant. Answer ONLY from the provided context. "
                "If the context doesn't contain enough information, say so clearly. "
                "Cite sources by referencing their file paths."
            )
            user_prompt = (
                f"Context:\n{context_text}\n\n"
                f"Question: {query}\n\n"
                "Provide a thorough, grounded answer based only on the context above."
            )
            confidence_score = 0.7
            grounding_score = 0.7
            citations_needed = list(range(len(context)))
        else:
            system_prompt = (
                "You are NeuraX, a helpful, polite AI assistant. "
                "Answer the user's query clearly and conversationally. "
                "If the user asks a specific question about documents or facts not provided, clearly state that no relevant documents were found in the index."
            )
            user_prompt = query
            confidence_score = 0.0
            grounding_score = 0.0
            citations_needed = []

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_length or self.max_tokens,
            "temperature": self.temperature,
        }

        try:
            resp = requests.post(
                f"{self.api_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()

            text = data["choices"][0]["message"]["content"]
            model_used = data.get("model", self.model)

            return GeneratedResponse(
                response_text=text,
                confidence_score=confidence_score,
                processing_time=time.time() - start,
                context_used=context,
                grounding_score=grounding_score,
                citations_needed=citations_needed,
                model_used=model_used,
            )

        except requests.RequestException as exc:
            logger.error(f"Cloud API request failed: {exc}")
            return GeneratedResponse(
                response_text=f"Cloud API error: {exc}",
                confidence_score=0.0,
                processing_time=time.time() - start,
                context_used=context,
                grounding_score=0.0,
                citations_needed=[],
                model_used=self.model,
            )

    def generate_grounded_response_stream(
        self,
        query: str,
        context: List[Dict],
        max_length: Optional[int] = None,
    ) -> Iterator[str]:
        """Token-streaming variant of generate_grounded_response.

        Yields response tokens as they arrive (SSE deltas); when exhausted,
        the generator's return value is the same GeneratedResponse the
        non-streaming call would produce, so citations/confidence behave
        identically.
        """
        start = time.time()

        context_text = "\n\n".join(
            f"[Source: {doc.get('file_path', 'unknown')}]\n{doc.get('content', '')}"
            for doc in context
            if doc.get("content")
        )
        if context_text:
            system_prompt = (
                "You are a precise, grounded assistant. Answer ONLY from the provided context. "
                "If the context doesn't contain enough information, say so clearly. "
                "Cite sources by referencing their file paths."
            )
            user_prompt = (
                f"Context:\n{context_text}\n\n"
                f"Question: {query}\n\n"
                "Provide a thorough, grounded answer based only on the context above."
            )
            confidence_score = 0.7
            citations_needed = list(range(len(context)))
        else:
            system_prompt = (
                "You are NeuraX, a helpful, polite AI assistant. "
                "Answer the user's query clearly and conversationally. "
                "If the user asks a specific question about documents or facts not provided, clearly state that no relevant documents were found in the index."
            )
            user_prompt = query
            confidence_score = 0.0
            citations_needed = []

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_length or self.max_tokens,
            "temperature": self.temperature,
            "stream": True,
        }

        parts: list[str] = []
        model_used = self.model
        resp = requests.post(
            f"{self.api_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.timeout,
            stream=True,
        )
        resp.raise_for_status()
        try:
            for raw_line in resp.iter_lines(decode_unicode=True):
                if not raw_line or not raw_line.startswith("data:"):
                    continue
                data_str = raw_line[len("data:"):].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                if chunk.get("model"):
                    model_used = chunk["model"]
                choices = chunk.get("choices") or []
                delta = (choices[0].get("delta") or {}).get("content") if choices else None
                if delta:
                    parts.append(delta)
                    yield delta
        finally:
            resp.close()

        return GeneratedResponse(
            response_text="".join(parts),
            confidence_score=confidence_score,
            processing_time=time.time() - start,
            context_used=context,
            grounding_score=confidence_score,
            citations_needed=citations_needed,
            model_used=model_used,
        )

    def generate_summary(self, documents: List[Dict], max_length: int = 300) -> str:
        result = self.generate_grounded_response(
            query="Summarize the following documents concisely.",
            context=documents,
            max_length=max_length,
        )
        return result.response_text

    def supports_multimodal(self) -> bool:
        return False  # ponytail: text-only for now; upgrade: send vision payload if model supports it

    def get_model_info(self) -> Dict:
        return {
            "model_loaded": True,
            "current_model": self.model,
            "mode": "cloud",
            "api_url": self.api_url,
            "supports_multimodal": False,
        }
