"""Bounded local generation with final-answer traces and model fingerprints."""

from __future__ import annotations

import time
from typing import Any

import httpx


class OllamaClient:
    """Also supplies the httpx-shaped post method used by the production extractor."""

    def __init__(
        self,
        base_url: str,
        *,
        seed: int = 42,
        num_ctx: int = 8192,
        num_predict: int = 2048,
        timeout: float = 300,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.options = {
            "temperature": 0,
            "seed": seed,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
        }
        self.client = httpx.Client(timeout=timeout)
        self.calls: list[dict[str, Any]] = []

    def close(self) -> None:
        self.client.close()

    def fingerprint(self, model: str) -> dict[str, Any]:
        tags = self.client.get(f"{self.base_url}/api/tags")
        tags.raise_for_status()
        installed = next((m for m in tags.json()["models"] if m["name"] == model), None)
        if installed is None:
            raise ValueError(f"model is not installed in Ollama: {model}")
        show = self.client.post(f"{self.base_url}/api/show", json={"model": model})
        show.raise_for_status()
        version = self.client.get(f"{self.base_url}/api/version")
        version.raise_for_status()
        return {
            "ollama_version": version.json().get("version"),
            "name": model,
            "digest": installed["digest"],
            "details": installed["details"],
            "capabilities": show.json().get("capabilities", []),
            "options": self.options,
        }

    def post(self, url: str, *, json: dict[str, Any]) -> httpx.Response:
        request = {**json, "options": {**self.options, **json.get("options", {})}}
        start = time.perf_counter()
        response = self.client.post(url, json=request)
        response.raise_for_status()
        data = response.json()
        # Retain the final answer and operational counts; thinking text is not a benchmark artifact.
        self.calls.append(
            {
                "model": request["model"],
                "think": request.get("think"),
                "request_format": request.get("format", "prompt_only_json"),
                "thinking_returned": bool(data.get("thinking")),
                "latency_ms": (time.perf_counter() - start) * 1000,
                **{
                    k: data.get(k)
                    for k in (
                        "response",
                        "done_reason",
                        "prompt_eval_count",
                        "eval_count",
                        "total_duration",
                        "load_duration",
                    )
                },
            }
        )
        return response

    def generate(
        self, model: str, prompt: str, *, system: str, think: bool | None = None
    ) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "system": system,
            "stream": False,
        }
        if think is not None:
            request["think"] = think
        else:
            request["format"] = "json"
        self.post(f"{self.base_url}/api/generate", json=request)
        return self.calls[-1]
