from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import requests

from mm_rcna.config import APIConfig


class NullAPIClient:
    def __init__(self, name: str = "null-client") -> None:
        self.name = name
        self.ready = False

    def text_chat(self, model: str, messages: List[Dict[str, Any]], max_completion_tokens: int = 800) -> str:
        raise RuntimeError("NullAPIClient cannot call text_chat")

    def json_chat(self, model: str, messages: List[Dict[str, Any]], max_completion_tokens: int = 800) -> Dict[str, Any]:
        raise RuntimeError("NullAPIClient cannot call json_chat")


class OpenAICompatibleClient:
    def __init__(self, cfg: APIConfig):
        self.cfg = cfg
        base_url = (cfg.base_url or "").rstrip("/")
        # 支持用户传 /v1 或 /v1/embeddings 两种格式
        if base_url.endswith("/embeddings"):
            base_url = base_url[: -len("/embeddings")]
        if base_url.endswith("/chat/completions"):
            base_url = base_url[: -len("/chat/completions")]
        self.base_url = base_url
        self.auth_token = os.getenv(cfg.auth_token_env, "")
        self.ready = bool(self.base_url and self.auth_token)

    def _post(self, route: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self.ready:
            raise RuntimeError(
                f"API client not ready. Need base_url and env var {self.cfg.auth_token_env}."
            )

        url = f"{self.base_url}/{route.lstrip('/')}"
        r = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {self.auth_token}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.cfg.timeout,
        )
        r.raise_for_status()
        return r.json()

    def text_chat(self, model: str, messages: List[Dict[str, Any]], max_completion_tokens: int = 800) -> str:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": self.cfg.temperature,
            "max_completion_tokens": max_completion_tokens,
        }
        obj = self._post("chat/completions", payload)
        return obj["choices"][0]["message"]["content"].strip()

    def json_chat(self, model: str, messages: List[Dict[str, Any]], max_completion_tokens: int = 800) -> Dict[str, Any]:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": self.cfg.temperature,
            "max_completion_tokens": max_completion_tokens,
            "response_format": {"type": "json_object"},
        }
        obj = self._post("chat/completions", payload)
        content = obj["choices"][0]["message"]["content"]
        return json.loads(content)

    def embeddings(self, model: str, input_texts: List[str]) -> Dict[str, Any]:
        payload = {
            "model": model,
            "input": input_texts,
        }
        return self._post("embeddings", payload)