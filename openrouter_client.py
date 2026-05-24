import time
from typing import List, Dict, Any, Optional

import requests


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterClient:
    """
    Thin wrapper around OpenRouter's OpenAI-compatible chat API.

    - Tracks latency and usage (prompt/completion tokens).
    - Surfaces response cache headers so users can see HIT/MISS behaviour.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        enable_response_cache: bool = True,
        app_title: str = "Gemma Agent Playground",
        app_url: str = "https://share.streamlit.io/",
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.enable_response_cache = enable_response_cache
        # Optional attribution headers so the app becomes visible on OpenRouter.[web:12]
        self._base_headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": app_url,
            "X-OpenRouter-Title": app_title,
        }
        if self.enable_response_cache:
            # Enable edge response caching for identical requests.[web:3][web:15]
            self._base_headers["X-OpenRouter-Cache"] = "true"

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Send a chat completion request and return:
        {
          "content": str,
          "usage": {...},
          "cache": {...},
          "raw": full_json
        }
        """
        headers = dict(self._base_headers)
        if extra_headers:
            headers.update(extra_headers)

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        t_start = time.perf_counter()
        resp = requests.post(OPENROUTER_BASE_URL, headers=headers, json=payload, timeout=60)
        t_end = time.perf_counter()

        latency_s = t_end - t_start

        if resp.status_code != 200:
            # Bubble up something useful to the UI.
            return {
                "content": f"Error from OpenRouter ({resp.status_code}): {resp.text}",
                "usage": {},
                "cache": {},
                "latency_s": latency_s,
                "raw": {},
                "ok": False,
            }

        data = resp.json()
        content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )

        usage = data.get("usage", {}) or {}
        # Some models expose cached prompt tokens via nested fields.[file:1][web:6]
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        cached_tokens = 0
        details = usage.get("prompt_tokens_details") or usage.get("prompt_tokens_details".replace("_", "")) or {}
        if isinstance(details, dict):
            cached_tokens = details.get("cached_tokens", 0)

        cache_status = resp.headers.get("X-OpenRouter-Cache-Status", "MISS")
        cache_age = resp.headers.get("X-OpenRouter-Cache-Age", "0")
        cache_ttl = resp.headers.get("X-OpenRouter-Cache-TTL", "0")

        cache_info = {
            "status": cache_status,
            "age": cache_age,
            "ttl": cache_ttl,
            # Interpret HIT as “KV/edge cache hit” from the user’s point of view.
            "kv_cache_hit": cache_status.upper() == "HIT",
        }

        usage_info = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cached_prompt_tokens": cached_tokens,
            "total_tokens": usage.get("total_tokens", prompt_tokens + completion_tokens),
            "latency_s": latency_s,
        }

        return {
            "content": content,
            "usage": usage_info,
            "cache": cache_info,
            "raw": data,
            "ok": True,
        }