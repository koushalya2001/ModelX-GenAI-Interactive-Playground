import time
from typing import List, Dict, Any, Optional

import requests
from requests.exceptions import ReadTimeout  # NEW


GOOGLE_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


class GoogleGemmaClient:
    """
    Minimal client for Gemma 4 via the Google AI Studio / Gemini API.

    Uses the REST generateContent endpoint:
    POST https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key=API_KEY

    The interface matches OpenRouterClient.chat(...) so the rest of the app can stay unchanged.
    """

    def __init__(self, api_key: str, model: str, timeout_s: float = 60.0) -> None:
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        messages: list of {"role": "system"|"user"|"assistant", "content": str}

        Returns:
        {
          "content": str,
          "usage": {...},
          "cache": {},        # kept for compatibility with OpenRouterClient
          "raw": full_json,
          "latency_s": float,
          "ok": bool,
        }
        """

        # Convert to Google "contents" format.
        # We map: user -> "user", assistant -> "model", system -> "user" (prepended).
        contents = []
        for m in messages:
            role = m.get("role", "user")
            if role == "assistant":
                g_role = "model"
            else:
                # Treat both system & user as "user" turns, with system first.
                g_role = "user"
            contents.append(
                {
                    "role": g_role,
                    "parts": [{"text": m.get("content", "")}],
                }
            )

        generation_config: Dict[str, Any] = {"temperature": temperature}
        if max_tokens is not None:
            generation_config["maxOutputTokens"] = max_tokens

        payload: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": generation_config,
        }

        params = {"key": self.api_key}
        headers = {"Content-Type": "application/json"}
        if extra_headers:
            headers.update(extra_headers)

        url = f"{GOOGLE_BASE_URL}/models/{self.model}:generateContent"
        t_start = time.perf_counter()
        try:
            resp = requests.post(
                url,
                params=params,
                headers=headers,
                json=payload,
                timeout=self.timeout_s,
            )
        except ReadTimeout:
            t_end = time.perf_counter()
            latency_s = t_end - t_start
            return {
                "content": (
                    f"Error from Google AI Studio: request timed out after {self.timeout_s:.0f} seconds. "
                    "Try shortening the prompt/context or reducing tool output."
                ),
                "usage": {},
                "cache": {},
                "latency_s": latency_s,
                "raw": {},
                "ok": False,
            }

        t_end = time.perf_counter()
        latency_s = t_end - t_start

        if resp.status_code != 200:
            return {
                "content": f"Error from Google AI Studio ({resp.status_code}): {resp.text}",
                "usage": {},
                "cache": {},
                "latency_s": latency_s,
                "raw": {},
                "ok": False,
            }

        data = resp.json()

        # Extract text from first candidate
        text = ""
        candidates = data.get("candidates") or []
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            text = "".join(p.get("text", "") for p in parts)

        # Usage metadata (if available)
        # Gemini returns promptTokenCount, candidatesTokenCount, totalTokenCount,
        # and cachedContentTokenCount when context caching is used.[web:175][web:178]
        usage_meta = data.get("usageMetadata") or {}
        prompt_tokens = usage_meta.get("promptTokenCount", 0)
        completion_tokens = usage_meta.get("candidatesTokenCount", 0)
        cached_tokens = usage_meta.get("cachedContentTokenCount", 0)
        total_tokens = usage_meta.get("totalTokenCount", prompt_tokens + completion_tokens)

        # Avoid divide-by-zero; you'll compute the ratio in the scorecard.
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cached_prompt_tokens": cached_tokens,  # used for context cache token rate
            "total_tokens": total_tokens,
            "latency_s": latency_s,
        }

        # No OpenRouter-style edge cache here; keep a compatible shape.
        cache_info: Dict[str, Any] = {
            "status": "N/A",
            "age": "0",
            "ttl": "0",
            "kv_cache_hit": False,
        }

        return {
            "content": text,
            "usage": usage,
            "cache": cache_info,
            "raw": data,
            "latency_s": latency_s,
            "ok": True,
        }
