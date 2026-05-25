import time
from typing import List, Dict, Any, Optional

import requests


GOOGLE_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


class GoogleGemmaClient:
    """
    Minimal client for Gemma 4 via the Google AI Studio / Gemini API.

    Uses the REST generateContent endpoint:
    POST https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key=API_KEY[web:104][web:95]

    The interface matches OpenRouterClient.chat(...) so the rest of the app can stay unchanged.
    """

    def __init__(self, api_key: str, model: str) -> None:
        self.api_key = api_key
        self.model = model

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
        # We map: user -> "user", assistant -> "model", system -> "user" (prepended).[web:104][web:100]
        contents = []
        for m in messages:
            role = m.get("role", "user")
            if role == "assistant":
                g_role = "model"
            else:
                # Treat both system & user as "user" turns, with system first.
                g_role = "user"
            contents.append({
                "role": g_role,
                "parts": [{"text": m.get("content", "")}],
            })

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
        resp = requests.post(url, params=params, headers=headers, json=payload, timeout=60)
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

        # Usage metadata (if available)[web:94][web:104]
        usage_meta = data.get("usageMetadata") or {}
        prompt_tokens = usage_meta.get("promptTokenCount", 0)
        completion_tokens = usage_meta.get("candidatesTokenCount", 0)
        total_tokens = prompt_tokens + completion_tokens

        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cached_prompt_tokens": 0,
            "total_tokens": total_tokens,
            "latency_s": latency_s,
        }

        # No explicit cache headers here; keep empty dict for compatibility.
        cache_info: Dict[str, Any] = {}

        return {
            "content": text,
            "usage": usage,
            "cache": cache_info,
            "raw": data,
            "latency_s": latency_s,
            "ok": True,
        }
