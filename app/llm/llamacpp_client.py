from __future__ import annotations

import time
from typing import List, Dict, Any, Optional

import requests


class LlamaCppClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8080") -> None:
        self.base_url = base_url.rstrip("/")

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 300,
        model: Optional[str] = None,
    ) -> tuple[str, float]:
        t0 = time.time()
        payload: Dict[str, Any] = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if model:
            payload["model"] = model

        r = requests.post(f"{self.base_url}/v1/chat/completions", json=payload, timeout=300)
        r.raise_for_status()
        data = r.json()
        text = data["choices"][0]["message"]["content"].strip()
        dt = time.time() - t0
        return text, dt
