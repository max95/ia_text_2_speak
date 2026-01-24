from __future__ import annotations

import time
from typing import List, Dict, Any, Optional, Tuple

import requests
from openai import OpenAI


class LlamaCppClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8080", timeout_s: float = 300.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 300,
        model: Optional[str] = None,
    ) -> Tuple[str, float]:
        t0 = time.time()
        payload: Dict[str, Any] = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if model:
            payload["model"] = model

        r = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            timeout=self.timeout_s,
        )
        r.raise_for_status()
        data = r.json()
        text = data["choices"][0]["message"]["content"].strip()
        dt = time.time() - t0
        return text, dt

    def chat_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int = 300,
        model: Optional[str] = None,
    ) -> Tuple[str, List[Dict[str, Any]], float]:
        text, dt = self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
        )
        return text, [], dt


class OpenAIChatClient:
    """
    Client OpenAI compatible avec l'interface de LlamaCppClient.

    - Entrée: messages (role/content), temperature, max_tokens, model optionnel
    - Sortie: (text, dt)
    """

    def __init__(self, default_model: str = "gpt-4.1-mini", timeout_s: float = 300.0) -> None:
        # La clé est lue depuis OPENAI_API_KEY (env), éventuellement alimentée par un .env
        self.client = OpenAI()
        self.default_model = default_model
        self.timeout_s = timeout_s

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 300,
        model: Optional[str] = None,
    ) -> Tuple[str, float]:
        t0 = time.time()

        resp = self.client.chat.completions.create(
            model=model or self.default_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=self.timeout_s,  # si votre version openai le supporte
        )

        text = (resp.choices[0].message.content or "").strip()
        dt = time.time() - t0
        return text, dt

    def chat_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int = 300,
        model: Optional[str] = None,
    ) -> Tuple[str, List[Dict[str, Any]], float]:
        t0 = time.time()

        resp = self.client.chat.completions.create(
            model=model or self.default_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice="auto",
            timeout=self.timeout_s,
        )

        message = resp.choices[0].message
        text = (message.content or "").strip()
        tool_calls: List[Dict[str, Any]] = []
        if message.tool_calls:
            for call in message.tool_calls:
                tool_calls.append(
                    {
                        "id": call.id,
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    }
                )
        dt = time.time() - t0
        return text, tool_calls, dt


# Exemple: chargez votre .env au point d’entrée (main/orchestrator), pas dans les classes.
def load_env() -> None:
    """
    Charge .env si python-dotenv est installé.
    Utilisation recommandée: appeler load_env() au tout début de votre programme.
    """
    try:
        from dotenv import load_dotenv  # type: ignore
    except ImportError:
        return
    load_dotenv()
