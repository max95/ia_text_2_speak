from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional

import requests


@dataclass(frozen=True)
class ToolEndpoint:
    name: str
    description: str
    url: str
    method: str = "POST"
    timeout_s: float = 20.0


class ToolRegistry:
    def __init__(self, endpoints: Iterable[ToolEndpoint]) -> None:
        self._endpoints: Dict[str, ToolEndpoint] = {
            endpoint.name: endpoint for endpoint in endpoints
        }

    def tool_specs(self) -> List[Dict[str, Any]]:
        tools: List[Dict[str, Any]] = []
        for endpoint in self._endpoints.values():
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": endpoint.name,
                        "description": endpoint.description,
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "payload": {
                                    "type": "object",
                                    "description": "Données JSON à envoyer au service.",
                                }
                            },
                            "required": [],
                        },
                    },
                }
            )
        return tools

    def execute(self, tool_name: str, arguments: Mapping[str, Any]) -> Dict[str, Any]:
        endpoint = self._endpoints.get(tool_name)
        if not endpoint:
            return {
                "ok": False,
                "error": f"tool_not_found: {tool_name}",
            }

        payload = arguments.get("payload") if isinstance(arguments, Mapping) else None
        try:
            response = requests.request(
                endpoint.method.upper(),
                endpoint.url,
                json=payload,
                timeout=endpoint.timeout_s,
            )
        except requests.RequestException as exc:
            return {
                "ok": False,
                "error": f"request_failed: {exc}",
            }

        content_type = response.headers.get("content-type", "")
        data: Optional[Any] = None
        if "application/json" in content_type.lower():
            try:
                data = response.json()
            except json.JSONDecodeError:
                data = None

        return {
            "ok": response.ok,
            "status_code": response.status_code,
            "data": data,
            "text": response.text if data is None else None,
        }
