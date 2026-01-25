from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import os
from fastapi import FastAPI

from app.core.store import TurnStore
from app.core.worker import WorkerPool
from app.core.pipeline import VoicePipeline
from app.core.memory import SQLiteMemory
from app.stt.whisper_asr import WhisperASR
from app.llm.llm_client import LlamaCppClient
from app.llm.llm_client import OpenAIChatClient
from app.tts.piper_tts import PiperTTS
from app.tools.tool_registry import ToolEndpoint, ToolRegistry
from app.api.routes_finance import fetch_finance_price
from app.api.routes_trains import fetch_line_l_departures


@dataclass
class Deps:
    store: TurnStore
    worker: WorkerPool


deps: Deps  # rempli au startup


def create_app() -> FastAPI:
    logging.basicConfig(level=logging.INFO)
    app = FastAPI(title="ia_text_2_speak")

    store = TurnStore()

    asr = WhisperASR(model_name="small", language="fr")
    #llm = LlamaCppClient(base_url="http://127.0.0.1:8080")
    llm = OpenAIChatClient()
    tts = PiperTTS(model_path="app/tts/models/fr_FR-upmc-medium.onnx")

    tool_registry = _build_tool_registry()
    memory = SQLiteMemory()
    pipeline = VoicePipeline(asr=asr, llm=llm, tts=tts, tool_registry=tool_registry, memory=memory)

    worker = WorkerPool(store=store, pipeline=pipeline, concurrency=1)

    global deps
    deps = Deps(store=store, worker=worker)

    @app.on_event("startup")
    async def _startup():
        await worker.start()

    @app.on_event("shutdown")
    async def _shutdown():
        await worker.stop()

    from app.api.routes_turns import router as turns_router
    from app.api.routes_finance import router as finance_router
    from app.api.routes_trains import router as trains_router
    app.include_router(turns_router)
    app.include_router(finance_router)
    app.include_router(trains_router)

    return app


def _build_tool_registry() -> ToolRegistry | None:
    raw = os.getenv("TOOL_ENDPOINTS_JSON", "").strip()
    endpoints = []
    if raw:
        try:
            entries = json.loads(raw)
        except json.JSONDecodeError:
            entries = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            name = entry.get("name")
            url = entry.get("url")
            description = entry.get("description", "")
            method = entry.get("method", "POST")
            timeout_s = entry.get("timeout_s", 20.0)
            if not name or not url:
                continue
            endpoints.append(
                ToolEndpoint(
                    name=name,
                    description=description,
                    url=url,
                    method=method,
                    timeout_s=timeout_s,
                )
            )

    endpoints.append(
        ToolEndpoint(
            name="finance_price",
            description="Retourne le dernier prix connu pour un symbole financier (ex: BTCUSD).",
            parameters={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Symbole financier à interroger (ex: BTCUSD, AAPL.US).",
                    }
                },
                "required": ["symbol"],
            },
            handler=_handle_finance_tool,
        )
    )
    endpoints.append(
        ToolEndpoint(
            name="train_line_l_departures",
            description="Retourne les prochains départs de la ligne L (Transilien) pour une gare IDF.",
            parameters={
                "type": "object",
                "properties": {
                    "stop_area_id": {
                        "type": "string",
                        "description": "Identifiant stop_area SNCF (ex: stop_area:IDF:SA:8754523).",
                    },
                    "count": {
                        "type": "integer",
                        "description": "Nombre de départs à retourner (1-20).",
                    },
                },
                "required": ["stop_area_id"],
            },
            handler=_handle_line_l_tool,
        )
    )
    if not endpoints:
        return None
    return ToolRegistry(endpoints)


def _handle_finance_tool(arguments: dict) -> dict:
    symbol = ""
    if isinstance(arguments, dict):
        symbol = str(arguments.get("symbol") or "").strip()
    return fetch_finance_price(symbol)


def _handle_line_l_tool(arguments: dict) -> dict:
    stop_area_id = ""
    count = 5
    if isinstance(arguments, dict):
        stop_area_id = str(arguments.get("stop_area_id") or "").strip()
        if arguments.get("count") is not None:
            try:
                count = int(arguments.get("count"))
            except (TypeError, ValueError):
                count = 5
    return fetch_line_l_departures(stop_area_id, count)
