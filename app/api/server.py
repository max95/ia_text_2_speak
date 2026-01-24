from __future__ import annotations

from dataclasses import dataclass
import json
import os
from fastapi import FastAPI

from app.core.store import TurnStore
from app.core.worker import WorkerPool
from app.core.pipeline import VoicePipeline
from app.stt.whisper_asr import WhisperASR
from app.llm.llm_client import LlamaCppClient
from app.llm.llm_client import OpenAIChatClient
from app.tts.piper_tts import PiperTTS
from app.tools.tool_registry import ToolEndpoint, ToolRegistry


@dataclass
class Deps:
    store: TurnStore
    worker: WorkerPool


deps: Deps  # rempli au startup


def create_app() -> FastAPI:
    app = FastAPI(title="ia_text_2_speak")

    store = TurnStore()

    asr = WhisperASR(model_name="small", language="fr")
    #llm = LlamaCppClient(base_url="http://127.0.0.1:8080")
    llm = OpenAIChatClient()
    tts = PiperTTS(model_path="app/tts/models/fr_FR-upmc-medium.onnx")

    tool_registry = _load_tool_registry()
    pipeline = VoicePipeline(asr=asr, llm=llm, tts=tts, tool_registry=tool_registry)

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
    app.include_router(turns_router)

    return app


def _load_tool_registry() -> ToolRegistry | None:
    raw = os.getenv("TOOL_ENDPOINTS_JSON", "").strip()
    if not raw:
        return None
    try:
        entries = json.loads(raw)
    except json.JSONDecodeError:
        return None
    endpoints = []
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
    if not endpoints:
        return None
    return ToolRegistry(endpoints)
