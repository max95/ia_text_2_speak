from __future__ import annotations

from dataclasses import dataclass
from fastapi import FastAPI

from app.core.store import TurnStore
from app.core.worker import WorkerPool
from app.core.pipeline import VoicePipeline
from app.stt.whisper_asr import WhisperASR
from app.llm.llamacpp_client import LlamaCppClient
from app.tts.piper_tts import PiperTTS


@dataclass
class Deps:
    store: TurnStore
    worker: WorkerPool


deps: Deps  # rempli au startup


def create_app() -> FastAPI:
    app = FastAPI(title="ia_text_2_speak")

    store = TurnStore()

    asr = WhisperASR(model_name="small", language="fr")
    llm = LlamaCppClient(base_url="http://127.0.0.1:8080")
    tts = PiperTTS(model_path="app/tts/models/fr_FR-upmc-medium.onnx")

    pipeline = VoicePipeline(asr=asr, llm=llm, tts=tts)

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
