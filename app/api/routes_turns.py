from __future__ import annotations

import os
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse

from app.core.models import Turn
from app.api.server import deps

router = APIRouter()


@router.post("/v1/turns")
async def create_turn(audio: UploadFile = File(...), session_id: str | None = None):
    turn = Turn.new(session_id=session_id)

    # stocker audio input
    in_dir = "app/stt/outputs"
    os.makedirs(in_dir, exist_ok=True)
    in_path = os.path.join(in_dir, f"turn_{turn.turn_id}.wav")

    with open(in_path, "wb") as f:
        f.write(await audio.read())

    turn.audio_in_path = in_path
    deps.store.put(turn)

    await deps.worker.enqueue(turn.turn_id)

    return {"turn_id": turn.turn_id, "session_id": turn.session_id}


@router.get("/v1/turns/{turn_id}")
def get_turn(turn_id: str):
    turn = deps.store.get(turn_id)
    if not turn:
        raise HTTPException(status_code=404, detail="turn not found")

    audio_url = None
    if turn.audio_out_path and os.path.exists(turn.audio_out_path):
        audio_url = f"/v1/turns/{turn_id}/audio"

    return {
        "turn_id": turn.turn_id,
        "session_id": turn.session_id,
        "status": turn.status,
        "transcript": turn.transcript,
        "assistant_text": turn.assistant_text,
        "tool_calls": turn.tool_calls,
        "tool_results": turn.tool_results,
        "audio_url": audio_url,
        "error": turn.error,
        "timings": turn.timings,
    }


@router.get("/v1/turns/{turn_id}/audio")
def get_turn_audio(turn_id: str):
    turn = deps.store.get(turn_id)
    if not turn or not turn.audio_out_path:
        raise HTTPException(status_code=404, detail="audio not available")
    if not os.path.exists(turn.audio_out_path):
        raise HTTPException(status_code=404, detail="audio file missing")

    return FileResponse(
        turn.audio_out_path,
        media_type="audio/wav",
        filename=os.path.basename(turn.audio_out_path),
    )
