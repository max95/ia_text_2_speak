from __future__ import annotations

from typing import Optional, List, Dict

from .models import TurnStatus, Turn
from app.stt.whisper_asr import WhisperASR
from app.llm.llamacpp_client import LlamaCppClient
from app.tts.piper_tts import PiperTTS


class VoicePipeline:
    def __init__(
        self,
        asr: WhisperASR,
        llm: LlamaCppClient,
        tts: PiperTTS,
        system_prompt: str = "Tu es un assistant vocal local, concis et utile. Réponds en français.",
    ) -> None:
        self.asr = asr
        self.llm = llm
        self.tts = tts
        self.system_prompt = system_prompt

        # MVP: historique en mémoire par session
        self._history: dict[str, List[Dict[str, str]]] = {}

    def run(self, turn: Turn) -> Turn:
        if not turn.audio_in_path:
            turn.status = TurnStatus.error
            turn.error = "audio_in_path is missing"
            return turn

        # 1) Whisper
        turn.status = TurnStatus.transcribing
        transcript, dt_asr = self.asr.transcribe(turn.audio_in_path)
        turn.transcript = transcript
        turn.timings["asr_s"] = dt_asr

        # 2) LLM
        turn.status = TurnStatus.generating
        history = self._history.setdefault(turn.session_id, [])
        messages = [{"role": "system", "content": self.system_prompt}] + history + [
            {"role": "user", "content": transcript or ""}
        ]
        answer, dt_llm = self.llm.chat(messages)
        turn.assistant_text = answer
        turn.timings["llm_s"] = dt_llm

        # push history (MVP)
        history.append({"role": "user", "content": transcript or ""})
        history.append({"role": "assistant", "content": answer})

        # 3) TTS
        turn.status = TurnStatus.synthesizing
        out_path = f"app/tts/outputs/turn_{turn.turn_id}.wav"
        audio_path, dt_tts = self.tts.synthesize(answer, out_path)
        turn.audio_out_path = audio_path
        turn.timings["tts_s"] = dt_tts

        turn.status = TurnStatus.done
        return turn
