from __future__ import annotations

import json
from typing import Optional, List, Dict, Any

from .models import TurnStatus, Turn
from app.stt.whisper_asr import WhisperASR
from app.llm.llm_client import OpenAIChatClient
from app.tts.piper_tts import PiperTTS
from app.tools.tool_registry import ToolRegistry


class VoicePipeline:
    def __init__(
        self,
        asr: WhisperASR,
        llm: OpenAIChatClient,
        tts: PiperTTS,
        tool_registry: Optional[ToolRegistry] = None,
        system_prompt: str = "Tu es un assistant vocal local, concis et utile. Réponds en français.",
        max_history_turns: int = 6,
    ) -> None:
        self.asr = asr
        self.llm = llm
        self.tts = tts
        self.tool_registry = tool_registry
        self.system_prompt = system_prompt
        self.max_history_turns = max_history_turns

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
        messages: List[Dict[str, Any]] = [{"role": "system", "content": self.system_prompt}] + history + [
            {"role": "user", "content": transcript or ""}
        ]
        tool_calls: List[Dict[str, Any]] = []
        if self.tool_registry:
            answer, tool_calls, dt_llm = self.llm.chat_with_tools(
                messages=messages,
                tools=self.tool_registry.tool_specs(),
            )
        else:
            answer, dt_llm = self.llm.chat(messages)
        turn.timings["llm_s"] = dt_llm

        if tool_calls and self.tool_registry:
            tool_messages: List[Dict[str, Any]] = [
                {
                    "role": "assistant",
                    "content": answer,
                    "tool_calls": [
                        {
                            "id": call["id"],
                            "type": "function",
                            "function": {
                                "name": call["name"],
                                "arguments": call["arguments"],
                            },
                        }
                        for call in tool_calls
                    ],
                }
            ]
            tool_results: List[Dict[str, Any]] = []
            for call in tool_calls:
                arguments: Dict[str, Any] = {}
                try:
                    arguments = json.loads(call.get("arguments") or "{}")
                except json.JSONDecodeError:
                    arguments = {}
                result = self.tool_registry.execute(call["name"], arguments)
                tool_results.append(
                    {
                        "tool_call_id": call["id"],
                        "name": call["name"],
                        "result": result,
                    }
                )
                tool_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "name": call["name"],
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )

            followup_messages = messages + tool_messages
            final_answer, dt_tool_llm = self.llm.chat(followup_messages)
            turn.timings["llm_tools_s"] = dt_tool_llm
            turn.assistant_text = final_answer
            turn.tool_calls = tool_calls
            turn.tool_results = tool_results
        else:
            turn.assistant_text = answer

        # push history (MVP)
        history.append({"role": "user", "content": transcript or ""})
        history.append({"role": "assistant", "content": turn.assistant_text or ""})
        max_messages = self.max_history_turns * 2
        if len(history) > max_messages:
            history[:] = history[-max_messages:]

        # 3) TTS
        turn.status = TurnStatus.synthesizing
        out_path = f"app/tts/outputs/turn_{turn.turn_id}.wav"
        audio_path, dt_tts = self.tts.synthesize(turn.assistant_text or "", out_path)
        turn.audio_out_path = audio_path
        turn.timings["tts_s"] = dt_tts

        turn.status = TurnStatus.done
        return turn
