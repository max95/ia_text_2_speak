from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any
import time
import uuid


class TurnStatus(str, Enum):
    queued = "queued"
    transcribing = "transcribing"
    generating = "generating"
    synthesizing = "synthesizing"
    done = "done"
    error = "error"


@dataclass
class Turn:
    turn_id: str
    session_id: str
    status: TurnStatus = TurnStatus.queued

    audio_in_path: Optional[str] = None
    transcript: Optional[str] = None
    assistant_text: Optional[str] = None
    audio_out_path: Optional[str] = None

    error: Optional[str] = None

    created_at: float = field(default_factory=lambda: time.time())
    timings: Dict[str, float] = field(default_factory=dict)

    @staticmethod
    def new(session_id: Optional[str] = None) -> "Turn":
        return Turn(
            turn_id=str(uuid.uuid4()),
            session_id=session_id or str(uuid.uuid4()),
        )
