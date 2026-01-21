from __future__ import annotations

from typing import Dict, Optional
from .models import Turn


class TurnStore:
    def __init__(self) -> None:
        self._turns: Dict[str, Turn] = {}

    def put(self, turn: Turn) -> None:
        self._turns[turn.turn_id] = turn

    def get(self, turn_id: str) -> Optional[Turn]:
        return self._turns.get(turn_id)

    def all(self) -> Dict[str, Turn]:
        return self._turns
