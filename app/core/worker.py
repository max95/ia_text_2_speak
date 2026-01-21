from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Optional

from .store import TurnStore
from .models import Turn, TurnStatus
from .pipeline import VoicePipeline


@dataclass
class Job:
    turn_id: str


class WorkerPool:
    def __init__(self, store: TurnStore, pipeline: VoicePipeline, concurrency: int = 1) -> None:
        self.store = store
        self.pipeline = pipeline
        self.concurrency = concurrency
        self.queue: asyncio.Queue[Job] = asyncio.Queue()
        self._tasks: list[asyncio.Task] = []

    async def start(self) -> None:
        for _ in range(self.concurrency):
            self._tasks.append(asyncio.create_task(self._worker_loop()))

    async def stop(self) -> None:
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

    async def enqueue(self, turn_id: str) -> None:
        await self.queue.put(Job(turn_id=turn_id))

    async def _worker_loop(self) -> None:
        while True:
            job = await self.queue.get()
            turn = self.store.get(job.turn_id)
            if not turn:
                self.queue.task_done()
                continue
            try:
                self.pipeline.run(turn)
            except Exception as e:
                turn.status = TurnStatus.error
                turn.error = str(e)
            finally:
                self.store.put(turn)
                self.queue.task_done()
