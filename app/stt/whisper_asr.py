from __future__ import annotations

import time
from typing import Optional, Tuple, List

from faster_whisper import WhisperModel


class WhisperASR:
    """
    Wrapper ASR basé sur faster-whisper.

    Notes:
      - compute_type impacte beaucoup perf/mémoire.
      - Sur CPU: int8 ou int8_float16 sont souvent les meilleurs compromis.
      - language="fr" si tu veux forcer le français. Sinon None => auto.
    """

    def __init__(
        self,
        model_name: str = "small",
        device: str = "cpu",                 # "cpu" ou "cuda"
        compute_type: str = "int8",          # CPU: "int8" / "int8_float16" ; GPU: "float16"
        language: Optional[str] = "fr",      # "fr" ou None (auto)
        beam_size: int = 5,
        vad_filter: bool = True,
        vad_parameters: Optional[dict] = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.beam_size = beam_size
        self.vad_filter = vad_filter
        self.vad_parameters = vad_parameters or {
            # Valeurs raisonnables pour couper le silence
            "min_silence_duration_ms": 500,
        }

        # download_root optionnel si tu veux forcer un cache local
        self._model = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
        )

    def transcribe(self, wav_path: str) -> Tuple[str, float]:
        """
        Retourne (texte, durée_secondes).
        """
        t0 = time.time()

        segments, info = self._model.transcribe(
            wav_path,
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=self.vad_filter,
            vad_parameters=self.vad_parameters,
            # Tu peux ajouter ici:
            # condition_on_previous_text=False,
            # temperature=0.0,
        )

        parts: List[str] = []
        for seg in segments:
            if seg.text:
                parts.append(seg.text.strip())

        text = " ".join([p for p in parts if p]).strip()
        dt = time.time() - t0
        return text, dt
