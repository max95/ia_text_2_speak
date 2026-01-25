from __future__ import annotations

import time
from typing import Optional, Tuple, List
import numpy as np
import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel


def record_to_wav(
    path: str,
    max_seconds: float,
    sr: int = 16000,
    silence_duration: float = 1.0,
    silence_threshold: float = 0.01,
    block_duration: float = 0.1,
) -> None:
    max_samples = int(max_seconds * sr)
    block_samples = int(block_duration * sr)
    silence_samples = int(silence_duration * sr)
    frames: List[np.ndarray] = []
    total_samples = 0
    silent_run = 0
    has_speech = False

    with sd.InputStream(samplerate=sr, channels=1, dtype="float32") as stream:
        while total_samples < max_samples:
            data, _ = stream.read(block_samples)
            frames.append(data.copy())
            total_samples += len(data)

            rms = float(np.sqrt(np.mean(np.square(data))))
            if rms >= silence_threshold:
                has_speech = True
                silent_run = 0
            elif has_speech:
                silent_run += len(data)
                if silent_run >= silence_samples:
                    break

    audio = np.concatenate(frames, axis=0) if frames else np.zeros((0, 1), dtype="float32")
    sf.write(path, audio, sr)

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

if __name__ == "__main__":
    wav = "app/stt/outputs/mic.wav"
    print("[rec] enregistrement (arrêt au silence)...")
    record_to_wav(wav, max_seconds=10.0, sr=16000)

    asr = WhisperASR(model_name="small", device="cpu", compute_type="int8", language="fr")
    text, dt = asr.transcribe(wav)
    print(f"[asr] ({dt:.2f}s) -> {text}")
