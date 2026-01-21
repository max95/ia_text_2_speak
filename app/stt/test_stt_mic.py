#!/usr/bin/env python3
"""
Enregistre depuis le micro puis transcrit en français via faster-whisper.

Usage:
  python stt_mic.py --seconds 6
  python stt_mic.py --seconds 8 --outfile /tmp/mic.wav
  python stt_mic.py --list-devices
  python stt_mic.py --device 1 --seconds 6

Notes:
- Pour de meilleures perfs CPU sur Mac ancien : modèle "base" ou "small" + int8
- Le script écrit un WAV (PCM 16-bit) puis lance la transcription.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write as wav_write

from faster_whisper import WhisperModel


def list_devices() -> None:
    print(sd.query_devices())


def record_wav(
    outfile: Path,
    seconds: float,
    samplerate: int = 16000,
    channels: int = 1,
    device: int | None = None,
) -> Path:
    outfile.parent.mkdir(parents=True, exist_ok=True)

    print(f"Recording {seconds:.1f}s @ {samplerate}Hz, channels={channels}, device={device}")
    print("Speak now...")

    try:
        sd.default.samplerate = samplerate
        sd.default.channels = channels
        if device is not None:
            sd.default.device = (device, None)  # input device, keep output default

        audio = sd.rec(int(seconds * samplerate), dtype="float32")
        sd.wait()
    except Exception as e:
        raise RuntimeError(
            "Impossible d'enregistrer depuis le micro. "
            "Vérifie les permissions Micro de macOS et le device sélectionné."
        ) from e

    # Convert float32 [-1,1] -> int16 PCM
    audio = np.clip(audio, -1.0, 1.0)
    audio_i16 = (audio * 32767.0).astype(np.int16)

    wav_write(str(outfile), samplerate, audio_i16)
    print(f"Saved: {outfile}")
    return outfile


def transcribe(
    wav_path: Path,
    model_name: str = "small",
    language: str = "fr",
    device: str = "cpu",
    compute_type: str = "int8",
) -> str:
    print(f"Loading model: {model_name} (device={device}, compute_type={compute_type})")
    model = WhisperModel(model_name, device=device, compute_type=compute_type)

    segments, info = model.transcribe(
        str(wav_path),
        language=language,
        vad_filter=True,  # aide à ignorer les silences
    )

    print(f"Detected language: {info.language} (p={info.language_probability:.2f})")
    text_parts = []
    for seg in segments:
        text_parts.append(seg.text)

    return "".join(text_parts).strip()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seconds", type=float, default=6.0, help="Durée d'enregistrement")
    p.add_argument("--samplerate", type=int, default=16000, help="Fréquence d'échantillonnage (Hz)")
    p.add_argument("--channels", type=int, default=1, help="Nombre de canaux (1=mono)")
    p.add_argument("--outfile", type=str, default="outputs/mic.wav", help="Chemin WAV de sortie")
    p.add_argument("--device", type=int, default=None, help="Index du device micro (voir --list-devices)")
    p.add_argument("--list-devices", action="store_true", help="Liste les devices audio et quitte")

    p.add_argument("--model", type=str, default="small", help="base | small | medium | large-v3 ...")
    p.add_argument("--lang", type=str, default="fr", help="Langue (ex: fr)")
    p.add_argument("--compute-type", type=str, default="int8", help="int8 recommandé sur CPU")
    args = p.parse_args()

    if args.list_devices:
        list_devices()
        return 0

    wav_path = record_wav(
        outfile=Path(args.outfile),
        seconds=args.seconds,
        samplerate=args.samplerate,
        channels=args.channels,
        device=args.device,
    )

    text = transcribe(
        wav_path=wav_path,
        model_name=args.model,
        language=args.lang,
        compute_type=args.compute_type,
    )

    print("\n--- TRANSCRIPTION ---")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
