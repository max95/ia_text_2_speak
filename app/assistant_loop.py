#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

# Chemins déjà existants dans TON repo
MIC_WAV = Path("app/stt/outputs/mic.wav")
TTS_WAV = Path("app/tts/outputs/assistant.wav")


def wait_for_wake_word():
    """
    Lance ton script Vosk existant et bloque
    jusqu'à détection du mot-clé.
    """
    print("[assistant] écoute wake word (vosk)")
    subprocess.run(
        [sys.executable, "app/stt/vosk_hotwords.py"],
        check=True,
    )


def record_question():
    """
    Lance ton script d'enregistrement micro existant.
    Il doit produire mic.wav
    """
    print("[assistant] enregistrement question")
    subprocess.run(
        [sys.executable, "app/stt/whisper_asr.py"],
        check=True,
    )
    assert MIC_WAV.exists(), "mic.wav non généré"


def run_pipeline():
    """
    Appelle TA pipeline FastAPI existante
    (Whisper -> LLM -> Piper)
    """
    print("[assistant] appel pipeline FastAPI")

    import requests

    r = requests.post(
        "http://127.0.0.1:8000/v1/turns",
        files={"audio": MIC_WAV.open("rb")},
        timeout=180,
    )
    r.raise_for_status()
    turn_id = r.json()["turn_id"]

    # poll
    while True:
        time.sleep(0.5)
        s = requests.get(
            f"http://127.0.0.1:8000/v1/turns/{turn_id}",
            timeout=120,
        ).json()
        if s["status"] == "done":
            audio_url = s["audio_url"]
            break
        if s["status"] == "error":
            raise RuntimeError(s.get("error"))

    # download wav
    wav = requests.get(
        f"http://127.0.0.1:8000{audio_url}",
        timeout=30,
    ).content

    TTS_WAV.parent.mkdir(parents=True, exist_ok=True)
    TTS_WAV.write_bytes(wav)


def play_audio():
    print("[assistant] lecture réponse")
    if sys.platform == "darwin":
        subprocess.run(["afplay", str(TTS_WAV)], check=False)
    else:
        subprocess.run(["aplay", str(TTS_WAV)], check=False)


def main():
    print("=== Assistant vocal prêt ===")
    while True:
        wait_for_wake_word()
        record_question()
        run_pipeline()
        play_audio()
        print("[assistant] retour à l'écoute\n")


if __name__ == "__main__":
    main()
