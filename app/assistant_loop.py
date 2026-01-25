#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
import sounddevice as sd
import soundfile as sf

from tts.piper_tts import PiperTTS

# Chemins déjà existants dans TON repo
MIC_WAV = Path("app/stt/outputs/mic.wav")
TTS_WAV = Path("app/tts/outputs/assistant.wav")
JINGLE_WAV = Path("app/hotword_chime.wav")

def select_microphone() -> None:
    devices = sd.query_devices()
    input_devices = [
        (index, device)
        for index, device in enumerate(devices)
        if device.get("max_input_channels", 0) > 0
    ]
    if not input_devices:
        print("[assistant] aucun micro détecté, utilisation par défaut")
        return

    print("=== Microphones disponibles ===")
    for index, device in input_devices:
        print(f"{index}: {device.get('name')}")

    while True:
        choice = input("Choisissez l'index du micro: ").strip()
        if choice.isdigit():
            device_index = int(choice)
            if any(idx == device_index for idx, _ in input_devices):
                sd.default.device = (device_index, None)
                print(f"[assistant] micro sélectionné: {device_index}")
                return
        print("Index invalide, réessayez.")


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

def record_to_wav(path: str, seconds: float, sr: int = 16000) -> None:
    audio = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    sf.write(path, audio, sr)

def record_question():
    """
    Lance ton script d'enregistrement micro existant.
    Il doit produire mic.wav
    """
    print("[assistant] enregistrement question")
    """    subprocess.run(
        [sys.executable, "app/stt/whisper_asr.py"],
        check=True,
    )"""
    record_to_wav(MIC_WAV, seconds=4.0, sr=16000)
    assert MIC_WAV.exists(), "mic.wav non généré"


def run_pipeline(session_id: str | None = None) -> tuple[str, str | None]:
    """
    Appelle TA pipeline FastAPI existante
    (Whisper -> LLM -> Piper)
    """
    print("[assistant] appel pipeline FastAPI")

    import requests

    params = {}
    if session_id:
        params["session_id"] = session_id

    r = requests.post(
        "http://127.0.0.1:8000/v1/turns",
        files={"audio": MIC_WAV.open("rb")},
        params=params,
        timeout=180,
    )
    r.raise_for_status()
    response_payload = r.json()
    turn_id = response_payload["turn_id"]
    session_id = response_payload["session_id"]

    # poll
    assistant_text: str | None = None
    while True:
        time.sleep(0.5)
        s = requests.get(
            f"http://127.0.0.1:8000/v1/turns/{turn_id}",
            timeout=120,
        ).json()
        if s["status"] == "done":
            audio_url = s["audio_url"]
            assistant_text = s.get("assistant_text")
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
    return session_id, assistant_text


def is_follow_up_question(text: str | None) -> bool:
    if not text:
        return False
    return "?" in text


def play_audio(path: str):
    print("[assistant] lecture réponse")
    if sys.platform == "darwin":
        subprocess.run(["afplay", path], check=False)
    else:
        subprocess.run(["aplay", path], check=False)

def play_synthesize(text: str, wav_path: str = "audio/out.wav") -> str:
    tts = PiperTTS(
        piper_bin="piper",  # ou chemin absolu si nécessaire
        model_path="app/tts/models/fr_FR-upmc-medium.onnx",
    )
    out_path, dt = tts.synthesize(text=text, out_wav_path=wav_path)
    play_wav(out_path)
    return out_path

def play_wav(path: str):
    data, samplerate = sf.read(path, dtype="float32")
    sd.play(data, samplerate)
    sd.wait()  # bloque jusqu'à la fin de la lecture

def main():
    select_microphone()
    print("=== Assistant vocal prêt ===")
    session_id: str | None = None
    while True:
        wait_for_wake_word()
        #play_wav(str(JINGLE_WAV))
        play_synthesize("Que puis-je faire pour vous ?")
        follow_up = True
        while follow_up:
            record_question()
            session_id, assistant_text = run_pipeline(session_id=session_id)
            #play_audio()
            play_wav(str(TTS_WAV))
            if is_follow_up_question(assistant_text):
                print("[assistant] follow-up sans hotword")
                follow_up = True
            else:
                follow_up = False
        print("[assistant] retour à l'écoute\n")


if __name__ == "__main__":
    main()
