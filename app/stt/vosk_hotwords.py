#!/usr/bin/env python3
"""
Détection simple d'un mot-clé via Vosk (streaming micro).

Mot-clé détecté : "dis jarvis"
- Faible CPU
- Temps réel
- Idéal pour wake word
"""

import json
import os
import queue
import sys
from collections import deque

import numpy as np
import sounddevice as sd
import soundfile as sf
from vosk import Model, KaldiRecognizer

# =====================
# CONFIG
# =====================
MODEL_PATH = "app/stt/models/vosk-model-small-fr-0.22"
SAMPLE_RATE = 16000
HOTWORD = "test"
BLOCK_SIZE = 4000
HOTWORD_GRAMMAR = [HOTWORD, f"dis {HOTWORD}"]
DETECTION_STREAK = 2
PRE_ROLL_SECONDS = 0.5
POST_ROLL_SECONDS = 0.5
HOTWORD_CONTEXT_WAV = "app/stt/outputs/hotword_context.wav"

# =====================
# AUDIO CALLBACK
# =====================
audio_queue = queue.Queue()


def audio_callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    audio_queue.put(bytes(indata))


# =====================
# MAIN
# =====================
def main():
    print("Loading Vosk model...")
    model = Model(MODEL_PATH)

    recognizer = KaldiRecognizer(model, SAMPLE_RATE, json.dumps(HOTWORD_GRAMMAR))
    recognizer.SetWords(False)

    print("Listening... (say 'Test')")

    detected_streak = 0
    pre_roll_blocks = max(1, int(PRE_ROLL_SECONDS * SAMPLE_RATE / BLOCK_SIZE))
    post_roll_blocks = max(1, int(POST_ROLL_SECONDS * SAMPLE_RATE / BLOCK_SIZE))
    pre_roll = deque(maxlen=pre_roll_blocks)
    post_roll: list[bytes] = []
    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        dtype="int16",
        channels=1,
        callback=audio_callback,
    ):
        while True:
            data = audio_queue.get()
            pre_roll.append(data)

            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                text = result.get("text", "").lower()
            else:
                result = json.loads(recognizer.PartialResult())
                text = result.get("partial", "").lower()

            if text:
                print(f"Heard: {text}")

            if HOTWORD in text:
                detected_streak += 1
            else:
                detected_streak = 0

            if detected_streak >= DETECTION_STREAK:
                print("\n🔥 HOTWORD DETECTED 🔥")
                print("Jarvis is listening...\n")
                for _ in range(post_roll_blocks):
                    post_roll.append(audio_queue.get())
                os.makedirs(os.path.dirname(HOTWORD_CONTEXT_WAV), exist_ok=True)
                raw_audio = b"".join(list(pre_roll) + post_roll)
                audio_i16 = np.frombuffer(raw_audio, dtype="int16")
                audio_i16 = audio_i16.reshape(-1, 1)
                sf.write(HOTWORD_CONTEXT_WAV, audio_i16, SAMPLE_RATE, subtype="PCM_16")
                break


if __name__ == "__main__":
    main()
