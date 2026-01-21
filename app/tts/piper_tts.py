from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Optional


class PiperTTS:
    def __init__(self, piper_bin: str = "piper", model_path: str = "app/tts/models/fr_FR-gilles-low.onnx") -> None:
        self.piper_bin = piper_bin
        self.model_path = model_path

    def synthesize(self, text: str, out_wav_path: str, speaker: Optional[int] = None) -> tuple[str, float]:
        Path(os.path.dirname(out_wav_path)).mkdir(parents=True, exist_ok=True)
        t0 = time.time()

        cmd = [
            self.piper_bin,
            "--model", self.model_path,
            "--output_file", out_wav_path,
        ]
        if speaker is not None:
            cmd += ["--speaker", str(speaker)]

        # piper lit le texte sur stdin
        p = subprocess.run(
            cmd,
            input=text.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if p.returncode != 0:
            raise RuntimeError(f"Piper failed: {p.stderr.decode('utf-8', errors='ignore')[:500]}")

        dt = time.time() - t0
        return out_wav_path, dt
