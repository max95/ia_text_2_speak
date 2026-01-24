from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Optional


class PiperTTS:
    def __init__(self, piper_bin: str = "piper", model_path: str = "app/tts/models/fr_FR-upmc-medium.onnx") -> None:
        self.piper_bin = piper_bin
        self.model_path = model_path

    def synthesize(
        self,
        text: Optional[str],
        out_wav_path: str,
        speaker: Optional[int] = None,
        text_path: Optional[str] = None,
    ) -> tuple[str, float]:
        Path(os.path.dirname(out_wav_path)).mkdir(parents=True, exist_ok=True)
        t0 = time.time()

        if text is None and text_path is not None:
            text = Path(text_path).read_text(encoding="utf-8")
        if text is None:
            raise ValueError("text is required when text_path is not provided")

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
