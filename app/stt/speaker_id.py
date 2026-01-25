#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torchaudio
from speechbrain.pretrained import EncoderClassifier


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def _load_profile(path: Path) -> np.ndarray:
    return np.load(path)


def _save_profile(path: Path, embedding: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, embedding)


def _iter_profiles(profile_dir: Path) -> Iterable[tuple[str, np.ndarray]]:
    if not profile_dir.exists():
        return []
    profiles = []
    for profile_path in profile_dir.glob("*.npy"):
        profiles.append((profile_path.stem, _load_profile(profile_path)))
    return profiles


def _load_wav(path: Path) -> torch.Tensor:
    wav, _ = torchaudio.load(path)
    return wav


def enroll(profile_name: str, wav_path: Path, profile_dir: Path) -> Path:
    wav = _load_wav(wav_path)
    encoder = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
    embedding = encoder.encode_batch(wav).squeeze(0).mean(dim=0).cpu().numpy()
    profile_path = profile_dir / f"{profile_name}.npy"
    _save_profile(profile_path, embedding)
    return profile_path


def identify(wav_path: Path, profile_dir: Path) -> tuple[str | None, float | None]:
    profiles = list(_iter_profiles(profile_dir))
    if not profiles:
        return None, None

    wav = _load_wav(wav_path)
    encoder = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
    embedding = encoder.encode_batch(wav).squeeze(0).mean(dim=0).cpu().numpy()

    best_name = None
    best_score = None
    for name, profile_embedding in profiles:
        score = _cosine_similarity(embedding, profile_embedding)
        if best_score is None or score > best_score:
            best_score = score
            best_name = name
    return best_name, best_score


def main() -> None:
    parser = argparse.ArgumentParser(description="Enroll or identify a speaker embedding.")
    parser.add_argument("wav", type=Path, help="Path to WAV file")
    parser.add_argument("--profiles-dir", type=Path, default=Path("app/stt/speaker_profiles"))
    parser.add_argument("--enroll", type=str, help="Profile name to enroll")
    args = parser.parse_args()

    if args.enroll:
        profile_path = enroll(args.enroll, args.wav, args.profiles_dir)
        print(f"[speaker-id] profil enregistré: {profile_path}")
        return

    name, score = identify(args.wav, args.profiles_dir)
    if name is None:
        print("[speaker-id] aucun profil trouvé")
    else:
        print(f"[speaker-id] meilleur profil: {name} (score={score:.3f})")


if __name__ == "__main__":
    main()
