#!/usr/bin/env python3
"""
Usage:
  python llama_local.py "Ton prompt ici"
  python llama_local.py            # mode interactif

Prérequis:
  - llama.cpp installé (commande: llama-server accessible dans le PATH)
  - modèle GGUF local (ex: models/Llama-3.2-3B-Instruct-IQ3_M.gguf)
  - pip install requests

Ce script:
  1) démarre llama-server si non déjà actif
  2) interroge l'API OpenAI-like (/v1/chat/completions)
  3) affiche la réponse
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from typing import Optional

import requests


DEFAULT_MODEL_PATH = "models/Llama-3.2-3B-Instruct-IQ3_M.gguf"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080
DEFAULT_CTX = 4096
DEFAULT_MAX_TOKENS = 350
DEFAULT_TEMPERATURE = 0.2

# System prompt "verrou" pour éviter les sorties JSON / tool-calling halluciné
DEFAULT_SYSTEM_PROMPT = (
    "Tu es un assistant conversationnel. "
    "Réponds uniquement en texte naturel. "
    "Ne réponds jamais en JSON. "
    "N'invente pas d'appel de fonction ou d'outil. "
    "Si l'utilisateur dit bonjour, réponds simplement et brièvement."
)


def sanitize_utf8(s: str) -> str:
    """
    Evite les erreurs JSON côté serveur (surrogates invalides).
    Remplace les caractères non encodables UTF-8 par '�'.
    """
    return s.encode("utf-8", errors="replace").decode("utf-8")


def base_url(host: str, port: int) -> str:
    return f"http://{host}:{port}/v1"


def is_server_up(url: str, timeout_s: float = 1.0) -> bool:
    try:
        r = requests.get(f"{url}/models", timeout=timeout_s)
        return r.status_code == 200
    except Exception:
        return False


def wait_server(url: str, max_wait_s: int = 30) -> None:
    start = time.time()
    while time.time() - start < max_wait_s:
        if is_server_up(url, timeout_s=1.0):
            return
        time.sleep(0.5)
    raise RuntimeError(f"Le serveur llama.cpp ne répond pas après {max_wait_s}s: {url}")


def start_server(model_path: str, host: str, port: int, ctx: int) -> subprocess.Popen:
    cmd = [
        "llama-server",
        "-m",
        model_path,
        "--host",
        host,
        "--port",
        str(port),
        "-c",
        str(ctx),
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=sys.stdout,
        stderr=sys.stderr,
        preexec_fn=os.setsid if hasattr(os, "setsid") else None,
    )
    return proc


def stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return

    try:
        if hasattr(os, "getpgid"):
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        else:
            proc.terminate()
    except Exception:
        proc.terminate()

    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            if hasattr(os, "getpgid"):
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            else:
                proc.kill()
        except Exception:
            proc.kill()
        proc.wait(timeout=5)


def looks_like_json(s: str) -> bool:
    s = s.strip()
    return (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]"))


def chat_completion(
    url: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    system_prompt: str,
) -> str:
    prompt = sanitize_utf8(prompt)
    system_prompt = sanitize_utf8(system_prompt)

    payload = {
        "model": "local",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    r = requests.post(f"{url}/chat/completions", json=payload, timeout=300)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]


def ask_text_only(
    url: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    system_prompt: str,
) -> str:
    """
    Appel principal avec garde-fou:
    - si la sortie ressemble à du JSON, on renforce la consigne et on retente une fois
    """
    out = chat_completion(url, prompt, temperature, max_tokens, system_prompt)

    if looks_like_json(out):
        # On tente de détecter un faux "tool call"
        try:
            obj = json.loads(out)
            if isinstance(obj, dict) and "name" in obj and "parameters" in obj:
                reinforced_system = (
                    system_prompt
                    + " IMPORTANT: ne produis pas d'objet JSON. Réponds par une phrase en texte."
                )
                out2 = chat_completion(url, prompt, temperature, max_tokens, reinforced_system)
                return out2
        except Exception:
            # JSON invalide -> on retente aussi
            reinforced_system = (
                system_prompt + " IMPORTANT: ne produis pas de JSON. Réponds en texte."
            )
            out2 = chat_completion(url, prompt, temperature, max_tokens, reinforced_system)
            return out2

    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("prompt", nargs="?", default=None, help="Prompt utilisateur")
    p.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Chemin vers le .gguf")
    p.add_argument("--host", default=DEFAULT_HOST, help="Host du serveur")
    p.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port du serveur")
    p.add_argument("--ctx", type=int, default=DEFAULT_CTX, help="Taille de contexte (-c)")
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="max_tokens")
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="temperature")
    p.add_argument("--system", default=DEFAULT_SYSTEM_PROMPT, help="System prompt")
    p.add_argument("--no-autostart", action="store_true", help="Ne démarre pas llama-server")
    p.add_argument("--stop-after", action="store_true", help="Stoppe le serveur si démarré par le script")

    args = p.parse_args()
    url = base_url(args.host, args.port)

    started_proc: Optional[subprocess.Popen] = None

    # Démarrage serveur si nécessaire
    if not is_server_up(url):
        if args.no_autostart:
            print(f"Serveur non détecté sur {url} (et --no-autostart activé).", file=sys.stderr)
            return 2

        if not os.path.exists(args.model):
            print(f"Modèle introuvable: {args.model}", file=sys.stderr)
            return 2

        print(f"Démarrage llama-server sur {url} avec modèle: {args.model}")
        started_proc = start_server(args.model, args.host, args.port, args.ctx)
        wait_server(url, max_wait_s=60)

    try:
        # Mode "one shot"
        if args.prompt:
            out = ask_text_only(url, args.prompt, args.temperature, args.max_tokens, args.system)
            print(out)
            return 0

        # Mode interactif (boucle)
        print("Mode interactif. Ctrl+C pour quitter.")
        while True:
            try:
                line = input("> ").strip()
            except EOFError:
                print()
                break
            if not line:
                continue
            out = ask_text_only(url, line, args.temperature, args.max_tokens, args.system)
            print(out)

        return 0

    finally:
        if started_proc is not None and args.stop_after:
            stop_server(started_proc)


if __name__ == "__main__":
    raise SystemExit(main())
