# IA Text to Speak — Assistant vocal local

Ce projet met en place un assistant vocal **local** qui écoute un mot-clé, enregistre la question, la transcrit en texte, génère une réponse via un LLM (local ou OpenAI), puis synthétise la réponse en audio. Le point d’entrée de l’usage “bout-en-bout” est `app/assistant_loop.py`.

## Objectifs du programme

- **Écouter en continu** un mot-clé (wake word) pour démarrer une interaction.
- **Transcrire la voix** en texte avec une brique ASR fiable.
- **Générer une réponse** via un LLM, en local ou via API.
- **Restituer la réponse** en audio pour une interaction naturelle.

## Architecture (4 briques principales)

### 1) Wake word (Vosk)
- **Ce que fait la brique :** écoute en streaming micro et détecte un mot-clé (ex: “jarvis”).
- **Avantages techniques :**
  - **Très léger CPU** et temps réel, parfait pour une écoute continue.
  - **Local/offline** (aucune donnée envoyée), donc respect de la vie privée.
  - **Simple à déployer** : modèle Vosk + micro.
- Script : `app/stt/vosk_hotwords.py`.

### 2) Transcription (Whisper via faster-whisper)
- **Ce que fait la brique :** enregistre le micro puis convertit l’audio en texte.
- **Avantages techniques :**
  - **Très bonne précision** sur la reconnaissance vocale, y compris en français.
  - **Contrôle des performances** via `compute_type` (ex. `int8` pour CPU).
  - **VAD intégré** (filtre de silence) pour améliorer la qualité.
- Script : `app/stt/whisper_asr.py`.

### 3) LLM (local ou OpenAI)
- **Ce que fait la brique :** génère une réponse à partir du texte.
- **Options :**
  - **Local (llama.cpp)** : utilisé via une API type OpenAI (`/v1/chat/completions`).
  - **OpenAI (ChatGPT)** : plus rapide à démarrer, plus qualitatif selon le modèle.
- **Avantages techniques :**
  - **LLM local** : confidentialité totale, pas de coût API, fonctionne hors-ligne.
  - **OpenAI** : **rapidité**, **qualité**, **simplicité** (clé API) — c’est **le choix par défaut** dans ce code.
- Code : `app/llm/llm_client.py`.

### 3bis) Outils (tool calling)
- **Objectif :** déclencher un appel API vers un outil après la réponse du LLM, lorsque c’est pertinent.
- **Principe :** le LLM peut sélectionner un outil parmi une liste déclarée, puis le pipeline appelle l’endpoint et renvoie le résultat au LLM pour produire la réponse finale.
- **Configuration :** définir `TOOL_ENDPOINTS_JSON` avec la liste des outils disponibles (nom, description, url, méthode).
- **Outil intégré :** `finance_price` est disponible par défaut pour récupérer un prix de marché (ex: BTCUSD).

Exemple :
```bash
export TOOL_ENDPOINTS_JSON='[
  {"name": "calendar_create", "description": "Créer un évènement", "url": "http://127.0.0.1:9000/calendar", "method": "POST"},
  {"name": "weather_lookup", "description": "Récupérer la météo", "url": "http://127.0.0.1:9000/weather", "method": "POST"}
]'
```

### 4) Synthèse vocale (Piper TTS)
- **Ce que fait la brique :** transforme le texte en audio (WAV) pour la réponse.
- **Avantages techniques :**
  - **Local/offline** et rapide.
  - **Voix naturelles** avec des modèles ONNX.
  - **Pipeline léger** et simple à automatiser.
- Code : `app/tts/piper_tts.py`.

## Flux global

1. **Vosk** attend le mot-clé.
2. Lecture d’un **jingle** pour indiquer que l’assistant écoute.
3. **Whisper** transcrit la question.
4. **LLM** génère la réponse.
5. **Piper** synthétise la réponse en audio.

Ce flux est orchestré dans `app/assistant_loop.py` et la partie pipeline est gérée côté API FastAPI (`app/api/server.py`).

## Installation rapide

### 1) Dépendances Python
```bash
pip install -r requirements.txt
pip install openai vosk soundfile
```

### 2) Modèles requis
- **Vosk** : modèle FR à placer dans `app/stt/models/vosk-model-small-fr-0.22`.
- **Whisper** : `faster-whisper` télécharge automatiquement les modèles (ex. `small`).
- **Piper** : modèle `.onnx` dans `app/tts/models/`.
  - Par défaut dans l’API : `app/tts/models/fr_FR-upmc-medium.onnx`.
  - Par défaut dans la classe Piper : `app/tts/models/fr_FR-gilles-low.onnx`.
- **LLM local (optionnel)** : un modèle `.gguf` si vous utilisez llama.cpp.

### 3) LLM OpenAI (par défaut)
Définissez la variable d’environnement :
```bash
export OPENAI_API_KEY="..."
```
Le client est instancié par défaut dans `app/api/server.py`.

## Lancer le programme

### 1) Démarrer l’API FastAPI
```bash
uvicorn main:app --host 127.0.0.1 --port 8000
```

### 2) Lancer la boucle d’assistant
```bash
python app/assistant_loop.py
```

> Le script écoute le mot-clé, joue un jingle, enregistre la question, appelle l’API, puis lit la réponse audio.

## API (générée par ChatGPT, acceptée telle quelle)

> **Important :** la partie API (`app/api`) a été générée par ChatGPT et **acceptée en l’état sans revue approfondie**. Il est recommandé de la relire si vous souhaitez la durcir (sécurité, validation d’entrée, robustesse).

### Endpoints
- `POST /v1/turns`
  - Entrée : un fichier audio (`audio`) en multipart.
  - Sortie : `turn_id` + `session_id`.
- `GET /v1/turns/{turn_id}`
  - Statut (`status`), transcript, réponse texte, URL audio si disponible.
- `GET /v1/turns/{turn_id}/audio`
  - Télécharge le WAV de la réponse.
- `GET /v1/finance/price?symbol=...`
  - Retourne le dernier prix disponible pour un symbole financier (source Stooq).

Cette API pilote la pipeline : **Whisper → LLM → Piper**.

## Tests par brique

Chaque dossier contient un script de test :

### Wake word (Vosk)
```bash
python app/stt/test_hotwords_wosk.py
```

### Transcription (Whisper)
- Test micro + transcription :
```bash
python app/stt/test_stt_mic.py --seconds 6
```
- Test avec un fichier WAV existant :
```bash
python app/stt/whisper_test.py
```

### LLM local (llama.cpp)
```bash
python app/llm/llama_test.py "Bonjour !"
```

### Synthèse vocale (Piper)
```bash
python app/tts/pyper_test.py
```

## Structure des dossiers

```
app/
  assistant_loop.py      # boucle principale (wake word -> ASR -> LLM -> TTS)
  api/                   # API FastAPI (turns, worker, pipeline)
  core/                  # pipeline, worker, store
  stt/                   # hotword + transcription (Vosk + Whisper)
  llm/                   # clients LLM (local + OpenAI)
  tts/                   # Piper TTS
```

## Remarques importantes

- **Audio** : `sounddevice` nécessite des permissions micro, surtout sur macOS.
- **Performance** : ajustez `compute_type` et le modèle Whisper si besoin.
- **LLM local** : nécessite `llama-server` (llama.cpp) exposé sur `http://127.0.0.1:8080` si vous souhaitez l’utiliser.
