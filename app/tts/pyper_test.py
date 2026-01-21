import subprocess
from pathlib import Path
from datetime import datetime
import soundfile as sf
import sounddevice as sd

def play_wav(path: str):
    data, samplerate = sf.read(path, dtype="float32")
    sd.play(data, samplerate)
    sd.wait()  # bloque jusqu'à la fin de la lecture

def synthesize_piper(
    text: str | None = None,
    model_path: str = "models/fr_FR-upmc-medium.onnx",
    output_dir: str = "outputs",
    piper_bin: str = "piper"
) -> Path:
    """
    Synthétise une phrase en audio WAV avec Piper.

    - text : phrase à synthétiser (français par défaut si None)
    - model_path : chemin vers le modèle Piper (.onnx)
    - output_dir : dossier de sortie
    - piper_bin : chemin ou nom du binaire piper

    Retourne le chemin du fichier audio généré.
    """

    DEFAULT_TEST_TEXT = (
        "Bonjour. "
        "Je suis en train de tester la synthèse vocale en local avec Piper. "
        "L'objectif est de vérifier que la voix est naturelle, fluide, et agréable à écouter. "
        "Si vous entendez ce message clairement, alors la configuration audio fonctionne correctement. "
        "Nous pourrons ensuite passer à des tests plus avancés, avec des conversations et des réponses dynamiques."
    )

    DEFAULT_TEST_DEBUG = (
        "Première phrase courte. "
        "Voici une phrase un peu plus longue, avec plusieurs segments, afin d'évaluer le rythme et la fluidité. "
        "Attention aux chiffres : vingt-trois, mille deux cent quarante-cinq. "
        "Et enfin, une dernière phrase pour conclure le test."
    )

    if not text:
        text =     DEFAULT_TEST_DEBUG

    model_path = Path(model_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_wav = output_dir / f"piper_test_{timestamp}.wav"

    command = [
        piper_bin,
        "--model", str(model_path),
        "--output_file", str(output_wav),
    ]

    try:
        process = subprocess.run(
            command,
            input=text,
            text=True,
            check=True
        )
    except FileNotFoundError:
        raise RuntimeError("Binaire 'piper' introuvable. Vérifie son installation.")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Erreur lors de l'exécution de Piper : {e}")

    return output_wav


if __name__ == "__main__":
    wav_path = synthesize_piper()
    print(f"Fichier audio généré : {wav_path}")
    play_wav(str(wav_path))

