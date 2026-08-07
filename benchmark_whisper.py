#!/usr/bin/env python3
# =========================================================================
#  benchmark_whisper.py
#
#  Compara varios modelos ggml de whisper.cpp (tiny/base/small/etc.) sobre
#  el MISMO set de grabaciones .wav, usando exactamente el mismo comando
#  que agent.py (transcribe_audio) ejecuta en producción. Reporta:
#    - la transcripción cruda de cada modelo
#    - una similitud aproximada contra el texto de referencia (0-100%)
#    - el tiempo que tardó cada transcripción
#
#  CÓMO PREPARAR LOS AUDIOS DE PRUEBA (una sola vez):
#    1. Crea una carpeta ./benchmark_audio/
#    2. Graba 5-10 frases cortas representativas (las que normalmente le
#       dices a BMO) como .wav mono, ej: benchmark_audio/audio_01.wav
#       Puedes usar el propio agente: activa el wake word, di la frase, y
#       copia el input.wav que genera antes de que lo sobreescriba.
#    3. Crea ./benchmark_audio/reference.json con el texto real de cada
#       archivo:
#         {
#           "audio_01.wav": "¿Qué hora es?",
#           "audio_02.wav": "Apaga el computador",
#           "audio_03.wav": "Cuál es la raíz cuadrada de mil setecientos"
#         }
#
#  CÓMO DESCARGAR MODELOS PARA COMPARAR (downgrade/upgrade de tamaño):
#    cd whisper.cpp
#    bash ./models/download-ggml-model.sh tiny
#    bash ./models/download-ggml-model.sh base   # el que ya tienes
#    bash ./models/download-ggml-model.sh small
#    (NO uses los que terminan en ".en.bin" — son solo-inglés y con
#    stt_language="es" no van a funcionar.)
#
#  Uso:
#      python benchmark_whisper.py
#      python benchmark_whisper.py --models tiny base small
#      python benchmark_whisper.py --audio-dir ./benchmark_audio
# =========================================================================

import argparse
import csv
import json
import subprocess
import time
import unicodedata
from pathlib import Path

DEFAULT_MODELS = ["tiny", "base", "small"]
WHISPER_BIN = "./whisper.cpp/build/bin/whisper-cli"
MODELS_DIR = "./whisper.cpp/models"


def normalize(text):
    text = text.lower().strip()
    text = "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")
    return "".join(ch for ch in text if ch.isalnum() or ch.isspace())


def similarity(a, b):
    """Similitud simple por palabras (no es WER real, pero sirve para
    comparar modelos entre sí de forma consistente)."""
    a_words = normalize(a).split()
    b_words = normalize(b).split()
    if not a_words and not b_words:
        return 100.0
    if not a_words or not b_words:
        return 0.0
    # Longest common subsequence sobre palabras, normalizado por la
    # longitud de la referencia.
    n, m = len(a_words), len(b_words)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a_words[i - 1] == b_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[n][m]
    return round(lcs / len(a_words) * 100, 1)


def transcribe(model_path, wav_path, language="es", prompt_bias=""):
    cmd = [WHISPER_BIN, "-m", model_path, "-t", "8", "-f", str(wav_path), "-sns"]
    if language and language.lower() != "auto":
        cmd.extend(["-l", language])
    if prompt_bias:
        cmd.extend(["--prompt", prompt_bias])

    start = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return "[TIMEOUT]", time.time() - start
    except FileNotFoundError:
        return f"[ERROR: no encontré el binario {WHISPER_BIN}. ¿Compilaste whisper.cpp?]", 0.0
    elapsed = time.time() - start

    lines = result.stdout.strip().split("\n")
    if lines and lines[-1].strip():
        last = lines[-1].strip()
        text = last.split("]")[1].strip() if "]" in last else last
    else:
        text = ""
    return text, elapsed


def resolve_model_path(name):
    """Acepta tanto un nombre corto ('tiny', 'base', 'small') como una
    ruta directa a un .bin."""
    if name.endswith(".bin"):
        return Path(name)
    candidate = Path(MODELS_DIR) / f"ggml-{name}.bin"
    return candidate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS,
                         help="Nombres cortos (tiny/base/small/medium) o rutas .bin completas")
    parser.add_argument("--audio-dir", default="./benchmark_audio",
                         help="Carpeta con los .wav de prueba y reference.json")
    parser.add_argument("--language", default="es")
    parser.add_argument("--prompt", default="BMO, Cali, Valle del Cauca, Universidad del Valle, Raspberry Pi, iPod, WhatsApp.",
                         help="stt_initial_prompt, igual al de tu config.json")
    parser.add_argument("--out", default="benchmark_whisper_results.csv")
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir)
    ref_path = audio_dir / "reference.json"
    if not audio_dir.exists() or not ref_path.exists():
        print(f"[ERROR] No encontré '{ref_path}'.")
        print("Prepara la carpeta primero — instrucciones al inicio de este script (docstring).")
        return

    reference = json.loads(ref_path.read_text(encoding="utf-8"))
    wav_files = sorted(f for f in audio_dir.glob("*.wav") if f.name in reference)
    if not wav_files:
        print(f"[ERROR] No hay .wav en {audio_dir} que coincidan con reference.json")
        return

    model_paths = []
    for name in args.models:
        p = resolve_model_path(name)
        if not p.exists():
            print(f"[AVISO] No encontré el modelo '{name}' en {p} — sáltatelo o descárgalo con "
                  f"'bash ./whisper.cpp/models/download-ggml-model.sh {name}'. Se omite.")
            continue
        model_paths.append((name, p))

    if not model_paths:
        print("[ERROR] Ningún modelo disponible para probar.")
        return

    print(f"Modelos: {[n for n, _ in model_paths]}")
    print(f"Audios de prueba: {len(wav_files)}")
    print("=" * 100)

    rows = []
    summary = {name: {"scores": [], "times": []} for name, _ in model_paths}

    for wav in wav_files:
        expected = reference[wav.name]
        print(f"\n🎤 {wav.name}  (referencia: \"{expected}\")")
        for name, model_path in model_paths:
            text, elapsed = transcribe(str(model_path), wav, args.language, args.prompt)
            score = similarity(expected, text)
            summary[name]["scores"].append(score)
            summary[name]["times"].append(elapsed)
            mark = "✅" if score >= 80 else ("⚠️" if score >= 50 else "❌")
            print(f"  {mark} {name:10s} [{elapsed:5.2f}s] ({score:5.1f}%): \"{text}\"")
            rows.append([wav.name, expected, name, text, score, f"{elapsed:.2f}"])

    print("\n" + "=" * 100)
    print("RESUMEN")
    print("=" * 100)
    for name, stats in summary.items():
        avg_score = sum(stats["scores"]) / len(stats["scores"]) if stats["scores"] else 0
        avg_time = sum(stats["times"]) / len(stats["times"]) if stats["times"] else 0
        print(f"{name:10s}  similitud_prom={avg_score:5.1f}%   tiempo_prom={avg_time:5.2f}s")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["audio", "referencia", "modelo", "transcripcion", "similitud_%", "tiempo_s"])
        writer.writerows(rows)
    print(f"\nResultados detallados guardados en: {args.out}")


if __name__ == "__main__":
    main()
