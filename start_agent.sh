#!/bin/bash
set -euo pipefail

# Resuelve el symlink (ej. ~/.local/bin/bmo) hasta la ruta real del script,
# para que BASE_DIR sea siempre la carpeta del repo, sin importar desde dónde
# se invoque.
SOURCE="${BASH_SOURCE[0]}"
while [ -L "$SOURCE" ]; do
  DIR="$(cd -P "$(dirname "$SOURCE")" && pwd)"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE"
done
BASE_DIR="$(cd -P "$(dirname "$SOURCE")" && pwd)"

cd "$BASE_DIR"
source venv/bin/activate

# whisper-cli necesita saber dónde están sus librerías compartidas
# (libwhisper.so.1, libggml*.so), que viven junto al binario en build/bin.
export LD_LIBRARY_PATH="$BASE_DIR/whisper.cpp/build/bin:${LD_LIBRARY_PATH:-}"

exec python agent.py
