#!/usr/bin/env python3
# =========================================================================
#  benchmark_tools.py
#
#  Prueba, de forma repetible y serial, qué tan confiable es cada
#  text_model llamando las tools correctas (get_time, battery_status,
#  calculate, search_web, system_shutdown/reboot/suspend, capture_image)
#  vs. contestando en texto plano cuando no corresponde.
#
#  Usa el MISMO SYSTEM_PROMPT, TOOL_FEW_SHOT, TOOLS y OLLAMA_OPTIONS_ROUTE
#  que agent.py (los importa directo del archivo), así que el resultado
#  refleja exactamente lo que pasaría en el agente real, no una versión
#  aparte que se puede desincronizar.
#
#  Uso:
#      python benchmark_tools.py                     # usa MODELS de abajo
#      python benchmark_tools.py qwen2.5:1.5b gemma3:1b
#      python benchmark_tools.py --repeats 5 qwen2.5:1.5b
#
#  Requiere: ollama corriendo (`ollama serve`) y los modelos ya
#  descargados (`ollama pull <modelo>`).
# =========================================================================

import sys
import time
import json
import csv
import argparse
import importlib.util
from pathlib import Path

# --- Importar agent.py como módulo sin correr su GUI -----------------------
# agent.py solo lanza tkinter bajo `if __name__ == "__main__"`, así que
# importarlo ejecuta la carga de config/prompt/tools pero no abre ventana.
AGENT_PATH = Path(__file__).parent / "agent.py"
if not AGENT_PATH.exists():
    print(f"[ERROR] No encontré agent.py junto a este script ({AGENT_PATH}).")
    sys.exit(1)

spec = importlib.util.spec_from_file_location("agent", AGENT_PATH)
agent = importlib.util.module_from_spec(spec)
sys.modules["agent"] = agent
spec.loader.exec_module(agent)

import ollama

# Modelos a comparar si no se pasan por línea de comandos. Ajusta esta
# lista a lo que tengas descargado (`ollama list`).
#
# Instalación de los candidatos nuevos (ver benchmark_models.md si lo
# generaste, o corre estos comandos):
#
#   ollama pull qwen2.5:1.5b          # baseline, ya probado: 15/16 en tu CSV
#   ollama pull qwen3:1.7b            # reintentar con --think off (ver abajo)
#   ollama pull llama3.2:1b
#   ollama pull llama3.2:3b
#   ollama pull granite4:tiny-h       # MoE: 7B total / ~1B activo, tool-calling nativo
#   ollama pull granite4:micro        # 3B denso, alternativa sin MoE
#
#   # Dedicados a function-calling (no están en el library oficial de
#   # Ollama, se importan como GGUF desde Hugging Face; si el pull falla
#   # prueba con el tag de cuantización, ej. ":Q4_K_M"):
#   ollama pull hf.co/katanemo/Arch-Function-3B.gguf
#   ollama pull hf.co/mradermacher/Hammer2.1-1.5b-GGUF:Q4_K_M
#   ollama pull hf.co/mradermacher/Hammer2.1-3b-GGUF:Q4_K_M
#
# OJO con los dos últimos (Arch-Function/Hammer): están afinados para
# function-calling, pero no todos los GGUF de terceros declaran soporte
# de "tools" en su chat template — si ollama.chat(tools=...) no les
# genera tool_calls y solo responden texto plano, es señal de que ese
# GGUF en particular no trae la plantilla correcta, no que el modelo sea
# malo. Este script lo va a marcar igual como FAIL si pasa eso.
MODELS = [
    "qwen2.5:1.5b",
    "qwen3:1.7b",
    "llama3.2:1b",
    "llama3.2:3b",
    "granite4:tiny-h",
    "granite4:micro",
]

# Modelos "razonadores" (qwen3, granite4, deepseek-r1, gpt-oss...) meten
# un bloque de "pensamiento" en texto plano antes de decidir si llaman
# una tool. Para un agente de voz que necesita responder rápido, eso es
# pura latencia y además puede hacer que el modelo "razone" en vez de
# llamar la función. think=False lo apaga. Modelos que no son
# razonadores simplemente ignoran el parámetro.
THINKING_MODEL_HINTS = ("qwen3", "granite4", "deepseek-r1", "gpt-oss")

# --- Casos de prueba ---------------------------------------------------
# (prompt en español, tool esperada o None si debe responder en texto
# plano, argumento clave esperado o None si no aplica / no se valida)
TEST_CASES = [
    ("¿Qué hora es?", "get_time", None),
    ("¿Qué hora tienes?", "get_time", None),
    ("¿Cuánta batería te queda?", "battery_status", None),
    ("¿Cómo está la batería del computador?", "battery_status", None),
    ("¿Cuánto es 45 por 12?", "calculate", None),
    ("Calcula la raíz cuadrada de 1750", "calculate", None),
    ("Busca en internet quién es Baltazar Renfrew", "search_web", "query"),
    ("¿Qué está pasando hoy en las noticias?", "search_web", None),
    ("¿Qué ves ahorita?", "capture_image", None),
    ("Apaga el computador", "system_shutdown", None),
    ("Reinicia el computador", "system_reboot", None),
    ("Pon el computador a dormir", "system_suspend", None),
    ("Hola, ¿cómo estás?", None, None),
    ("Cuéntame un chiste", None, None),
    ("¿Cuál es la capital de Francia?", None, None),
]


def run_case(model_name, prompt):
    messages = [
        {"role": "system", "content": agent.SYSTEM_PROMPT},
        *agent.TOOL_FEW_SHOT,
        {"role": "user", "content": prompt},
    ]

    start = time.time()
    call_kwargs = dict(
        model=model_name,
        messages=messages,
        tools=agent.TOOLS,
        stream=False,
        options=agent.OLLAMA_OPTIONS_ROUTE,
    )
    if any(hint in model_name.lower() for hint in THINKING_MODEL_HINTS):
        call_kwargs["think"] = False

    try:
        try:
            response = ollama.chat(**call_kwargs)
        except TypeError:
            # Cliente ollama-python viejo que no conoce el kwarg "think" —
            # reintenta sin él en vez de fallar el caso completo.
            call_kwargs.pop("think", None)
            response = ollama.chat(**call_kwargs)
    except Exception as e:
        return {"error": str(e), "latency": time.time() - start}

    latency = time.time() - start
    reply = response.get("message", {}) or {}
    tool_calls = reply.get("tool_calls") or []

    if tool_calls:
        fn = (tool_calls[0].get("function") or {})
        return {
            "called_tool": fn.get("name"),
            "args": fn.get("arguments"),
            "text": None,
            "latency": latency,
        }
    else:
        return {
            "called_tool": None,
            "args": None,
            "text": (reply.get("content") or "")[:80],
            "latency": latency,
        }


def grade(expected_tool, got_tool):
    if expected_tool is None:
        return "OK" if got_tool is None else "FAIL"
    return "OK" if got_tool == expected_tool else "FAIL"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="*", help="Modelos a probar (si se omite, usa la lista MODELS del script)")
    parser.add_argument("--repeats", type=int, default=3, help="Repeticiones por caso (default 3, para ver variación con temperature>0)")
    parser.add_argument("--out", default="benchmark_tools_results.csv", help="Archivo CSV de salida")
    args = parser.parse_args()

    models = args.models if args.models else MODELS
    print(f"Modelos a probar: {models}")
    print(f"Repeticiones por caso: {args.repeats}")
    print(f"Casos de prueba: {len(TEST_CASES)}")
    print("=" * 100)

    rows = []
    summary = {m: {"ok": 0, "total": 0, "errors": 0, "latencies": []} for m in models}

    for model_name in models:
        print(f"\n### Modelo: {model_name} ###")
        for prompt, expected_tool, _ in TEST_CASES:
            for rep in range(args.repeats):
                result = run_case(model_name, prompt)
                summary[model_name]["total"] += 1

                if "error" in result:
                    summary[model_name]["errors"] += 1
                    print(f"  [ERROR] '{prompt}' (rep {rep+1}): {result['error']}")
                    rows.append([model_name, prompt, expected_tool, "ERROR", result["error"], "", f"{result['latency']:.2f}"])
                    continue

                got_tool = result["called_tool"]
                verdict = grade(expected_tool, got_tool)
                summary[model_name]["latencies"].append(result["latency"])
                if verdict == "OK":
                    summary[model_name]["ok"] += 1

                extra = result["text"] if got_tool is None else json.dumps(result["args"], ensure_ascii=False)
                mark = "✅" if verdict == "OK" else "❌"
                print(f"  {mark} '{prompt}' (rep {rep+1}) -> esperado={expected_tool or 'texto'} "
                      f"obtenido={got_tool or 'texto'} [{result['latency']:.2f}s]")
                if verdict == "FAIL":
                    print(f"       contenido: {extra}")

                rows.append([model_name, prompt, expected_tool or "", got_tool or "",
                             extra, verdict, f"{result['latency']:.2f}"])

    print("\n" + "=" * 100)
    print("RESUMEN")
    print("=" * 100)
    for model_name, stats in summary.items():
        total = stats["total"]
        ok = stats["ok"]
        errors = stats["errors"]
        pct = (ok / total * 100) if total else 0
        avg_latency = (sum(stats["latencies"]) / len(stats["latencies"])) if stats["latencies"] else 0
        print(f"{model_name:20s}  {ok}/{total} correctos ({pct:.0f}%)  "
              f"errores={errors}  latencia_prom={avg_latency:.2f}s")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["modelo", "prompt", "tool_esperada", "tool_obtenida", "detalle", "veredicto", "latencia_s"])
        writer.writerows(rows)
    print(f"\nResultados detallados guardados en: {args.out}")


if __name__ == "__main__":
    main()
