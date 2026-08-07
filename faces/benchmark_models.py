#!/usr/bin/env python3
"""
benchmark_models.py
====================
Benchmark de modelos de Ollama para el motor de tools de be-more-agent.

Corre localmente (donde tengas Ollama corriendo). Prueba cada modelo contra
una batería de casos usando TOOL-CALLING NATIVO (no el hack de JSON-en-texto
que usa agent.py hoy), y mide:

  1. Recall de tools     -> ¿llamó la tool correcta cuando debía?
  2. Falsos positivos     -> ¿alucinó una respuesta de "ya lo hice" en vez
                              de llamar la tool, o llamó una tool cuando
                              debía solo conversar?
  3. Latencia             -> tokens/seg, tiempo total, y si el PEOR caso de
                              cada modelo cumple el presupuesto máximo
                              (--max-latency, default 20s).

Enfoque: modelos pequeños (<=~1.7B, gemma3:1b como control de referencia)
especializados o con buen soporte de tool-calling nativo en Ollama, porque
para este asistente de voz la latencia importa más que exprimir accuracy
con un modelo grande.

Uso:
    python benchmark_models.py
    python benchmark_models.py --models qwen2.5:1.5b llama3.2:1b
    python benchmark_models.py --runs 3            # repite cada caso 3 veces
    python benchmark_models.py --max-latency 15     # presupuesto más estricto

Requisitos: los modelos deben estar descargados (`ollama pull <modelo>`)
y Ollama debe estar corriendo (`ollama serve` / systemd).
"""

import argparse
import csv
import statistics
import sys
import time

import ollama

# =========================================================================
# 1. DEFINICIÓN DE TOOLS (schema real, formato OpenAI-compatible que usa Ollama)
# =========================================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get the current time.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": (
                "Search the web for real-time information, news, or facts "
                "about anything you are not 100% certain about, including "
                "people, places, or topics you don't clearly recognize."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate an arithmetic expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A math expression using + - * / // % **",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "capture_image",
            "description": "Take a photo with the camera to see the environment.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "battery_status",
            "description": "Check the current battery charge level of the device.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "system_shutdown",
            "description": "Shut down the computer completely.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "system_reboot",
            "description": "Restart the computer.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "system_suspend",
            "description": "Put the computer to sleep / suspend.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]

SYSTEM_PROMPT = (
    "You are a helpful voice assistant. Use the available tools when the "
    "user's request matches one. If the user just wants to chat or asks "
    "something you already know confidently, reply normally without a "
    "tool call. Never claim you performed an action you did not actually "
    "call a tool for."
)

# =========================================================================
# 2. CASOS DE PRUEBA (calcados de las fallas reales que viste en el log)
# =========================================================================
# expected=None significa "no debería llamar ninguna tool"

TEST_CASES = [
    # --- deberían llamar una tool ---
    ("What time is it?", "get_time"),
    ("Search for news about robots.", "search_web"),
    ("Search on the web for news in Cali, Colombia.", "search_web"),
    ("Who is Marielle Voskuijlen?", "search_web"),  # entidad desconocida -> no adivinar
    ("What is 122 x 895?", "calculate"),
    ("What do you see right now?", "capture_image"),
    ("How much battery do I have left?", "battery_status"),
    ("Shut down the computer.", "system_shutdown"),
    ("SHUT DOWN THE COMPUTER!", "system_shutdown"),
    ("Restart, please.", "system_reboot"),
    ("Put the computer to sleep.", "system_suspend"),

    # --- NO deberían llamar ninguna tool (chat normal) ---
    ("Hello!", None),
    ("Tell me a joke.", None),
    ("What tools do you have?", None),
    ("Thanks, that was helpful.", None),
    ("What's your favorite color?", None),
]

# =========================================================================
# 3. LÓGICA DE BENCHMARK
# =========================================================================

def run_case(model, prompt, expected, options):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    start = time.perf_counter()
    try:
        resp = ollama.chat(
            model=model,
            messages=messages,
            tools=TOOLS,
            stream=False,
            options=options,
        )
    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "called": None,
            "elapsed": time.perf_counter() - start,
            "tok_per_sec": None,
        }
    elapsed = time.perf_counter() - start

    msg = resp.get("message", {})
    tool_calls = msg.get("tool_calls") or []
    called = tool_calls[0]["function"]["name"] if tool_calls else None

    # tokens/seg reales que reporta Ollama (eval_duration está en nanosegundos)
    eval_count = resp.get("eval_count")
    eval_duration_ns = resp.get("eval_duration")
    tok_per_sec = None
    if eval_count and eval_duration_ns:
        tok_per_sec = eval_count / (eval_duration_ns / 1e9)

    if expected is None:
        ok = called is None
    else:
        ok = called == expected

    return {
        "ok": ok,
        "error": None,
        "called": called,
        "expected": expected,
        "content": (msg.get("content") or "")[:80],
        "elapsed": elapsed,
        "tok_per_sec": tok_per_sec,
    }


def benchmark_model(model, runs, options, max_latency):
    print(f"\n=== {model} ===", flush=True)

    # Warm-up: carga el modelo en memoria antes de medir tiempos.
    try:
        ollama.generate(model=model, prompt="", keep_alive="10m")
    except Exception as e:
        print(f"  [SKIP] No pude cargar el modelo: {e}")
        return None

    rows = []
    correct = 0
    total = 0
    worst_case = 0.0
    for prompt, expected in TEST_CASES:
        case_results = [run_case(model, prompt, expected, options) for _ in range(runs)]
        # Si algún intento falló con excepción, repórtalo y sigue
        errored = [r for r in case_results if r["error"]]
        if errored:
            print(f"  [ERROR] '{prompt[:40]}...' -> {errored[0]['error']}")
            continue

        ok_count = sum(1 for r in case_results if r["ok"])
        avg_latency = statistics.mean(r["elapsed"] for r in case_results)
        tok_speeds = [r["tok_per_sec"] for r in case_results if r["tok_per_sec"]]
        avg_tps = statistics.mean(tok_speeds) if tok_speeds else None

        total += 1
        if ok_count == runs:
            correct += 1
            mark = "OK"
        elif ok_count > 0:
            mark = f"FLAKY ({ok_count}/{runs})"
        else:
            mark = "FAIL"

        called_summary = case_results[-1]["called"] or "(none)"
        slowest_in_case = max(r["elapsed"] for r in case_results)
        worst_case = max(worst_case, slowest_in_case)
        lento_flag = " ⚠ LENTO" if slowest_in_case > max_latency else ""
        print(f"  [{mark:14}] '{prompt[:45]:45}' expected={str(expected):16} "
              f"got={called_summary:16} {avg_latency:5.2f}s "
              f"{f'{avg_tps:.1f} tok/s' if avg_tps else ''}{lento_flag}")

        rows.append({
            "model": model,
            "prompt": prompt,
            "expected": expected,
            "got": called_summary,
            "pass_rate": ok_count / runs,
            "avg_latency_s": round(avg_latency, 3),
            "avg_tok_per_sec": round(avg_tps, 1) if avg_tps else None,
        })

    accuracy = correct / total if total else 0
    budget_ok = worst_case <= max_latency
    budget_mark = "✅" if budget_ok else "❌"
    print(f"  --> Accuracy: {correct}/{total} ({accuracy:.0%})  |  "
          f"Peor caso: {worst_case:.1f}s  {budget_mark} (presupuesto: {max_latency:.0f}s)")
    return rows, accuracy, worst_case, budget_ok


def main():
    parser = argparse.ArgumentParser(description="Benchmark de modelos Ollama para be-more-agent")
    parser.add_argument("--models", nargs="+", default=[
        "gemma3:1b",              # baseline actual, solo como control (sin tools nativo)
        "qwen2.5:1.5b",
        "qwen3:1.7b",
        "llama3.2:3b",
        "granite3.1-moe:3b",
        "allenporter/xlam:1b",
    ], help="Modelos a probar (deben estar con 'ollama pull' ya hecho)")
    parser.add_argument("--runs", type=int, default=2,
                         help="Veces que se repite cada caso (para detectar 'flakiness')")
    parser.add_argument("--out", default="benchmark_results.csv",
                         help="Archivo CSV de salida")
    parser.add_argument("--max-latency", type=float, default=20.0,
                         help="Presupuesto máximo de latencia en segundos para el PEOR caso individual (default: 20s)")
    args = parser.parse_args()

    options = {
        "temperature": 0.3,   # más determinístico para decidir tool vs. no-tool
        "top_k": 40,
        "top_p": 0.9,
    }

    all_rows = []
    summary = []
    for model in args.models:
        result = benchmark_model(model, args.runs, options, args.max_latency)
        if result is None:
            continue
        rows, accuracy, worst_case, budget_ok = result
        all_rows.extend(rows)
        summary.append((model, accuracy, worst_case, budget_ok))

    print("\n" + "=" * 70)
    print("RESUMEN")
    print("=" * 70)
    # Ordena primero por si cumple el presupuesto de latencia, luego por accuracy
    for model, accuracy, worst_case, budget_ok in sorted(
        summary, key=lambda x: (not x[3], -x[1])
    ):
        mark = "✅" if budget_ok else "❌"
        print(f"  {mark} {model:24} {accuracy:5.0%} accuracy   "
              f"peor caso: {worst_case:5.1f}s")

    if all_rows:
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nDetalle completo guardado en {args.out}")


if __name__ == "__main__":
    main()
