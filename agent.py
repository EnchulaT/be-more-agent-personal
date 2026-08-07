# =========================================================================
#  Be More Agent 🤖
#  A Local, Offline-First AI Agent for Raspberry Pi
#
#  Copyright (c) 2026 brenpoly
#  Licensed under the MIT License
#  Source: https://github.com/brenpoly/be-more-agent
#
#  DISCLAIMER:
#  This software is provided "as is", without warranty of any kind.
#  This project is a generic framework and includes no copyrighted assets.
# =========================================================================

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading
import time
import json
import os
import subprocess
import random
import re
import unicodedata
import sys
import select
import traceback
import atexit
import contextlib
import datetime
import warnings
import wave
import struct 

# Suppress harmless library warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="ddgs")

# Core dependencies
import sounddevice as sd
import numpy as np
import scipy.signal 
try:
    # Solo se usa para hablar con whisper-server (STT persistente, ver
    # WHISPER_SERVER_BIN más abajo). Si no está instalado, el agente cae de
    # vuelta a whisper-cli por turno automáticamente -- no es dependencia
    # dura.
    import requests
except ImportError:
    requests = None
import cv2  # Captura de cámara genérica (V4L2) — reemplaza rpicam-still,
            # que es exclusivo de Raspberry Pi OS / libcamera.

# --- AI ENGINES ---
import openwakeword
from openwakeword.model import Model
import ollama 

# --- WEB SEARCH (Using your working import) ---
from ddgs import DDGS 

# =========================================================================
# 1. CONFIGURATION & CONSTANTS
# =========================================================================

CONFIG_FILE = "config.json"
MEMORY_FILE = "memory.json"
BMO_IMAGE_FILE = "current_image.jpg"
WAKE_WORD_MODEL = "./wakeword.onnx"
WAKE_WORD_THRESHOLD = 0.5

# HARDWARE SETTINGS
INPUT_DEVICE_NAME = None

DEFAULT_CONFIG = {
    "text_model": "gemma3:1b",
    "vision_model": "moondream",
    "voice_model": "piper/en_GB-semaine-medium.onnx",
    "chat_memory": True,
    # Pi-era default assumed a dedicated small LCD run in kiosk mode. On a
    # normal desktop (with a WM you also use for everything else) forcing
    # fullscreen just leaves a huge empty/grey area around the 800x480
    # face, since the canvas itself is never resized to match. Default is
    # a normal, fixed-size window; set to true if you ever drive this from
    # a dedicated small screen again.
    "fullscreen": False,
    "camera_rotation": 0,
    "system_prompt_extras": "",
    "input_device": None,
    "input_sample_rate": None,
    # Speech-to-text (whisper.cpp). "stt_model" must be a MULTILINGUAL ggml
    # model (no ".en" in the filename) for "stt_language" to have any
    # effect — an ".en" model is English-only regardless of -l. Default
    # here is Spanish since that's the daily-use language; set to "auto"
    # to let whisper detect the language per-utterance (slightly slower,
    # useful if you regularly mix English/Spanish in the same session).
    "stt_model": "./whisper.cpp/models/ggml-base.bin",
    "stt_language": "es",
    # Biases whisper's decoder toward this vocabulary (proper nouns,
    # acronyms, common commands) without slowing anything down — this is
    # whisper.cpp's native "--prompt" flag, not an LLM call. Tune it to
    # whatever names/words you say most and whisper keeps mangling.
    "stt_initial_prompt": "BMO, Cali, Valle del Cauca, Universidad del Valle, Raspberry Pi, iPod, WhatsApp.",
    # How many silent RMS chunks (see silence_duration/chunk math below) at
    # the very start of a recording are spent measuring the room's ambient
    # noise floor, instead of comparing straight against "silence_threshold".
    # A fan, traffic, or a noisier mic can sit above a fixed 0.006 RMS at
    # rest — when that happens "silence" never triggers and recording runs
    # all the way to max_record_time, which is the most likely reason BMO
    # feels like it waits too long after you stop talking. 0.25s matches
    # the old fixed warmup, just used on purpose now instead of discarded.
    "noise_calibration_seconds": 0.25
}

# --- HARDWARE-ADAPTIVE THREAD COUNTS ---
# The upstream project hardcoded num_thread=4 and "-t 4" everywhere, tuned
# for a 4-core Raspberry Pi with nothing else running. On a 12-core/16-thread
# machine that leaves most of the CPU idle during inference, so derive both
# from os.cpu_count() instead. OLLAMA_NUM_THREAD leaves a few threads free
# for the GUI thread, the audio callback, openWakeWord inference, and the
# Piper/whisper subprocesses that can run concurrently with an Ollama call.
_CPU_COUNT = os.cpu_count() or 4
OLLAMA_NUM_THREAD = max(4, _CPU_COUNT - 4)
WHISPER_NUM_THREAD = max(4, min(8, _CPU_COUNT))

# --- WHISPER SERVER (STT persistente) ---
# transcribe_audio() lanzaba whisper-cli como subprocess NUEVO en cada turno,
# lo que recarga el modelo ggml completo desde disco cada vez. Con
# ggml-base.bin (~140MB) ese costo ya se sentía; con ggml-large-v3-turbo.bin
# (~1.5GB, el modelo activo en config.json desde el cambio de ago-2026) es
# casi seguro el mayor cuello de botella de todo el pipeline -- se paga el
# reload completo en CADA turno, no solo una vez.
#
# whisper.cpp trae un servidor HTTP (examples/server) que carga el modelo
# una sola vez y responde por /inference sin recargar -- mismo principio que
# ya usa Ollama vía keep_alive=-1 más abajo. warm_up_logic() lo levanta al
# arrancar (dentro del estado "Calentando el cerebro", antes de la primera
# frase) y transcribe_audio() le pega por HTTP en vez de spawnear whisper-cli.
#
# Requiere compilar un target extra que whisper-cli ya no incluye por
# defecto en todos los setups de CMake:
#     cmake --build whisper.cpp/build --config Release -j --target whisper-server
# Si el binario no existe (o `requests` no está instalado), el agente NO se
# rompe: se degrada en silencio a whisper-cli por turno, igual que antes de
# este cambio.
WHISPER_SERVER_BIN = "./whisper.cpp/build/bin/whisper-server"
WHISPER_SERVER_PUBLIC_DIR = "./whisper.cpp/examples/server/public"
WHISPER_SERVER_HOST = "127.0.0.1"
WHISPER_SERVER_PORT = 8178
# large-v3-turbo por Vulkan en la Iris Xe puede tardar bastante en el primer
# load (compilación de shaders incluida) -- generoso a propósito porque solo
# se paga una vez, al arrancar, nunca por turno.
WHISPER_SERVER_STARTUP_TIMEOUT = 45.0

# LLM SETTINGS
# Two option sets on purpose: the tool-decision call ("should I call a tool,
# and with what arguments?") wants low temperature — this is exactly what
# the benchmark validated (0.3) and what production was NOT using before
# (it was running at 0.7, i.e. more randomness than what got tested). The
# chat/summary options stay warmer for natural-sounding spoken replies.
OLLAMA_OPTIONS_ROUTE = {
    'keep_alive': '-1',
    'num_thread': OLLAMA_NUM_THREAD,
    'temperature': 0.3,
    'top_k': 40,
    'top_p': 0.9
}

OLLAMA_OPTIONS_CHAT = {
    'keep_alive': '-1',
    'num_thread': OLLAMA_NUM_THREAD,
    'temperature': 0.7,
    'top_k': 40,
    'top_p': 0.9
}

# Kept as an alias for any code path that still refers to the old name
# (e.g. vision calls, which don't use tools and aren't part of the
# tool-decision benchmark).
OLLAMA_OPTIONS = OLLAMA_OPTIONS_CHAT

def load_config():
    config = DEFAULT_CONFIG.copy()
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                user_config = json.load(f)
                config.update(user_config)
        except Exception as e:
            # This used to print one line that scrolled off screen under
            # the rest of the startup log — easy to miss the fact that
            # your real config.json (including text_model) was silently
            # ignored in favor of DEFAULT_CONFIG. Made it impossible to miss.
            print("=" * 70, flush=True)
            print(f"[CONFIG ERROR] Could not parse {CONFIG_FILE}: {e}", flush=True)
            print(f"[CONFIG ERROR] Falling back to DEFAULT_CONFIG — this "
                  f"means text_model='{DEFAULT_CONFIG['text_model']}', NOT "
                  f"whatever you set in {CONFIG_FILE}. Fix the JSON syntax "
                  f"(common culprits: stray characters, missing/extra "
                  f"commas) and restart.", flush=True)
            print("=" * 70, flush=True)
    return config

CURRENT_CONFIG = load_config()
TEXT_MODEL = CURRENT_CONFIG["text_model"]
VISION_MODEL = CURRENT_CONFIG["vision_model"]

def check_tool_support_at_startup(model_name):
    """Best-effort early warning if TEXT_MODEL doesn't support native
    tool-calling, instead of only finding out after the first spoken
    request fails. Uses `ollama show`'s "capabilities" field (present on
    recent Ollama versions); silently skips the check if that field isn't
    available or the server isn't reachable yet — the runtime fallback in
    chat_and_respond() still catches it either way."""
    try:
        info = ollama.show(model_name)
        capabilities = info.get("capabilities") if isinstance(info, dict) else getattr(info, "capabilities", None)
        if capabilities is not None and "tools" not in capabilities:
            print("=" * 70, flush=True)
            print(f"[STARTUP WARNING] text_model '{model_name}' does not report "
                  f"'tools' in its capabilities — it likely does NOT support "
                  f"native tool-calling. time/search/battery/system tools will "
                  f"silently fall back to plain chat. Consider switching "
                  f"text_model to a model with confirmed tool support (e.g. "
                  f"qwen2.5:1.5b).", flush=True)
            print("=" * 70, flush=True)
    except Exception:
        # Older ollama client/server without "capabilities", model not
        # pulled yet, server not up yet, etc. — not fatal, just skip.
        pass

check_tool_support_at_startup(TEXT_MODEL)

def resolve_input_device(config):
    requested = config.get("input_device")
    if requested in (None, "", "default"):
        return None

    try:
        devices = sd.query_devices()
    except Exception as e:
        print(f"[AUDIO] Device query failed: {e}", flush=True)
        return None

    if isinstance(requested, int) or (isinstance(requested, str) and requested.isdigit()):
        index = int(requested)
        if 0 <= index < len(devices):
            return index
        print(f"[AUDIO] Input device index not found: {index}", flush=True)
        return None

    requested_lower = str(requested).lower()
    for idx, dev in enumerate(devices):
        if dev.get("max_input_channels", 0) > 0 and requested_lower in dev.get("name", "").lower():
            return idx

    print(f"[AUDIO] Input device name not found: {requested}", flush=True)
    return None

INPUT_DEVICE_NAME = resolve_input_device(CURRENT_CONFIG)
if INPUT_DEVICE_NAME is not None:
    try:
        device_info = sd.query_devices(INPUT_DEVICE_NAME)
        print(f"[AUDIO] Using input device: {device_info.get('name', INPUT_DEVICE_NAME)}", flush=True)
    except Exception:
        print(f"[AUDIO] Using input device index: {INPUT_DEVICE_NAME}", flush=True)

def choose_input_samplerate(device, preferred=None):
    candidates = []
    if preferred:
        candidates.append(preferred)
    try:
        device_info = sd.query_devices(device)
        if "default_samplerate" in device_info:
            candidates.append(int(device_info["default_samplerate"]))
    except Exception as e:
        print(f"[AUDIO DEBUG] Query failed: {e}", flush=True)
        pass

    candidates.extend([48000, 44100, 32000, 16000])
    seen = set()
    for rate in candidates:
        if not rate or rate in seen:
            continue
        seen.add(rate)
        try:
            sd.check_input_settings(device=device, samplerate=rate, channels=1, dtype="int16")
            return rate
        except Exception:
            continue

    return int(candidates[0]) if candidates else 44100

class BotStates:
    IDLE = "idle"             
    LISTENING = "listening"   
    THINKING = "thinking"     
    SPEAKING = "speaking"     
    ERROR = "error"           
    CAPTURING = "capturing" 
    WARMUP = "warmup"       

# --- SYSTEM PROMPT ---
# NOTE: as of the native tool-calling migration, this prompt no longer asks
# the model to hand-write JSON. Ollama's /api/chat exposes a real `tools`
# parameter (see TOOLS below); the model either returns a structured
# `tool_calls` entry or plain text. Free-text JSON examples used to actively
# confuse small models once real tool-calling was enabled, so they are gone.
BASE_SYSTEM_PROMPT = """You are a helpful robot assistant running on a personal computer.
Personality: Cute, helpful, robot.
Style: Short sentences. Enthusiastic.

INSTRUCTIONS:
- If the user asks for a physical action (time, search, photo, math, power
  control, battery), call the matching tool. Do not describe the action in
  text and do not say you already did it — actually call the tool.
- If the user just wants to chat, reply with NORMAL TEXT and do not call any
  tool.
- Your only real tools are: get_time, search_web, capture_image, calculate,
  system_shutdown, system_reboot, system_suspend, battery_status.
  If asked what tools/abilities you have, list exactly these in plain text.
  Never say you have no tools, and never invent tools you don't have.

WHEN TO USE search_web (read this carefully, this is the most important rule):
- You do NOT have real-time knowledge and your training data can be old or
  incomplete. Never guess or invent facts about something you are not
  completely sure about.
- If the user asks about a specific person, place, product, game, or term you
  do not clearly recognize, you MUST call search_web instead of answering
  from memory. This applies to ANY person's name that is not extremely
  famous — if you would have to guess who they are, search instead. Guessing
  / making up a plausible-sounding answer is strictly forbidden.
- ANY request to search, look up, or find news/info about ANY topic or place
  (including "search the news in <city>") must call search_web.
- Any question about current events, prices, scores, schedules, or "what is
  happening now" must call search_web.
- If you are not 100% sure, prefer search_web over answering directly. It is
  always better to search than to make something up.

Example of the search rule: if asked "Who is Baltazar Renfrew?" and that name
means nothing to you, call search_web with that name as the query instead of
inventing a biography. If the search comes back empty, say plainly that you
couldn't find anything about them — never fill the gap from imagination.

OTHER RULES:
- For math questions, convert them into a plain arithmetic expression
  (numbers and + - * / // % ** only) and call calculate with that
  expression. Never compute the answer yourself instead of calling the tool.
- If asked what you currently see, call capture_image.
- system_shutdown, system_reboot, and system_suspend are DESTRUCTIVE actions.
  Only call them when the user clearly and explicitly asks to shut down,
  restart, or suspend/sleep the machine. Never trigger them from an
  ambiguous or unrelated request. The system will always ask the user to
  confirm before actually doing it, so just call the tool normally when the
  request is clear — you do not need to ask for confirmation yourself.
- battery_status is safe and read-only; call it whenever the user asks about
  battery, charge, or power level.
"""

SYSTEM_PROMPT = BASE_SYSTEM_PROMPT + "\n\n" + CURRENT_CONFIG.get("system_prompt_extras", "")

# --- NATIVE TOOL SCHEMA (Ollama /api/chat "tools" parameter) ---
# Passed directly to ollama.chat(tools=TOOLS). The model returns a structured
# `tool_calls` list instead of free-text JSON, which is what the old
# chunk-by-chunk '{"' detection in chat_and_respond used to try to fake.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get the current time on this device.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": (
                "Search the web for information you don't clearly know, "
                "including any person, place, product, game, or term you "
                "don't recognize with certainty, and any current "
                "events/news/prices/scores question. Use this instead of "
                "guessing whenever you are not 100% sure."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query, in a few plain keywords.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "capture_image",
            "description": "Take a photo with the camera to see what is currently in front of you.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": (
                "Evaluate a plain arithmetic expression. Convert the "
                "user's question into numbers and + - * / // % ** only, "
                "never compute the answer yourself."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Arithmetic expression, e.g. '542*3'.",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "battery_status",
            "description": "Check the remaining battery charge. Safe and read-only.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "system_shutdown",
            "description": (
                "Shut down the computer. DESTRUCTIVE. Only call when the "
                "user clearly and explicitly asks to shut down / turn off "
                "the machine. The system will ask the user to confirm "
                "before this actually runs."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "system_reboot",
            "description": (
                "Restart the computer. DESTRUCTIVE. Only call when the "
                "user clearly and explicitly asks to restart/reboot the "
                "machine. The system will ask the user to confirm before "
                "this actually runs."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "system_suspend",
            "description": (
                "Suspend/sleep the computer. DESTRUCTIVE. Only call when "
                "the user clearly and explicitly asks to suspend/sleep the "
                "machine. The system will ask the user to confirm before "
                "this actually runs."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]

# --- STATIC FEW-SHOT (not conversation history) ---
# In real usage, qwen2.5:1.5b consistently did NOT call zero-argument tools
# (get_time, battery_status) — it answered in plain hallucinated text
# instead — while search_web (which takes a "query" argument) fired
# correctly some of the time. Prose instructions in the system prompt
# weren't enough to fix this; a concrete demonstrated tool_call per shape
# (empty args vs. one argument) is what the benchmark showed actually works
# for a model this size. This list is FIXED and prepended to every turn's
# `messages` — it does not grow and has nothing to do with session_memory,
# so it doesn't break the stateless design (see chat_and_respond).
#
# IMPORTANT: each demo closes the loop — user -> assistant(tool_call) ->
# tool(result) -> assistant(final reply) — instead of leaving the tool_call
# dangling with no result/reply. A previous version of this list stopped
# right after the last assistant tool_call, with the real user turn
# following immediately after an unresolved call. That malformed pattern
# (a tool_call with no matching "tool" result the model has ever seen
# completed) is almost certainly why search_web started getting a blanket
# "I'm sorry, I can't assist with that" on every single request — the
# model was pattern-matching onto a broken, half-finished exchange instead
# of a real one.
TOOL_FEW_SHOT = [
    {"role": "user", "content": "¿Qué hora es?"},
    {"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": "get_time", "arguments": {}}}
    ]},
    {"role": "tool", "content": "TIME_RESULT::Son las 9:52 de la mañana."},

    {"role": "user", "content": "¿Cuánta batería te queda?"},
    {"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": "battery_status", "arguments": {}}}
    ]},
    {"role": "tool", "content": "BATTERY_RESULT::La batería está al 47 por ciento, actualmente descargando."},

    {"role": "user", "content": "Busca en internet quién es Baltazar Renfrew"},
    {"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": "search_web", "arguments": {"query": "Baltazar Renfrew"}}}
    ]},
    {"role": "tool", "content": "No se encontraron resultados para Baltazar Renfrew."},
    {"role": "assistant", "content": "Busqué, pero no encontré nada sobre Baltazar Renfrew."},
]

# --- MODEL-SPECIFIC "THINKING" OVERRIDE ---
# benchmark_models.py detectó que algunos modelos de Ollama (familia qwen3,
# granite3.1-moe, etc.) traen "thinking" activado por defecto y, si no se
# fuerza think=False explícitamente, la latencia se dispara (se llegó a
# medir hasta 68s en un solo turno). Esa lógica (get_model_config) vivía
# SOLO en el harness de benchmark y nunca se portó al agente real: ninguna
# de las llamadas a ollama.chat() de más abajo pasaba `think=`. Si algún
# día cambias text_model en config.json a un modelo de una de estas
# familias, este bloque es lo que evita repetir el mismo blowup en
# producción. Agrega el substring aquí si detectas el mismo síntoma
# (primer token muy lento, sin contenido de razonamiento visible) en un
# modelo nuevo.
THINKING_MODEL_HINTS = ("qwen3", "granite3.1-moe")


def get_think_override(model_name):
    """None = no tocar `think` (dejar que el modelo/servidor decida).
    False = forzar think=False porque este modelo lo necesita."""
    lower = (model_name or "").lower()
    if any(hint in lower for hint in THINKING_MODEL_HINTS):
        return False
    return None


def ollama_chat_safe(**kwargs):
    """Wrapper delgado sobre ollama.chat() que agrega `think=` solo cuando
    el modelo activo lo necesita (ver THINKING_MODEL_HINTS), y se degrada
    con gracia si el paquete `ollama` instalado es demasiado viejo para
    aceptar ese kwarg (TypeError en vez de ignorarlo silenciosamente) —
    mismo fallback que ya se validó en benchmark_models.py."""
    think = get_think_override(kwargs.get("model", ""))
    if think is None:
        return ollama.chat(**kwargs)
    kwargs["think"] = think
    try:
        return ollama.chat(**kwargs)
    except TypeError:
        kwargs.pop("think", None)
        return ollama.chat(**kwargs)


@contextlib.contextmanager
def step_timer(label):
    """Imprime la hora exacta de inicio y, al salir, cuánto tardó ese paso
    del pipeline (grabación, transcripción, LLM, tool call, TTS...). Un solo
    helper reusado en todos los pasos para que el formato sea consistente y
    fácil de grep-ear en los logs (`grep '\\[TIMING\\]'`)."""
    start = time.time()
    print(f"[TIMING] {label}: inicio {datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]}", flush=True)
    try:
        yield
    finally:
        print(f"[TIMING] {label}: {time.time() - start:.2f}s", flush=True)


# Sound Directories
greeting_sounds_dir = "sounds/greeting_sounds"
ack_sounds_dir = "sounds/ack_sounds"
thinking_sounds_dir = "sounds/thinking_sounds"
error_sounds_dir = "sounds/error_sounds"

# =========================================================================
# 2. GUI CLASS
# =========================================================================

class BotGUI:
    BG_WIDTH, BG_HEIGHT = 800, 480 
    OVERLAY_WIDTH, OVERLAY_HEIGHT = 400, 300 

    def __init__(self, master):
        self.master = master
        master.title("Pi Assistant")
        self.fullscreen_enabled = bool(CURRENT_CONFIG.get("fullscreen", False))
        if self.fullscreen_enabled:
            master.attributes('-fullscreen', True)
        else:
            # Windowed at the assets' native size instead of fullscreen —
            # see the "fullscreen" comment in DEFAULT_CONFIG. Plays nicely
            # with a tiling WM: it's just one more normal-sized window.
            master.geometry(f"{self.BG_WIDTH}x{self.BG_HEIGHT}")
            master.resizable(False, False)
        master.bind('<Escape>', self.exit_fullscreen)
        
        # Inputs
        master.bind('<Return>', self.handle_ptt_toggle)
        master.bind('<space>', self.handle_speaking_interrupt)
        atexit.register(self.safe_exit)
        
        # State
        self.current_state = BotStates.WARMUP
        self.current_volume = 0 
        self.animations = {}
        self.current_frame_index = 0
        self.current_overlay_image = None
        
        self.permanent_memory = self.load_chat_history()
        self.session_memory = []
        self.thinking_sound_active = threading.Event()
        
        self.last_ptt_time = 0 
        self.ptt_event = threading.Event()       
        self.recording_active = threading.Event() 
        self.interrupted = threading.Event() 
        
        self.tts_queue = []          
        self.tts_queue_lock = threading.Lock() 
        self.tts_thread = None       
        self.tts_active = threading.Event()
        self.current_audio_process = None 
        self.exiting = False

        # whisper-server persistente (ver WHISPER_SERVER_BIN). None/False
        # hasta que warm_up_logic() lo levante con éxito; transcribe_audio()
        # cae a whisper-cli por turno mientras tanto.
        self.whisper_server_proc = None
        self.whisper_server_ready = False

        # Cached index of the working /dev/video* node for capture_image()
        # (see _grab_camera_frame). None means "not probed yet".
        self._camera_index = None

        # Pending confirmation for destructive system actions
        # (shutdown / reboot / suspend). Set by chat_and_respond() when the
        # LLM requests one of these, cleared on confirm/deny/timeout.
        self.pending_confirmation = None
        self.pending_confirmation_expiry = 0.0
        self.CONFIRMATION_WINDOW_SECONDS = 15

        # Whether TEXT_MODEL actually supports Ollama's native tools=
        # parameter. Starts True (optimistic); chat_and_respond flips it to
        # False the first time Ollama rejects the call with "does not
        # support tools", so a misconfigured/incompatible model degrades to
        # plain chat instead of throwing "¡Se colgó el cerebro!" on every turn.
        self.tools_supported = True

        # --- WAKE WORD INITIALIZATION ---
        print("[INIT] Loading Wake Word...", flush=True)
        self.oww_model = None
        if os.path.exists(WAKE_WORD_MODEL):
            try:
                self.oww_model = Model(wakeword_model_paths=[WAKE_WORD_MODEL])
                print("[INIT] Wake Word Loaded.", flush=True)
            except TypeError:
                try:
                    self.oww_model = Model(wakeword_models=[WAKE_WORD_MODEL])
                    print("[INIT] Wake Word Loaded (New API).", flush=True)
                except Exception as e:
                    print(f"[CRITICAL] Failed to load model: {e}")
            except Exception as e:
                print(f"[CRITICAL] Failed to load model: {e}")
        else:
            print(f"[CRITICAL] Model not found: {WAKE_WORD_MODEL}")

        # GUI Setup
        self.background_label = tk.Label(master)
        self.background_label.place(x=0, y=0, width=self.BG_WIDTH, height=self.BG_HEIGHT)
        self.background_label.bind('<Button-1>', self.toggle_hud_visibility) 
        
        self.overlay_label = tk.Label(master, bg='black')
        self.overlay_label.bind('<Button-1>', self.toggle_hud_visibility)
        
        self.response_text = tk.Text(master, height=6, width=60, wrap=tk.WORD, 
                                     state=tk.DISABLED, bg="#ffffff", fg="#000000", font=('Arial', 12)) 
        
        self.status_var = tk.StringVar(value="Initializing...")
        self.status_label = ttk.Label(master, textvariable=self.status_var, background="#2e2e2e", foreground="white")
        
        self.exit_button = ttk.Button(master, text="Exit & Save", command=self.safe_exit)

        self.load_animations()
        self.update_animation() 
        
        threading.Thread(target=self.safe_main_execution, daemon=True).start()

    # --- HELPERS ---

    def _safe_eval_math(self, expr):
        """Evalúa una expresión aritmética simple sin usar eval() crudo.
        Solo permite números y +, -, *, /, //, %, **, paréntesis y signo unario."""
        import ast, operator

        ops = {
            ast.Add: operator.add, ast.Sub: operator.sub,
            ast.Mult: operator.mul, ast.Div: operator.truediv,
            ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
            ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos,
        }

        def _eval(node):
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return node.value
            if isinstance(node, ast.BinOp) and type(node.op) in ops:
                return ops[type(node.op)](_eval(node.left), _eval(node.right))
            if isinstance(node, ast.UnaryOp) and type(node.op) in ops:
                return ops[type(node.op)](_eval(node.operand))
            raise ValueError("Unsupported expression")

        if not expr:
            raise ValueError("Empty expression")
        tree = ast.parse(str(expr), mode="eval")
        return _eval(tree.body)

    def safe_exit(self):
        if self.exiting:
            return
        self.exiting = True
        print("\n--- SHUTDOWN SEQUENCE ---", flush=True)

        # Deja que el hilo de audio (_listen_loop / record_voice_*) note self.exiting
        # y cierre su propio sd.InputStream desde dentro. Forzar el cierre desde este
        # hilo mientras el otro está bloqueado en stream.read() causa un segfault
        # (condición de carrera a nivel de PortAudio/C). El sd.stop() de abajo queda
        # solo como red de seguridad.
        time.sleep(0.2)

        if self.current_audio_process:
            try:
                self.current_audio_process.terminate()
                self.current_audio_process.wait(timeout=1)
            except: pass

        self._stop_whisper_server()

        self.recording_active.clear()
        self.thinking_sound_active.clear()
        self.tts_active.clear() 
        
        self.save_chat_history()
        
        try:
            ollama.generate(model=TEXT_MODEL, prompt="", keep_alive=0)
        except: pass
        try:
            sd.stop()
        except: pass

        try:
            self.master.quit()
        except Exception:
            pass
        
    def exit_fullscreen(self, event=None):
        if self.fullscreen_enabled:
            self.master.attributes('-fullscreen', False)
        self.safe_exit()

    def toggle_hud_visibility(self, event=None):
        try:
            if self.response_text.winfo_ismapped():
                self.response_text.place_forget()
                self.status_label.place_forget()
                self.exit_button.place_forget()
            else:
                self.response_text.place(relx=0.5, rely=0.82, anchor=tk.S)
                self.status_label.place(relx=0.5, rely=1.0, anchor=tk.S, relwidth=1)
                self.exit_button.place(x=10, y=10)
        except tk.TclError: pass

    def handle_ptt_toggle(self, event=None):
        current_time = time.time()
        if current_time - self.last_ptt_time < 0.5: 
            return 
        self.last_ptt_time = current_time

        if self.recording_active.is_set():
            print("[PTT] Toggle OFF", flush=True)
            self.recording_active.clear() 
        else:
            if self.current_state == BotStates.IDLE or "Wait" in self.status_var.get():
                print("[PTT] Toggle ON", flush=True)
                self.recording_active.set() 
                self.ptt_event.set()

    def handle_speaking_interrupt(self, event=None):
        if self.current_state == BotStates.SPEAKING or self.current_state == BotStates.THINKING:
            self.interrupted.set()
            self.thinking_sound_active.clear()
            with self.tts_queue_lock:
                self.tts_queue.clear()
            if self.current_audio_process:
                try: self.current_audio_process.terminate()
                except: pass
            self.set_state(BotStates.IDLE, "Interrumpido.")

    def load_animations(self):
        base_path = "faces"
        states = ["idle", "listening", "thinking", "speaking", "error", "capturing", "warmup"] 
        for state in states:
            folder = os.path.join(base_path, state)
            self.animations[state] = []
            if os.path.exists(folder):
                files = sorted([f for f in os.listdir(folder) if f.lower().endswith('.png')])
                for f in files:
                    img = Image.open(os.path.join(folder, f)).resize((self.BG_WIDTH, self.BG_HEIGHT))
                    self.animations[state].append(ImageTk.PhotoImage(img))
            if not self.animations[state]:
                if state in self.animations.get("idle", []):
                     self.animations[state] = self.animations["idle"]
                else:
                    # Blue screen fallback
                    blank = Image.new('RGB', (self.BG_WIDTH, self.BG_HEIGHT), color='#0000FF')
                    self.animations[state].append(ImageTk.PhotoImage(blank))

    def update_animation(self):
        frames = self.animations.get(self.current_state, []) or self.animations.get(BotStates.IDLE, [])
        if not frames:
            self.master.after(500, self.update_animation)
            return

        if self.current_state == BotStates.SPEAKING:
            if len(frames) > 1:
                self.current_frame_index = random.randint(1, len(frames) - 1)
            else:
                self.current_frame_index = 0 
        else:
            self.current_frame_index = (self.current_frame_index + 1) % len(frames)

        self.background_label.config(image=frames[self.current_frame_index])
        
        speed = 50 if self.current_state == BotStates.SPEAKING else 500
        self.master.after(speed, self.update_animation)

    def set_state(self, state, msg="", cam_path=None):
        def _update():
            if msg: print(f"[STATE] {state.upper()}: {msg}", flush=True)
            if self.current_state != state:
                self.current_state = state
                self.current_frame_index = 0
            if msg: self.status_var.set(msg)
            if cam_path and os.path.exists(cam_path) and state in [BotStates.THINKING, BotStates.SPEAKING]:
                try:
                    img = Image.open(cam_path).resize((self.OVERLAY_WIDTH, self.OVERLAY_HEIGHT))
                    self.current_overlay_image = ImageTk.PhotoImage(img)
                    self.overlay_label.config(image=self.current_overlay_image)
                    self.overlay_label.place(x=200, y=90)
                except: pass
            else:
                self.overlay_label.place_forget()
        self.master.after(0, _update)

    def append_to_text(self, text, newline=True):
        def _update():
            self.response_text.config(state=tk.NORMAL)
            if newline: 
                self.response_text.insert(tk.END, text + "\n")
            else: 
                self.response_text.insert(tk.END, text)
            
            self.response_text.see(tk.END)
            self.response_text.config(state=tk.DISABLED)
            
        self.master.after(0, _update)

    def _stream_to_text(self, chunk):
        def update_text_stream():
            self.response_text.config(state=tk.NORMAL)
            self.response_text.insert(tk.END, chunk)
            self.response_text.see(tk.END) 
            self.response_text.config(state=tk.DISABLED)
        self.master.after(0, update_text_stream)

    # =========================================================================
    # 3. ACTION ROUTER
    # =========================================================================
    
    def execute_action_and_get_result(self, tool_name, args=None):
        """Dispatches a tool call coming from Ollama's native tool_calls.

        tool_name: the function name the model called (e.g. "search_web").
        args: dict of arguments the model supplied (already parsed from
              JSON by the ollama client), e.g. {"query": "..."}.
        """
        args = args or {}
        raw_action = (tool_name or "").lower().strip()
        # Native tool-calling gives us the argument under its declared name
        # (query / expression), but keep the old "value" as a fallback in
        # case a model ignores the schema and free-forms a different key.
        value = (
            args.get("query")
            or args.get("expression")
            or args.get("value")
        )

        VALID_TOOLS = {
            "get_time", "search_web", "capture_image", "calculate",
            "system_shutdown", "system_reboot", "system_suspend", "battery_status"
        }
        
        ALIASES = {
            # --- search_web ---
            "google": "search_web", "browser": "search_web", "news": "search_web",
            "search_news": "search_web", "lookup": "search_web", "look_up": "search_web",
            "find": "search_web", "find_info": "search_web", "internet": "search_web",
            "web_search": "search_web", "search": "search_web", "research": "search_web",
            "check_news": "search_web", "whats_new": "search_web", "browse": "search_web",
            "search_online": "search_web", "search_the_web": "search_web",
            "duckduckgo": "search_web", "ddg": "search_web",

            # --- capture_image ---
            "look": "capture_image", "see": "capture_image", "photo": "capture_image",
            "picture": "capture_image", "camera": "capture_image", "watch": "capture_image",
            "take_photo": "capture_image", "snap": "capture_image", "look_around": "capture_image",
            "vision": "capture_image", "check_camera": "capture_image",

            # --- get_time ---
            "check_time": "get_time", "time": "get_time", "clock": "get_time",
            "current_time": "get_time", "whats_the_time": "get_time",

            # --- calculate ---
            "math": "calculate", "compute": "calculate", "calc": "calculate",
            "solve": "calculate", "arithmetic": "calculate", "do_math": "calculate",

            # --- system_shutdown ---
            "shutdown": "system_shutdown", "power_off": "system_shutdown",
            "turn_off": "system_shutdown", "poweroff": "system_shutdown",
            "shut_down": "system_shutdown", "halt": "system_shutdown",

            # --- system_reboot ---
            "reboot": "system_reboot", "restart": "system_reboot",
            "reset_system": "system_reboot", "restart_computer": "system_reboot",

            # --- system_suspend ---
            "suspend": "system_suspend", "sleep": "system_suspend",
            "standby": "system_suspend", "go_to_sleep": "system_suspend",
            "sleep_mode": "system_suspend", "hibernate": "system_suspend",

            # --- battery_status ---
            "battery": "battery_status", "check_battery": "battery_status",
            "power_level": "battery_status", "battery_level": "battery_status",
            "power_remaining": "battery_status", "charge_level": "battery_status",
        }

        action = ALIASES.get(raw_action, raw_action)
        print(f"ACTION: {raw_action} -> {action}", flush=True)

        if action not in VALID_TOOLS:
            # With native tool-calling the model can only call names we
            # declared in TOOLS, so this should be rare — but keep it as a
            # safety net in case a model hallucinates a close-but-wrong name.
            return "INVALID_ACTION"

        if action == "get_time":
            now_dt = datetime.datetime.now()
            # strftime("%p") da "AM"/"PM" en inglés; un TTS en español lo
            # lee letra por letra ("pe eme") en vez de sonar natural. Se
            # arma la frase a mano con una franja horaria en español.
            hour_12 = now_dt.strftime("%I:%M").lstrip("0") or "12:00"
            hour_24 = now_dt.hour
            if 0 <= hour_24 < 6:
                franja = "de la madrugada"
            elif 6 <= hour_24 < 12:
                franja = "de la mañana"
            elif 12 <= hour_24 < 19:
                franja = "de la tarde"
            else:
                franja = "de la noche"
            # Deterministic fact, no LLM needed to phrase it — same
            # short-circuit pattern as battery_status and calculate, so it
            # doesn't cost an extra LLM round trip or a chance to hallucinate.
            return f"TIME_RESULT::Son las {hour_12} {franja}."
        
        elif action == "search_web":
            print(f"Searching web for: {value}...", flush=True)
            try:
                with DDGS() as ddgs:
                    NEWS_HINTS = ("news", "noticias", "latest", "today", "hoy",
                                  "actualidad", "trending", "happening")
                    wants_news = any(h in value.lower() for h in NEWS_HINTS)
                    results = []

                    # If it smells like a news query, try news first.
                    if wants_news:
                        try:
                            results = list(ddgs.news(value, region='us-en', max_results=3))
                            if results:
                                print(f"[DEBUG] Found {len(results)} news result(s)", flush=True)
                        except Exception as e:
                            print(f"[DEBUG] News Search Error: {e}", flush=True)

                    # General text search: primary path for facts/people/places,
                    # and fallback if news search found nothing.
                    if not results:
                        try:
                            results = list(ddgs.text(value, region='us-en', max_results=3))
                            if results:
                                print(f"[DEBUG] Found {len(results)} text result(s)", flush=True)
                        except Exception as e:
                            print(f"[DEBUG] Text Search Error: {e}", flush=True)

                    if not results:
                        print("[DEBUG] Search returned 0 results.", flush=True)
                        return "SEARCH_EMPTY"

                    # Combine up to 3 sources into one grounded context block,
                    # instead of a single 300-char snippet from one result.
                    blocks = []
                    for i, r in enumerate(results[:3], start=1):
                        title = r.get('title', 'No Title')
                        body = r.get('body', r.get('snippet', 'No Body'))
                        blocks.append(f"[Source {i}] {title}: {body[:400]}")

                    return f"SEARCH RESULTS for '{value}':\n" + "\n".join(blocks)
            except Exception as e:
                print(f"[DEBUG] Connection/Library Error: {e}", flush=True)
                return "SEARCH_ERROR"
        
        elif action == "capture_image":
             return "IMAGE_CAPTURE_TRIGGERED"

        elif action == "calculate":
            try:
                result = self._safe_eval_math(value)
                # Spoken-friendly phrasing on purpose: the TTS cleanup in
                # speak() strips "*"/"=" (not in its allowed charset), so
                # repeating the raw expression came out as garbled gaps
                # like "122  895  109190". A plain sentence survives that
                # cleanup intact.
                return f"CALC_RESULT::Eso es igual a {result}."
            except Exception as e:
                return f"CALC_ERROR::{e}"

        elif action == "battery_status":
            return self._get_battery_status()

        elif action in ("system_shutdown", "system_reboot", "system_suspend"):
            # Never execute directly here. Just flag it; chat_and_respond()
            # will ask the user to confirm before anything destructive runs.
            return f"CONFIRM_ACTION::{action}"

        return None

    # --- SYSTEM TOOLS HELPERS ---

    def _get_battery_status(self):
        base = "/sys/class/power_supply"
        try:
            bat_dirs = sorted(d for d in os.listdir(base) if d.upper().startswith("BAT"))
            if not bat_dirs:
                return "BATTERY_RESULT::No encontré ninguna batería en este sistema."
            bat = bat_dirs[0]
            with open(os.path.join(base, bat, "capacity")) as f:
                capacity = f.read().strip()
            status = "unknown"
            try:
                with open(os.path.join(base, bat, "status")) as f:
                    status = f.read().strip().lower()
            except Exception:
                pass
            # El sysfs de Linux devuelve el status en inglés (charging,
            # discharging, full, not charging); lo traducimos para que no
            # se cuele una palabra en inglés en medio de la frase hablada.
            status_es = {
                "charging": "cargando",
                "discharging": "descargando",
                "full": "llena",
                "not charging": "sin cargar",
                "unknown": "estado desconocido",
            }.get(status, status)
            return f"BATTERY_RESULT::La batería está al {capacity} por ciento, actualmente {status_es}."
        except Exception as e:
            return f"BATTERY_RESULT::No pude leer el estado de la batería: {e}"

    def _execute_system_action(self, action_name):
        """Runs a previously-confirmed destructive system action."""
        commands = {
            "system_shutdown": ["systemctl", "poweroff"],
            "system_reboot": ["systemctl", "reboot"],
            "system_suspend": ["systemctl", "suspend"],
        }
        messages = {
            "system_shutdown": "Listo, apagando el sistema. ¡Chao!",
            "system_reboot": "Listo, reiniciando ahora.",
            "system_suspend": "Listo, entrando en suspensión.",
        }
        cmd = commands.get(action_name)
        if not cmd:
            return "No reconozco esa acción del sistema."
        try:
            subprocess.Popen(cmd)
            return messages[action_name]
        except Exception as e:
            print(f"[SYSTEM ACTION ERROR] {action_name}: {e}", flush=True)
            return (f"I couldn't do that ({e}). This usually means the current "
                    f"user session doesn't have polkit permission to run "
                    f"'{' '.join(cmd)}' without a password.")

    # Stock phrases whisper.cpp hallucinates on silence/near-silent audio
    # (trained on YouTube data, so this is its go-to filler). These show up
    # verbatim when the mic captures mostly dead air, e.g. while waiting on
    # a confirmation reply — treat them as if nothing was heard at all.
    _WHISPER_HALLUCINATIONS = {
        "suscribete", "suscribete al canal", "suscribanse",
        "gracias por ver", "gracias por ver el video",
        "like y suscribete", "no olvides suscribirte",
    }

    @staticmethod
    def _is_junk_transcription(text):
        """Filters common whisper.cpp non-speech artifacts ([Música],
        [BLANK_AUDIO], stray punctuation) and near-empty fragments before
        they ever reach the LLM. Small models tend to parrot back a literal
        instruction phrase (e.g. "Normal Text") when given nonsense input
        instead of asking for clarification, so it's cheaper and more
        reliable to catch this here than to prompt-engineer around it."""
        stripped = text.strip()
        if not stripped:
            return True
        if re.fullmatch(r"[\[\(].*[\]\)]?", stripped):
            return True
        alnum = re.sub(r"[^\w]", "", stripped, flags=re.UNICODE)
        normalized = unicodedata.normalize("NFKD", alnum.lower())
        normalized = "".join(c for c in normalized if not unicodedata.combining(c))
        if normalized in BotGUI._WHISPER_HALLUCINATIONS:
            return True
        # Short answers like "sí"/"no" are valid despite being under the
        # general junk-length threshold below — don't let a 2-letter reply
        # get discarded as noise.
        if normalized in {"si", "no", "ok"}:
            return False
        return len(alnum) < 3

    @staticmethod
    def _is_affirmative(text):
        text = text.lower().strip().strip(".!¡¿?")
        affirmative_words = {
            "yes", "yeah", "yep", "confirm", "confirmed", "do it", "go ahead",
            "sure", "okay", "ok", "affirmative",
            "si", "sí", "confirmo", "dale", "hazlo", "claro", "de una",
        }
        return text in affirmative_words or any(
            text.startswith(w) for w in affirmative_words
        )

    # =========================================================================
    # 4. CORE LOGIC
    # =========================================================================

    def safe_main_execution(self):
        try:
            self.warm_up_logic()
            self.tts_active.set()
            self.tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
            self.tts_thread.start()
            
            while True:
                awaiting_confirmation = (
                    self.pending_confirmation is not None
                    and time.time() < self.pending_confirmation_expiry
                )

                if awaiting_confirmation:
                    if self.exiting:
                        break
                    # Skip the wake word entirely: we just asked a yes/no
                    # question, so go straight back to listening for the
                    # reply instead of making the user say "bmo" again.
                    trigger_source = "CONFIRM"
                else:
                    trigger_source = self.detect_wake_word_or_ptt()
                    if trigger_source == "EXIT" or self.exiting:
                        break
                    if self.interrupted.is_set():
                        self.interrupted.clear()
                        self.set_state(BotStates.IDLE, "Reiniciando...")
                        continue

                self.set_state(BotStates.LISTENING, "¡Te escucho!")
                
                audio_file = None
                if trigger_source == "PTT":
                    audio_file = self.record_voice_ptt()
                else:
                    audio_file = self.record_voice_adaptive()
                
                if not audio_file: 
                    self.set_state(BotStates.IDLE, "No escuché nada.")
                    continue
                
                user_text = self.transcribe_audio(audio_file)
                if not user_text or self._is_junk_transcription(user_text):
                    self.set_state(BotStates.IDLE, "No entendí eso.")
                    continue
                
                self.append_to_text(f"YOU: {user_text}")
                self.interrupted.clear()
                self.chat_and_respond(user_text, img_path=None)
                    
        except Exception as e:
            traceback.print_exc()
            self.set_state(BotStates.ERROR, f"Error fatal: {str(e)[:40]}")

    def warm_up_logic(self):
        self.set_state(BotStates.WARMUP, "Calentando el cerebro...")
        try:
            # ollama.generate(prompt="") solo carga los pesos en memoria --
            # con prompt vacío no hay ningún forward pass real, así que
            # nunca ejercita el camino de cómputo real (en este equipo,
            # sobre la iGPU vía Vulkan). Eso dejaba la compilación de
            # shaders/kernels de la PRIMERA llamada real para la primera
            # pregunta del usuario: 35-36s medidos en logs [TIMING] reales
            # (dos sesiones distintas), contra 1.6-3.4s en todas las
            # llamadas siguientes -- el mismo fenómeno de "primer load" que
            # ya documentamos para whisper-server (ver
            # WHISPER_SERVER_STARTUP_TIMEOUT más arriba), pero sin cubrirlo
            # aquí.
            #
            # Fix: calentar con una llamada REPRESENTATIVA de la real --
            # mismo system prompt + TOOL_FEW_SHOT + tools=TOOLS + las
            # mismas OLLAMA_OPTIONS_ROUTE que usa chat_and_respond() -- y
            # descartar la respuesta. El único propósito es pagar ese costo
            # de compilación aquí, durante "Calentando el cerebro" (donde
            # ya estás esperando de todas formas), en vez de en la primera
            # pregunta real.
            warmup_messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                *TOOL_FEW_SHOT,
                {"role": "user", "content": "Hola"},
            ]
            with step_timer(f"Warmup LLM, llamada real (tools, {TEXT_MODEL})"):
                ollama_chat_safe(
                    model=TEXT_MODEL,
                    messages=warmup_messages,
                    tools=TOOLS,
                    stream=False,
                    options=OLLAMA_OPTIONS_ROUTE,
                )
        except Exception as e:
            print(f"Failed to warm up {TEXT_MODEL}: {e}", flush=True)
        self._start_whisper_server()
        self.play_sound(self.get_random_sound(greeting_sounds_dir))
        print("Models loaded.", flush=True)

    def _start_whisper_server(self):
        """Levanta whisper-server una sola vez (ver comentario junto a
        WHISPER_SERVER_BIN). Cualquier fallo aquí deja whisper_server_ready
        en False y transcribe_audio() sigue funcionando con whisper-cli por
        turno -- este método nunca debe poder tumbar el arranque del agente."""
        if requests is None:
            print(
                "[STT] Paquete 'requests' no instalado -- whisper-server "
                "desactivado, usando whisper-cli por turno. "
                "Instálalo con: pip install requests --break-system-packages",
                flush=True,
            )
            return

        if not os.path.exists(WHISPER_SERVER_BIN):
            print(
                f"[STT] whisper-server no encontrado en '{WHISPER_SERVER_BIN}' -- "
                f"usando whisper-cli por turno (recarga el modelo cada vez; "
                f"notorio con modelos grandes como large-v3-turbo). Para "
                f"activarlo, compila el target que falta: "
                f"cmake --build whisper.cpp/build --config Release -j --target whisper-server",
                flush=True,
            )
            return

        stt_model = CURRENT_CONFIG.get("stt_model", "./whisper.cpp/models/ggml-base.bin")
        if not os.path.exists(stt_model):
            return  # transcribe_audio ya reporta este error con detalle

        stt_language = CURRENT_CONFIG.get("stt_language", "es")
        stt_initial_prompt = CURRENT_CONFIG.get("stt_initial_prompt", "")

        cmd = [
            WHISPER_SERVER_BIN,
            "-m", stt_model,
            "-t", str(WHISPER_NUM_THREAD),
            "-sns",  # suprime tokens no-verbales, igual que whisper-cli antes
            "--host", WHISPER_SERVER_HOST,
            "--port", str(WHISPER_SERVER_PORT),
            "--public", WHISPER_SERVER_PUBLIC_DIR,
        ]
        if stt_language and stt_language.lower() != "auto":
            cmd.extend(["-l", stt_language])
        if stt_initial_prompt:
            cmd.extend(["--prompt", stt_initial_prompt])

        try:
            with step_timer(f"Arranque whisper-server ({os.path.basename(stt_model)})"):
                self.whisper_server_proc = subprocess.Popen(
                    cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                health_url = f"http://{WHISPER_SERVER_HOST}:{WHISPER_SERVER_PORT}/health"
                deadline = time.time() + WHISPER_SERVER_STARTUP_TIMEOUT
                while time.time() < deadline:
                    if self.whisper_server_proc.poll() is not None:
                        break  # el proceso murió al iniciar (puerto ocupado, modelo inválido...)
                    try:
                        if requests.get(health_url, timeout=0.5).ok:
                            self.whisper_server_ready = True
                            break
                    except requests.exceptions.RequestException:
                        pass
                    time.sleep(0.3)
        except Exception as e:
            print(f"[STT] No se pudo iniciar whisper-server: {e}", flush=True)
            self.whisper_server_proc = None

        if self.whisper_server_ready:
            print(
                f"[STT] whisper-server listo en {WHISPER_SERVER_HOST}:"
                f"{WHISPER_SERVER_PORT} -- modelo cargado una sola vez para "
                f"toda la sesión.",
                flush=True,
            )
        else:
            print("[STT] whisper-server no respondió a tiempo -- usando whisper-cli por turno.", flush=True)
            self._stop_whisper_server()

    def _stop_whisper_server(self):
        if self.whisper_server_proc:
            try:
                self.whisper_server_proc.terminate()
                self.whisper_server_proc.wait(timeout=2)
            except Exception:
                try:
                    self.whisper_server_proc.kill()
                except Exception:
                    pass
        self.whisper_server_proc = None
        self.whisper_server_ready = False

    def detect_wake_word_or_ptt(self):
        self.set_state(BotStates.IDLE, "Esperando...")
        self.ptt_event.clear()
        
        if self.oww_model: self.oww_model.reset()

        if self.oww_model is None:
            self.ptt_event.wait()
            self.ptt_event.clear()
            return "PTT"

        CHUNK_SIZE = 1280
        OWW_SAMPLE_RATE = 16000

        input_rate = choose_input_samplerate(INPUT_DEVICE_NAME, CURRENT_CONFIG.get("input_sample_rate"))
        use_resampling = (input_rate != OWW_SAMPLE_RATE)
        input_chunk_size = int(CHUNK_SIZE * (input_rate / OWW_SAMPLE_RATE)) if use_resampling else CHUNK_SIZE

        stream_args = {
            "samplerate": input_rate, 
            "channels": 1, 
            "dtype": 'int16', 
            "blocksize": input_chunk_size, 
            "device": INPUT_DEVICE_NAME
        }

        # Try to find a compatible block size and sample rate
        try:
            # First attempt: standard settings
            self._listen_loop(stream_args, input_chunk_size, CHUNK_SIZE, use_resampling)
        except StopIteration as si:
            return str(si)
        except Exception as e:
            print(f"[AUDIO] Stream failed with defaults: {e}. Retrying with loose settings...", flush=True)
            try:
                # Second attempt: Let PortAudio decide blocksize (0) and latency
                stream_args["blocksize"] = 0 
                stream_args["latency"] = "high"
                # If blocksize is variable, we must read specific amounts manually or handle buffering.
                # Simplest fallback: Just attempt small fixed block
                stream_args["blocksize"] = 1024
                use_resampling = True
                
                self._listen_loop(stream_args, 1024, CHUNK_SIZE, use_resampling)
            except StopIteration as si:
                return str(si)
            except Exception as e2:
                print(f"[CRITICAL] Wake Word Stream Error: {e2}")
                self.ptt_event.wait()
                return "PTT"
        
        return "WAKE"

    def _listen_loop(self, stream_args, input_chunk_size, target_chunk_size, use_resampling):
        # Force software backend (no mmap) via environment variable if possible, 
        # but here we can try to hint loop settings.
        # However, the most effective fix for ALSA mmap issues is often just asking for 'blocksize=0' 
        # and letting portaudio manage the buffering, OR very small chunks.
        
        # Let's try to be less aggressive with reads.

        with sd.InputStream(**stream_args) as stream:
                print(f"[AUDIO] Listening with rate {stream_args['samplerate']} and block {stream_args['blocksize']}", flush=True)
                
                # Pre-allocate buffer for speed
                # If blocksize is 0, we read what is available.
                
                while True:
                    if self.exiting:
                        raise StopIteration("EXIT")

                    if self.ptt_event.is_set():
                        self.ptt_event.clear()
                        raise StopIteration("PTT")

                    rlist, _, _ = select.select([sys.stdin], [], [], 0.001)
                    if rlist: 
                        sys.stdin.readline()
                        raise StopIteration("CLI")

                    # If fallback mode (blocksize 0), read fixed amount
                    read_size = input_chunk_size
                    if stream_args.get('blocksize') == 0:
                        read_size = 1024 # Safe small read
                    
                    try:
                        data, overflow = stream.read(read_size)
                        if overflow:
                            print("!", end="", flush=True) 
                            # If we overflow excessively, raise error to trigger fallback to SAFE MODE (PulseAudio/Software)
                            # We can use a simple counter attached to the function or object, but here raising immediately 
                            # after a few in a row is safest.
                            raise RuntimeError("Audio Buffer Overflow - Triggering Safe Mode")
                    except Exception as e:
                        # Convert uncatchable PaErrorCode wrapper to standard Exception if needed
                        # But honestly, `raise e` should work... unless it's a SystemExit?
                        # Let's wrap it in a new exception to be sure it bubbles up
                        raise RuntimeError(f"Audio read failed: {e}")

                    audio_data = np.frombuffer(data, dtype=np.int16)

                    # Ensure flattening for openwakeword compatibility
                    if audio_data.ndim > 1:
                        audio_data = audio_data.flatten()

                    if use_resampling:
                        # FAST RESAMPLING: Nearest-neighbor slicing instead of scipy.signal.resample
                        # This avoids the CPU bottleneck that causes overflow (!!!!!!!) on Raspberry Pi
                        step = len(audio_data) / target_chunk_size
                        indices = np.arange(0, len(audio_data), step)[:target_chunk_size].astype(int)
                        audio_data = audio_data[indices]
                    
                    # Convert to float for model prediction without needing heavy resampling logic
                    # The wake word model needs 16000, which we just faked above.
                    
                    # Debug volume occasionally
                    current_max = np.max(np.abs(audio_data))
                    
                    # Only predict if volume is significant to save CPU
                    if current_max > 200: 
                        prediction = self.oww_model.predict(audio_data)
                        for mdl in self.oww_model.prediction_buffer.keys():
                            score = list(self.oww_model.prediction_buffer[mdl])[-1]
                            if score > 0.1: # Show potential triggers
                                print(f"\r[Oww] Score: {score:.3f} | Vol: {current_max}   ", end="", flush=True)

                            if score > WAKE_WORD_THRESHOLD:
                                print(f"\n[WAKE] Triggered on '{mdl}' with score: {score:.2f}", flush=True)
                                self.oww_model.reset() 
                                return # Success


    def record_voice_adaptive(self, filename="input.wav"):
        print("Recording (Adaptive)...", flush=True)
        # 0.5s heredado de la era Raspberry Pi ("hardware contention causes
        # freezes" en los comentarios originales de sd.stop() más abajo).
        # Bajado a 0.15s para esta laptop (sin esa contención conocida) --
        # si notas cortes o congelamientos de audio al empezar a grabar,
        # esto es lo primero que hay que subir de vuelta.
        time.sleep(0.15)
        samplerate = choose_input_samplerate(INPUT_DEVICE_NAME, CURRENT_CONFIG.get("input_sample_rate"))

        configured_threshold = CURRENT_CONFIG.get("silence_threshold", 0.006)
        silence_duration = CURRENT_CONFIG.get("silence_duration", 0.9)
        max_record_time = CURRENT_CONFIG.get("max_record_time", 30.0)
        calibration_seconds = CURRENT_CONFIG.get("noise_calibration_seconds", 0.25)
        buffer = []
        silent_chunks = 0
        chunk_duration = 0.05 
        chunk_size = int(samplerate * chunk_duration)
        
        num_silent_chunks = max(1, int(silence_duration / chunk_duration))
        max_chunks = int(max_record_time / chunk_duration)
        num_calibration_chunks = max(1, int(calibration_seconds / chunk_duration))
        recorded_chunks = 0
        silence_started = False
        calibration_levels = []
        # Starts at the configured value; refined once the calibration
        # window (below) closes. Used as a fallback if calibration ever
        # comes back empty.
        effective_threshold = configured_threshold

        def callback(indata, frames, time_info, status):
            nonlocal silent_chunks, recorded_chunks, silence_started, effective_threshold
            volume_norm = np.linalg.norm(indata) / np.sqrt(len(indata))
            buffer.append(indata.copy())
            recorded_chunks += 1

            # First ~0.25s: measure the room's ambient noise instead of
            # assuming a fixed magic number. This replaces the old blind
            # "skip the first 5 chunks" warmup with the same warmup time,
            # now used on purpose.
            if recorded_chunks <= num_calibration_chunks:
                calibration_levels.append(volume_norm)
                return
            if recorded_chunks == num_calibration_chunks + 1:
                noise_floor = float(np.mean(calibration_levels)) if calibration_levels else 0.0
                # Whichever is higher: your configured floor, or ~2x the
                # measured ambient noise. A noisy room raises the bar for
                # "silence" so it doesn't take max_record_time to trigger;
                # a quiet room keeps the sensitive configured default.
                effective_threshold = max(configured_threshold, noise_floor * 2.0)

            if volume_norm < effective_threshold:
                silent_chunks += 1
                if silent_chunks >= num_silent_chunks: silence_started = True
            else: silent_chunks = 0

        try:
            with step_timer("Grabación (adaptativa)"):
                # Explicitly close stream if it exists to free hardware
                sd.stop()
                time.sleep(0.05)  # bajado de 0.2s, mismo motivo que arriba

                with sd.InputStream(samplerate=samplerate, channels=1, callback=callback,
                                    device=INPUT_DEVICE_NAME, blocksize=chunk_size):
                    while not silence_started and recorded_chunks < max_chunks:
                        if self.exiting:
                            break
                        sd.sleep(int(chunk_duration * 1000))
        except Exception as e: 
            print(f"[AUDIO ERROR] Adaptive Recording Failed: {e}", flush=True)
            return None 
        
        return self.save_audio_buffer(buffer, filename, samplerate)

    def record_voice_ptt(self, filename="input.wav"):
        print("Recording (PTT)...", flush=True)
        time.sleep(0.15)  # ver comentario en record_voice_adaptive
        samplerate = choose_input_samplerate(INPUT_DEVICE_NAME, CURRENT_CONFIG.get("input_sample_rate"))

        buffer = []
        def callback(indata, frames, time_info, status): buffer.append(indata.copy())
        
        try:
            # Explicitly close stream if it exists to free hardware
            # This is critical on Pi 5 where hardware contention causes freezes
            sd.stop() 
            time.sleep(0.05)  # bajado de 0.2s -- ver comentario en record_voice_adaptive
            
            with sd.InputStream(samplerate=samplerate, channels=1, callback=callback, device=INPUT_DEVICE_NAME):
                while self.recording_active.is_set(): 
                    if self.exiting:
                        break
                    sd.sleep(50)
        except Exception as e: 
            print(f"[AUDIO ERROR] PTT Recording Failed: {e}", flush=True)
            return None
            
        return self.save_audio_buffer(buffer, filename, samplerate)

    def save_audio_buffer(self, buffer, filename, samplerate=16000):
        if not buffer: return None
        audio_data = np.concatenate(buffer, axis=0).flatten()
        audio_data = np.nan_to_num(audio_data, nan=0.0, posinf=0.0, neginf=0.0)
        audio_data = (audio_data * 32767).astype(np.int16)
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(samplerate)
            wf.writeframes(audio_data.tobytes())
        self.play_sound(self.get_random_sound(ack_sounds_dir))
        return filename

    def transcribe_audio(self, filename):
        print("Transcribing...", flush=True)
        stt_model = CURRENT_CONFIG.get("stt_model", "./whisper.cpp/models/ggml-base.bin")
        stt_language = CURRENT_CONFIG.get("stt_language", "es")

        if not os.path.exists(stt_model):
            print(
                f"[STT ERROR] Model not found at '{stt_model}'. Download a "
                f"MULTILINGUAL ggml model (filename must NOT end in '.en.bin') "
                f"with, e.g.:\n"
                f"    bash ./whisper.cpp/models/download-ggml-model.sh base\n"
                f"then point \"stt_model\" in config.json at the resulting file.",
                flush=True,
            )
            return ""

        stt_initial_prompt = CURRENT_CONFIG.get("stt_initial_prompt", "")

        if self.whisper_server_ready:
            transcription = self._transcribe_via_server(filename, stt_language, stt_initial_prompt)
            if transcription is not None:
                print(f"Heard: '{transcription}'", flush=True)
                return transcription.strip()
            # El server estaba arriba pero falló a media sesión (se cayó,
            # timeout, etc.) -- cae a whisper-cli para este turno en vez de
            # dejar al usuario sin respuesta. Se desactiva para el resto de
            # la sesión: si murió una vez, insistir por turno solo agrega
            # timeouts de 0.5s a cada intento.
            print("[STT] whisper-server no respondió, usando whisper-cli para el resto de la sesión.", flush=True)
            self.whisper_server_ready = False

        try:
            cmd = [
                "./whisper.cpp/build/bin/whisper-cli", "-m", stt_model, "-t", str(WHISPER_NUM_THREAD), "-f", filename,
                # Suppress non-speech tokens (e.g. the "[Música]"/
                # "[BLANK_AUDIO]" tags whisper emits on silence or noise) at
                # the source, instead of only catching them afterwards in
                # _is_junk_transcription.
                "-sns",
            ]
            # "auto" lets whisper detect the language per-utterance instead
            # of forcing one; costs a little extra latency but useful if
            # you regularly switch between Spanish and English.
            if stt_language and stt_language.lower() != "auto":
                cmd.extend(["-l", stt_language])
            # Biases decoding toward your own names/vocabulary (see
            # DEFAULT_CONFIG's comment) — free accuracy, no extra LLM call.
            if stt_initial_prompt:
                cmd.extend(["--prompt", stt_initial_prompt])
            with step_timer(f"Transcripción, whisper-cli ({os.path.basename(stt_model)})"):
                result = subprocess.run(cmd, capture_output=True, text=True)
            transcription_lines = result.stdout.strip().split('\n')
            if transcription_lines and transcription_lines[-1].strip():
                last_line = transcription_lines[-1].strip()
                if ']' in last_line: transcription = last_line.split("]")[1].strip()
                else: transcription = last_line
            else: transcription = ""
            print(f"Heard: '{transcription}'", flush=True)
            return transcription.strip()
        except Exception as e:
            print(f"Transcription Error: {e}")
            return ""

    def _transcribe_via_server(self, filename, stt_language, stt_initial_prompt):
        """POST a whisper-server /inference. Devuelve el texto, o None si el
        server no respondió (transcribe_audio decide el fallback)."""
        url = f"http://{WHISPER_SERVER_HOST}:{WHISPER_SERVER_PORT}/inference"
        data = {"response_format": "json"}
        if stt_language and stt_language.lower() != "auto":
            data["language"] = stt_language
        if stt_initial_prompt:
            data["prompt"] = stt_initial_prompt
        try:
            with step_timer("Transcripción, whisper-server (modelo ya cargado)"):
                with open(filename, "rb") as f:
                    resp = requests.post(url, files={"file": f}, data=data, timeout=30)
                resp.raise_for_status()
                return resp.json().get("text", "")
        except Exception as e:
            print(f"[STT] Error consultando whisper-server: {e}", flush=True)
            return None

    def _grab_camera_frame(self):
        """Generic V4L2 capture (replaces rpicam-still, which only exists on
        Raspberry Pi OS/libcamera). Many UVC webcams expose more than one
        /dev/video* node (one real capture node, one metadata-only node), so
        this probes indices instead of assuming 0. The last working index is
        cached and tried first; if it ever stops working (camera unplugged/
        replugged into a different node) it falls back to a full re-probe
        automatically."""
        candidates = list(range(5))
        if self._camera_index is not None and self._camera_index in candidates:
            candidates.remove(self._camera_index)
            candidates.insert(0, self._camera_index)

        for idx in candidates:
            cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
            if not cap.isOpened():
                cap.release()
                continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            frame = None
            # Discard the first few frames: auto-exposure/auto-white-balance
            # on most UVC webcams hasn't settled yet and the very first
            # frame often comes back black or blown out.
            for _ in range(5):
                ok, frame = cap.read()
                if not ok:
                    frame = None
                    break
            cap.release()

            if frame is not None:
                self._camera_index = idx
                return frame

        self._camera_index = None
        return None

    def capture_image(self):
        self.set_state(BotStates.CAPTURING, "Observando...")
        try:
            frame = self._grab_camera_frame()
            if frame is None:
                print("Camera Error: no /dev/video* device returned a usable frame", flush=True)
                return None
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            rotation = CURRENT_CONFIG.get("camera_rotation", 0)
            if rotation != 0:
                img = img.rotate(rotation, expand=True) 
            img.save(BMO_IMAGE_FILE)
            return BMO_IMAGE_FILE
        except Exception as e:
            print(f"Camera Error: {e}")
            return None

    # =========================================================================
    # 5. CHAT & RESPOND
    # =========================================================================

    def _say(self, text, img_path=None, log_as_assistant=True):
        """Speak `text` right now: pushes it to the on-screen log + TTS
        queue. Small helper so every response branch below doesn't repeat
        the same four lines."""
        self.thinking_sound_active.clear()
        self.set_state(BotStates.SPEAKING, "Hablando...", cam_path=img_path)
        self.append_to_text("BOT: ", newline=False)
        self.append_to_text(text, newline=True)
        with self.tts_queue_lock:
            self.tts_queue.append(text)
        if log_as_assistant:
            self.session_memory.append({"role": "assistant", "content": text})

    def chat_and_respond(self, text, img_path=None):
        # Handle a pending yes/no confirmation for a destructive system
        # action before doing anything else (including waking the LLM).
        if self.pending_confirmation:
            action_name = self.pending_confirmation
            self.pending_confirmation = None
            if time.time() > self.pending_confirmation_expiry:
                fallback_text = "Se venció el tiempo para confirmar, así que no hice nada."
            elif self._is_affirmative(text):
                fallback_text = self._execute_system_action(action_name)
            else:
                fallback_text = "Listo, cancelado."
            self._say(fallback_text, img_path=img_path)
            self.wait_for_tts()
            self.set_state(BotStates.IDLE, "Listo")
            return

        _reset_memory_phrases = (
            "forget everything", "reset memory",
            "olvida todo", "olvida la memoria", "borra la memoria",
            "borra todo", "reinicia la memoria",
        )
        if any(p in text.lower() for p in _reset_memory_phrases):
            self.session_memory = []
            self.permanent_memory = [{"role": "system", "content": SYSTEM_PROMPT}]
            self.save_chat_history()
            with self.tts_queue_lock: 
                self.tts_queue.append("Listo, borré la memoria.")
            self.set_state(BotStates.IDLE, "Memoria borrada")
            return

        model_to_use = VISION_MODEL if img_path else TEXT_MODEL
        self.set_state(BotStates.THINKING, "Pensando...", cam_path=img_path)

        self.thinking_sound_active.set()
        threading.Thread(target=self._run_thinking_sound_loop, daemon=True).start()

        try:
            # --- VISION PATH: unrelated to tool-calling, keep streaming for
            # low perceived latency (moondream doesn't support tools=). ---
            if img_path:
                messages = [{"role": "user", "content": text, "images": [img_path]}]
                full_response_buffer = ""
                sentence_buffer = ""
                stream = ollama.chat(model=model_to_use, messages=messages, stream=True, options=OLLAMA_OPTIONS_CHAT)
                for chunk in stream:
                    if self.interrupted.is_set(): break
                    content = chunk['message']['content']
                    full_response_buffer += content
                    self.thinking_sound_active.clear()
                    if self.current_state != BotStates.SPEAKING:
                        self.set_state(BotStates.SPEAKING, "Hablando...", cam_path=img_path)
                        self.append_to_text("BOT: ", newline=False)
                    self._stream_to_text(content)
                    sentence_buffer += content
                    if any(punct in content for punct in ".!?\n"):
                        clean_sentence = sentence_buffer.strip()
                        if clean_sentence and re.search(r'[a-zA-Z0-9]', clean_sentence):
                            with self.tts_queue_lock: self.tts_queue.append(clean_sentence)
                        sentence_buffer = ""
                self.append_to_text("")
                self.session_memory.append({"role": "assistant", "content": full_response_buffer})
                self.wait_for_tts()
                self.set_state(BotStates.IDLE, "Listo")
                return

            # --- TEXT PATH: native tool-calling (tools=TOOLS). ---
            # STATELESS BY DESIGN: only the system prompt + this one turn go
            # to the model, never the accumulated session history. Feeding a
            # growing transcript on every call is what made the 1.5-2B
            # models "lose the thread" after a handful of turns in testing
            # (re-narrating old answers, skipping tool calls, inventing
            # numbers) even though the exact same model scored well in the
            # single-turn benchmark. session_memory is still appended below
            # and saved to disk via save_chat_history() so you keep a
            # transcript, it's just not replayed back into future prompts.
            user_msg = {"role": "user", "content": text}
            # Fixed few-shot (see TOOL_FEW_SHOT above) goes between the
            # system prompt and the real turn — same every call, doesn't
            # accumulate, just demonstrates the tool_call pattern the model
            # was skipping for zero-argument tools.
            messages = [self.permanent_memory[0], *TOOL_FEW_SHOT, user_msg]

            # Ollama needs the full reply to know whether tool_calls came
            # back, so this call is non-streaming — unlike the old free-text
            # JSON hack, there's no partial output worth streaming here.
            # Temperature: OLLAMA_OPTIONS_ROUTE (0.3), matching what was
            # actually benchmarked for tool-calling reliability. Note this
            # also governs plain chat replies since decision + reply happen
            # in one call now; if casual conversation feels flat, that's the
            # trade-off — raise it back up via system_prompt_extras/config
            # if reliability stops being the priority.
            response = None
            if self.tools_supported:
                try:
                    with step_timer(f"LLM decisión/tool-routing ({model_to_use})"):
                        response = ollama_chat_safe(
                            model=model_to_use,
                            messages=messages,
                            tools=TOOLS,
                            stream=False,
                            options=OLLAMA_OPTIONS_ROUTE,
                        )
                except ollama.ResponseError as e:
                    if "does not support tools" in str(e).lower():
                        self.tools_supported = False
                        print("=" * 70, flush=True)
                        print(f"[TOOLS DISABLED] '{model_to_use}' does not support "
                              f"native tool-calling. Falling back to plain chat — "
                              f"time/search/battery/system tools will NOT work "
                              f"until you switch text_model in config.json to a "
                              f"model with tool support (e.g. qwen2.5:1.5b).",
                              flush=True)
                        print("=" * 70, flush=True)
                    else:
                        raise

            if response is None:
                # Either tools were already known unsupported, or we just
                # found out — either way, degrade to a plain chat call so
                # the user still gets a spoken reply instead of silence
                # or a repeated crash.
                with step_timer(f"LLM chat plano/sin tools ({model_to_use})"):
                    response = ollama_chat_safe(
                        model=model_to_use,
                        messages=messages,
                        stream=False,
                        options=OLLAMA_OPTIONS_ROUTE,
                    )

            self.thinking_sound_active.clear()
            reply = response.get('message', {}) or {}
            tool_calls = reply.get('tool_calls') or []

            if not tool_calls:
                # Plain conversational reply, no tool needed.
                self._say(reply.get('content', '') or "...", img_path=img_path)
                self.wait_for_tts()
                self.set_state(BotStates.IDLE, "Listo")
                return

            # Only act on the first tool call — one action per turn. Small
            # models occasionally emit more than one call when unsure; the
            # rest would just be noise/duplicates.
            call = tool_calls[0]
            fn = call.get('function', {}) or {}
            tool_name = fn.get('name', '')
            raw_args = fn.get('arguments', {})
            # Some ollama client/server versions hand back arguments as a
            # JSON string instead of an already-parsed dict — accept both.
            if isinstance(raw_args, str):
                try:
                    raw_args = json.loads(raw_args)
                except Exception:
                    raw_args = {}
            args = raw_args if isinstance(raw_args, dict) else {}

            self.session_memory.append({"role": "user", "content": text})
            with step_timer(f"Tool call ({tool_name})"):
                tool_result = self.execute_action_and_get_result(tool_name, args)

            if tool_result == "IMAGE_CAPTURE_TRIGGERED":
                new_img_path = self.capture_image()
                if new_img_path:
                    self.chat_and_respond(text, img_path=new_img_path)
                    return
                self._say("No pude tomar la foto.", img_path=img_path, log_as_assistant=False)

            elif tool_result == "INVALID_ACTION":
                self._say("No estoy seguro de cómo hacer eso.", img_path=img_path, log_as_assistant=False)

            elif tool_result == "SEARCH_EMPTY":
                self._say("Busqué, pero no encontré nada sobre eso.", img_path=img_path, log_as_assistant=False)

            elif tool_result == "SEARCH_ERROR":
                self._say("No puedo conectarme a internet en este momento.", img_path=img_path, log_as_assistant=False)

            elif tool_result and tool_result.startswith("CONFIRM_ACTION::"):
                action_name = tool_result.split("::", 1)[1]
                friendly = {
                    "system_shutdown": "apague el computador",
                    "system_reboot": "reinicie el computador",
                    "system_suspend": "ponga el computador a dormir",
                }.get(action_name, action_name)
                self.pending_confirmation = action_name
                self.pending_confirmation_expiry = time.time() + self.CONFIRMATION_WINDOW_SECONDS
                self._say(
                    f"¿Seguro que quieres que {friendly}? Di 'sí' dentro de "
                    f"{self.CONFIRMATION_WINDOW_SECONDS} segundos para confirmar.",
                    img_path=img_path, log_as_assistant=False,
                )

            elif tool_result and tool_result.startswith(("TIME_RESULT::", "BATTERY_RESULT::", "CALC_RESULT::")):
                # Deterministic facts: speak the tool's own text directly,
                # no LLM round trip and no chance to embellish/hallucinate.
                self._say(tool_result.split("::", 1)[1], img_path=img_path, log_as_assistant=False)

            elif tool_result and tool_result.startswith("CALC_ERROR::"):
                self._say("No pude calcular eso, perdón.", img_path=img_path, log_as_assistant=False)

            elif tool_result:
                # e.g. search_web's raw SEARCH RESULTS block — needs a
                # natural-language pass, strictly grounded in what was
                # actually found (see the anchoring system prompt below).
                summary_prompt = [
                    {"role": "system", "content": (
                        "Answer the user's question using ONLY the facts contained in "
                        "RESULT below. Do not add any name, date, number, or claim that "
                        "is not explicitly present in RESULT. If RESULT does not contain "
                        "enough information to answer, say so plainly instead of "
                        "guessing. Keep it to 1-3 short, spoken-friendly sentences."
                    )},
                    {"role": "user", "content": f"RESULT:\n{tool_result}\n\nUser Question: {text}"}
                ]

                self.set_state(BotStates.THINKING, "Leyendo...")
                self.thinking_sound_active.set()

                with step_timer(f"LLM resumen de resultados ({model_to_use})"):
                    final_resp = ollama_chat_safe(model=model_to_use, messages=summary_prompt, stream=False, options=OLLAMA_OPTIONS_CHAT)
                final_text = final_resp['message']['content']
                self._say(final_text, img_path=img_path)

            self.wait_for_tts()
            self.set_state(BotStates.IDLE, "Listo")

        except Exception as e:
            print(f"LLM Error: {e}")
            self.set_state(BotStates.ERROR, "¡Se colgó el cerebro!")

    def wait_for_tts(self):
        while self.tts_queue or self.tts_active.is_set():
            if self.interrupted.is_set(): break
            time.sleep(0.1)

    def _tts_worker(self):
        while True:
            text = None
            with self.tts_queue_lock:
                if self.tts_queue: 
                    text = self.tts_queue.pop(0)
                    self.tts_active.set() 
            if text: 
                self.speak(text)
                self.tts_active.clear() 
            else: time.sleep(0.05)

    def speak(self, text):
        clean = re.sub(r"[^\w\s,.!?:-]", "", text)
        if not clean.strip(): return
        
        print(f"[PIPER SPEAKING] '{clean}'", flush=True)
        voice_model = CURRENT_CONFIG.get("voice_model", "piper/en_GB-semaine-medium.onnx")
        
        try:
            with step_timer("TTS (Piper, síntesis + reproducción)"):
                self.current_audio_process = subprocess.Popen(
                    ["./piper/piper", "--model", voice_model, "--output-raw"], 
                    stdin=subprocess.PIPE, 
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL
                )
            
                self.current_audio_process.stdin.write(clean.encode() + b'\n')
                self.current_audio_process.stdin.close() 

                try:
                    device_info = sd.query_devices(kind='output')
                    native_rate = int(device_info['default_samplerate'])
                except:
                    native_rate = 48000 

                PIPER_RATE = 22050
                use_native_rate = False
            
                try:
                    sd.check_output_settings(device=None, samplerate=PIPER_RATE)
                except:
                    use_native_rate = True

                with sd.RawOutputStream(samplerate=native_rate if use_native_rate else PIPER_RATE, 
                                        channels=1, dtype='int16', 
                                        device=None, latency='low', blocksize=2048) as stream:
                    was_interrupted = False
                    while True:
                        if self.interrupted.is_set():
                            was_interrupted = True
                            break
                        data = self.current_audio_process.stdout.read(4096)
                        if not data: break 
                    
                        audio_chunk = np.frombuffer(data, dtype=np.int16)
                        if len(audio_chunk) > 0:
                            self.current_volume = np.max(np.abs(audio_chunk))
                            if use_native_rate:
                                num_samples = int(len(audio_chunk) * (native_rate / PIPER_RATE))
                                audio_chunk = scipy.signal.resample(audio_chunk, num_samples).astype(np.int16)
                            stream.write(audio_chunk.tobytes())
                        else:
                            self.current_volume = 0
                    if not was_interrupted:
                        # stream.write() puede devolver el control antes de
                        # que el último bloque termine de sonar en el
                        # hardware. Salir del `with` ahora llamaría a
                        # close(), que -- si el stream sigue "activo" --
                        # descarta ese resto como abort(), cortando la
                        # última sílaba. stream.stop() bloquea solo lo que
                        # falta (normalmente unas decenas de ms con
                        # blocksize=2048), en vez del sleep(0.5) fijo que
                        # había antes -- que sumaba hasta medio segundo
                        # MUERTO por cada frase (varias por respuesta larga)
                        # para cubrir ese mismo margen a ciegas. Si te
                        # interrumpieron a media frase no queremos esperar
                        # nada: por eso el `if not was_interrupted`.
                        stream.stop()
                    
        except Exception as e:
            print(f"Audio Error: {e}")
        finally:
            self.current_volume = 0 
            if self.current_audio_process:
                if self.current_audio_process.stdout: self.current_audio_process.stdout.close()
                if self.current_audio_process.poll() is None: self.current_audio_process.terminate()
                self.current_audio_process = None

    def _run_thinking_sound_loop(self):
        time.sleep(0.5)
        while self.thinking_sound_active.is_set():
            sound = self.get_random_sound(thinking_sounds_dir)
            if sound: self.play_sound(sound)
            for _ in range(50):
                if not self.thinking_sound_active.is_set(): return
                time.sleep(0.1)

    def get_random_sound(self, directory):
        if os.path.exists(directory):
            files = [f for f in os.listdir(directory) if f.endswith(".wav")]
            return os.path.join(directory, random.choice(files)) if files else None
        return None

    def play_sound(self, file_path):
        if not file_path or not os.path.exists(file_path): return
        try:
            with wave.open(file_path, 'rb') as wf:
                file_sr = wf.getframerate()
                data = wf.readframes(wf.getnframes())
                audio = np.frombuffer(data, dtype=np.int16)

            try:
                device_info = sd.query_devices(kind='output')
                native_rate = int(device_info['default_samplerate'])
            except:
                native_rate = 48000 

            playback_rate = file_sr
            try:
                sd.check_output_settings(device=None, samplerate=file_sr)
            except:
                playback_rate = native_rate
                num_samples = int(len(audio) * (native_rate / file_sr))
                audio = scipy.signal.resample(audio, num_samples).astype(np.int16)

            sd.play(audio, playback_rate)
            sd.wait() 
        except: pass

    def load_chat_history(self):
        if os.path.exists(MEMORY_FILE):
            try:
                with open(MEMORY_FILE, "r") as f:
                    history = json.load(f)
                if isinstance(history, list) and history and history[0].get("role") == "system":
                    # ALWAYS resync the system message to the SYSTEM_PROMPT
                    # currently in code. Without this, a memory.json saved
                    # under an older version of this script permanently
                    # pins the model to whatever instructions were live
                    # back then — e.g. the pre-migration "reply with a JSON
                    # action" prompt — no matter how many times you edit
                    # SYSTEM_PROMPT afterwards. This was silently making
                    # the model split its behavior between old and new
                    # instructions turn to turn. The rest of the saved
                    # history (if any) is left untouched.
                    history[0] = {"role": "system", "content": SYSTEM_PROMPT}
                    return history
            except Exception:
                pass
        return [{"role": "system", "content": SYSTEM_PROMPT}]

    def save_chat_history(self):
        full = self.permanent_memory + self.session_memory
        conv = full[1:]
        if len(conv) > 10: conv = conv[-10:]
        with open(MEMORY_FILE, "w") as f: 
            json.dump([full[0]] + conv, f, indent=4)

if __name__ == "__main__":
    print("--- SYSTEM STARTING ---", flush=True)
    root = tk.Tk()
    app = BotGUI(root)
    root.mainloop()
