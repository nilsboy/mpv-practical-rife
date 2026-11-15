"""Shared configuration for mpv-practical-rife.

Settings can come from (later wins):
 1. defaults
 2. environment variables prefixed ``MPV_RIFE_``
 3. a JSON string passed as ``user_data`` (mpv ``vf_vapoursynth`` user-data
    suboption; may be base64 ``b64:<...>``)
 4. explicit function arguments

Keys (short and long forms both accepted, JSON uses short forms):
    model       m : path to the ONNX model
    backend     b : ort | ov | auto
    device      d : runtime device ("" = auto), e.g. "GPU", "CPU", "CUDA"
    threads     t : inference threads for the CPU runtime (0 = auto)
    factor_num  f : fps multiplying factor numerator (e.g. 2 for 2x)
    factor_den  g : fps multiplying factor denominator (e.g. 1)
    sc          s : scene-change threshold [0..1]; 0 disables (default 0.15)
    pad         p : pad both dimensions to a multiple of this (default 128)
    matrix      x : colorspace matrix for YUV->RGB ("709", "470bg", ...)
    vs_threads  v : VapourSynth core threads (default 2)
    fps_num/fps_den : source fps override (rarely needed)
"""

from __future__ import annotations

import base64
import json
import os

DEFAULTS = {
    "model": "",
    "backend": "auto",
    "device": "",
    "threads": 0,
    "factor_num": 2,
    "factor_den": 1,
    "sc": 0.15,
    "pad": 128,
    "matrix": "709",
    "vs_threads": 2,
    "fps_num": 0,
    "fps_den": 0,
}

KEY_ALIASES = {
    "m": "model",
    "b": "backend",
    "d": "device",
    "t": "threads",
    "f": "factor_num",
    "g": "factor_den",
    "s": "sc",
    "p": "pad",
    "x": "matrix",
    "v": "vs_threads",
    "fn": "fps_num",
    "fd": "fps_den",
}

_INT_KEYS = {"threads", "factor_num", "factor_den", "pad", "vs_threads", "fps_num", "fps_den"}
_FLOAT_KEYS = {"sc"}


def _cast(key: str, value):
    if value is None:
        return None
    if key in _INT_KEYS:
        return int(value)
    if key in _FLOAT_KEYS:
        return float(value)
    return str(value)


def _apply(settings: dict[str, object], raw: dict[str, object]) -> None:
    for key, value in raw.items():
        key = str(key)
        if key.startswith("MPV_RIFE_"):
            key = key[len("MPV_RIFE_") :].lower()
            if key == "onnx":
                key = "model"
        key = KEY_ALIASES.get(key, key)
        if key not in DEFAULTS:
            continue
        value = _cast(key, value)
        if value is not None and value != "":
            settings[key] = value


def decode_user_data(user_data: str | None) -> dict:
    """Decode mpv's vf user-data payload.

    Plain JSON (``{"model": ...}``) or base64-encoded JSON (anything not
    starting with ``{``, produced by mpv/rife.lua — colons are not allowed
    inside mpv filter suboptions).
    """
    if not user_data:
        return {}
    prefix = user_data.lstrip()
    if not prefix.startswith("{"):
        user_data = base64.b64decode(
            user_data + "=" * (-len(user_data) % 4)
        ).decode("utf-8")
    data = json.loads(user_data)
    if not isinstance(data, dict):
        raise ValueError("user data must be a JSON object")
    return data


def load_settings(
    user_data: str | None = None,
    overrides: dict | None = None,
) -> dict:
    settings: dict[str, object] = dict(DEFAULTS)
    _apply(settings, os.environ)
    if user_data:
        _apply(settings, decode_user_data(user_data))
    if overrides:
        _apply(settings, overrides)
    if settings["model"] == "":
        default_model = os.environ.get("MPV_RIFE_DEFAULT_MODEL")
        if default_model:
            settings["model"] = default_model
    return settings
