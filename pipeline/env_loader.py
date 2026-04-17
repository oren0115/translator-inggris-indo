"""Memuat variabel dari file `.env`"""

from __future__ import annotations

import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


def ensure_env_loaded() -> None:
    """Muat `.env` / `.env.local` di root proyek (tidak menimpa variabel yang sudah ada di OS)."""

    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(_ROOT / ".env")
    load_dotenv(_ROOT / ".env.local")


def gemini_api_key() -> str | None:
    """Kunci API untuk Gemini / google-genai (nama variabel yang didukung)."""

    ensure_env_loaded()
    for name in (
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "GOOGLE_GENAI_API_KEY",
    ):
        raw = os.environ.get(name)
        if raw is not None and str(raw).strip():
            return str(raw).strip()
    return None
