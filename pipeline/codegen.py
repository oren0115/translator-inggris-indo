"""Code generator: menghasilkan 'kode target' berupa terjemahan via Gemini API (google-genai SDK)."""

from __future__ import annotations

import re
from dataclasses import dataclass

from google import genai

from .env_loader import gemini_api_key
from .semantic import SemanticResult

_DEFAULT_MODEL = "gemini-2.5-flash"


def _quota_error_notes(exc: BaseException, model_name: str) -> list[str]:
    """Pesan tambahan untuk 429 / RESOURCE_EXHAUSTED (kuota & rate limit)."""

    raw = str(exc)
    notes = [
        "Google membalas **429 RESOURCE_EXHAUSTED**: kuota atau batas permintaan untuk model ini habis / tidak tersedia pada paket Anda.",
        f"Model yang dipakai: `{model_name}`.",
        "Yang bisa dicoba: (1) tunggu beberapa menit lalu **jalankan ulang**; (2) pilih **model lain** di sidebar (mis. `gemini-2.5-flash` atau `gemini-1.5-flash`); "
        "(3) cek **penagihan / paket** dan kuota di [Google AI — rate limits](https://ai.google.dev/gemini-api/docs/rate-limits) dan [pemakaian](https://ai.dev/rate-limit).",
    ]
    m = re.search(r"retry in ([\d.]+)\s*s", raw, re.IGNORECASE)
    if m:
        notes.append(f"Saran server: coba lagi setelah sekitar **{m.group(1)}** detik.")
    return notes


@dataclass
class CodegenResult:
    ok: bool
    translated_text: str
    model: str
    notes: list[str]


class CodeGenerator:
    """
    Fase generasi: memetakan IR semantik + teks sumber ke string bahasa target.
    """

    def __init__(
        self,
        text: str,
        target_lang_label: str,
        semantic: SemanticResult,
        source_lang_code: str | None,
        model_name: str | None = None,
    ) -> None:
        self._text = text.strip()
        self._target = target_lang_label
        self._semantic = semantic
        self._source_code = source_lang_code
        self._model_override = (model_name or "").strip() or None

    def generate(self) -> CodegenResult:
        notes: list[str] = []
        api_key = gemini_api_key()
        if not api_key:
            notes.append(
                "Kunci API tidak ditemukan. Set salah satu: GEMINI_API_KEY, GOOGLE_API_KEY, "
                "atau GOOGLE_GENAI_API_KEY (environment atau file `.env` di folder proyek)."
            )
            return CodegenResult(
                ok=False,
                translated_text="",
                model="",
                notes=notes,
            )

        if not self._semantic.ok:
            notes.append("Generasi dibatalkan karena analisis semantik gagal.")
            return CodegenResult(ok=False, translated_text="", model="", notes=notes)

        if not self._semantic.input_language_valid:
            notes.append(
                "Generasi dibatalkan: **validasi bahasa** — teks harus berbahasa Indonesia atau Inggris "
                "dan harus sesuai dengan pilihan \"Bahasa sumber\" (lihat pesan di layar)."
            )
            return CodegenResult(ok=False, translated_text="", model="", notes=notes)

        src_hint = self._semantic.detected_lang or "unknown"
        if self._source_code and self._source_code != "auto":
            src_hint = self._source_code

        model_name = self._model_override or _DEFAULT_MODEL

        prompt = f"""You are a professional translator (compilation code generation phase).
Translate the following text into {self._target}.
The application only supports source content in **Indonesian** or **English**; if the input mixes or drifts outside, still produce the best {self._target} translation of the intended meaning.
Preserve meaning, tone, and formatting (line breaks) as much as possible.
Source language hint: {src_hint}
Context hint: {self._semantic.context_summary}

Text to translate:
---
{self._text}
---

Output ONLY the translated text, no explanations."""

        try:
            client = genai.Client(api_key=api_key)
            resp = client.models.generate_content(
                model=model_name,
                contents=prompt,
            )
            out = (getattr(resp, "text", None) or "").strip()
            if not out and getattr(resp, "candidates", None):
                notes.append("Model mengembalikan respons kosong; periksa kebijakan konten atau kuota.")
            return CodegenResult(ok=bool(out), translated_text=out, model=model_name, notes=notes)
        except Exception as e:
            err = str(e)
            notes.append(f"Error API: {e}")
            if "429" in err or "RESOURCE_EXHAUSTED" in err or "quota" in err.lower():
                notes.extend(_quota_error_notes(e, model_name))
            return CodegenResult(ok=False, translated_text="", model=model_name, notes=notes)
