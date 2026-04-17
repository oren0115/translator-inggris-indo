"""Semantic analyzer: deteksi bahasa, konteks, validasi input (analog type checking)."""

from __future__ import annotations

from dataclasses import dataclass

from langdetect import DetectorFactory, LangDetectException, detect, detect_langs

from .parser import ParseNode


DetectorFactory.seed = 42


_LANG_NAMES = {
    "id": "Indonesia",
    "en": "Inggris",
}

_SUPPORTED_APP_LANGS = frozenset({"id", "en"})

# langdetect sering salah: Indonesia terbaca `tl` (Tagalog), `ms`, `jv`, `su` (kerabat Austronesia).
_ID_CONFUSABLE = frozenset({"tl", "ms", "jv", "su", "ceb"})

_MIN_ID_SCORE = 0.12
_MIN_EN_SCORE = 0.12


def _lang_prob(langs: list, code: str) -> float:
    for x in langs:
        if x.lang == code:
            return float(x.prob)
    return 0.0


def _heuristic_indonesian(
    content_lang: str | None,
    langs: list,
) -> tuple[bool, str]:
    """True jika teks layak diperlakukan sebagai Indonesia (bukan hanya kode `id`)."""

    if not content_lang or not langs:
        return False, ""
    id_p = _lang_prob(langs, "id")
    if content_lang == "id":
        return True, ""
    if id_p >= _MIN_ID_SCORE:
        return True, (
            f"Heuristik validasi: bahasa teratas `{content_lang}`, tetapi skor **id** = {id_p:.2f} "
            "(cukup untuk menerima sebagai Indonesia)."
        )
    if content_lang in _ID_CONFUSABLE:
        return True, (
            f"Heuristik validasi: kode `{content_lang}` sering tertukar dengan Indonesia oleh detektor; "
            "diterima karena Anda memilih **Bahasa sumber: Indonesia**."
        )
    return False, ""


def _heuristic_english(content_lang: str | None, langs: list) -> tuple[bool, str]:
    if not content_lang or not langs:
        return False, ""
    en_p = _lang_prob(langs, "en")
    if content_lang == "en":
        return True, ""
    if en_p >= _MIN_EN_SCORE:
        return True, (
            f"Heuristik validasi: bahasa teratas `{content_lang}`, tetapi skor **en** = {en_p:.2f} "
            "(cukup untuk menerima sebagai Inggris)."
        )
    return False, ""


@dataclass
class SemanticResult:
    ok: bool
    messages: list[str]
    detected_lang: str | None
    detected_lang_label: str | None
    confidence: float | None
    context_summary: str
    structure_hint: str
    # Bahasa isi teks (langdetect); input_language_valid = teks ID/EN dan cocok dengan sumber.
    content_lang_detected: str | None
    input_language_valid: bool


class SemanticAnalyzer:
    """
    Memvalidasi input dan menarik informasi semantik tingkat tinggi.
    """

    def __init__(self, text: str, source_lang_code: str | None, root: ParseNode) -> None:
        self._text = text.strip()
        self._source_lang = (source_lang_code or "").strip().lower() or None
        self._root = root

    def analyze(self) -> SemanticResult:
        messages: list[str] = []

        if not self._text:
            return SemanticResult(
                ok=False,
                messages=["Input kosong: tidak ada teks untuk dianalisis."],
                detected_lang=None,
                detected_lang_label=None,
                confidence=None,
                context_summary="",
                structure_hint="",
                content_lang_detected=None,
                input_language_valid=False,
            )

        if len(self._text) > 8000:
            messages.append("Peringatan: teks sangat panjang; terjemahan mungkin dipotong oleh model.")

        content_lang: str | None = None
        confidence: float | None = None
        langs: list = []
        try:
            langs = detect_langs(self._text)
            top = langs[0] if langs else None
            confidence = float(top.prob) if top else None
            content_lang = detect(self._text)
            messages.append(
                f"Bahasa isi teks (langdetect): {_LANG_NAMES.get(content_lang, content_lang)} (kode: {content_lang})."
            )
            if langs:
                topn = ", ".join(f"{x.lang}:{x.prob:.2f}" for x in langs[:5])
                messages.append(f"Skor teratas (langdetect): {topn}")
        except LangDetectException:
            langs = []
            content_lang = None
            messages.append("Deteksi bahasa isi teks gagal (terlalu pendek atau ambigu).")

        input_language_valid = False
        src = self._source_lang

        if content_lang is None:
            messages.append(
                "Validasi gagal: tidak dapat memastikan bahasa. Tulis kalimat yang jelas dalam **Indonesia** atau **Inggris**."
            )
        elif not langs:
            # detect() ada hasil tapi daftar skor kosong — jarang; pakai aturan longgar.
            if src == "id" and content_lang in _SUPPORTED_APP_LANGS | _ID_CONFUSABLE:
                input_language_valid = True
                messages.append("Catatan: skor bahasa tidak tersedia; validasi mengandalkan kode bahasa teratas.")
            elif src == "en" and content_lang in _SUPPORTED_APP_LANGS:
                input_language_valid = True
                messages.append("Catatan: skor bahasa tidak tersedia; validasi mengandalkan kode bahasa teratas.")
            else:
                messages.append("Validasi gagal: data skor bahasa tidak mencukupi.")
        elif src == "id":
            ok_id, hint = _heuristic_indonesian(content_lang, langs)
            if ok_id:
                input_language_valid = True
                if hint:
                    messages.append(hint)
                elif content_lang == "id":
                    messages.append("Validasi bahasa OK: isi diperlakukan sebagai **Indonesia**.")
            elif content_lang == "en" and _lang_prob(langs, "en") >= 0.35 and _lang_prob(langs, "id") < _MIN_ID_SCORE:
                messages.append(
                    "**Input ditolak:** teks terlihat seperti **Inggris**, sedangkan bahasa sumber **Indonesia**. "
                    "Ubah pilihan bahasa sumber ke Inggris atau tulis dalam bahasa Indonesia."
                )
            else:
                messages.append(
                    f"**Input ditolak:** tidak cukup bukti bahasa Indonesia (kode `{content_lang}`). "
                    "Aplikasi hanya menerjemahkan dari teks berbahasa Indonesia atau Inggris sesuai pilihan."
                )
        elif src == "en":
            ok_en, hint = _heuristic_english(content_lang, langs)
            if ok_en:
                input_language_valid = True
                if hint:
                    messages.append(hint)
                elif content_lang == "en":
                    messages.append("Validasi bahasa OK: isi diperlakukan sebagai **Inggris**.")
            elif content_lang == "id" and _lang_prob(langs, "id") >= 0.35 and _lang_prob(langs, "en") < _MIN_EN_SCORE:
                messages.append(
                    "**Input ditolak:** teks terlihat seperti **Indonesia**, sedangkan bahasa sumber **Inggris**. "
                    "Ubah pilihan bahasa sumber atau tulis dalam bahasa Inggris."
                )
            else:
                messages.append(
                    f"**Input ditolak:** tidak cukup bukti bahasa Inggris (kode `{content_lang}`). "
                    "Aplikasi hanya menerjemahkan dari teks berbahasa Indonesia atau Inggris sesuai pilihan."
                )
        else:
            messages.append("Bahasa sumber tidak dikenal; pilih Indonesia atau Inggris.")

        detected_lang = content_lang if content_lang in _SUPPORTED_APP_LANGS else content_lang
        detected_label = _LANG_NAMES.get(detected_lang, detected_lang) if detected_lang else None

        structure_hint = self._summarize_tree(self._root)
        context_summary = self._infer_context(self._text)

        return SemanticResult(
            ok=True,
            messages=messages,
            detected_lang=detected_lang,
            detected_lang_label=detected_label,
            confidence=confidence,
            context_summary=context_summary,
            structure_hint=structure_hint,
            content_lang_detected=content_lang,
            input_language_valid=input_language_valid,
        )

    def _summarize_tree(self, node: ParseNode) -> str:
        parts: list[str] = []

        def walk(n: ParseNode) -> None:
            if n.label in ("NP", "VP", "PUNCT") and n.children:
                snippet = n.leaves_text()[:60]
                if snippet.strip():
                    parts.append(f"{n.label}: «{snippet.strip()}»")
            for c in n.children:
                walk(c)

        walk(node)
        if not parts:
            return "Struktur frasa: tidak terbentuk NP/VP jelas (heuristik)."
        return "Struktur frasa (heuristik): " + "; ".join(parts[:8])

    def _infer_context(self, text: str) -> str:
        t = text.lower()
        if any(w in t for w in ("error", "exception", "compile", "syntax", "parser", "lexer")):
            return "Konteks perkiraan: teknis / pemrograman / kompilasi."
        if any(w in t for w in ("pasien", "dokter", "obat", "rumah sakit")):
            return "Konteks perkiraan: medis / kesehatan."
        if any(w in t for w in ("jual", "beli", "harga", "diskon", "produk")):
            return "Konteks perkiraan: komersial / e-commerce."
        if any(w in t for w in ("cinta", "sayang", "rindu", "hati")):
            return "Konteks perkiraan: emosional / interpersonal."
        return "Konteks perkiraan: umum / non-spesifik."
