"""
Media Translator — demonstrasi pipeline kompilasi (Lexer → Parser → Semantic → Codegen)
menggunakan Streamlit dan Gemini API untuk fase generasi.
"""

from __future__ import annotations

import os
import re
from typing import Any

import streamlit as st
from streamlit.errors import StreamlitAPIException

from pipeline import CodeGenerator, Lexer, Parser, SemanticAnalyzer
from pipeline.env_loader import ensure_env_loaded, gemini_api_key


def _hydrate_env_from_secrets() -> None:
    """Salin kunci dari `.streamlit/secrets.toml` ke environment bila belum ada."""

    try:
        sec = getattr(st, "secrets", None)
        if sec is None:
            return
        for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GEMINI_MODEL"):
            if k in sec and not os.environ.get(k):
                os.environ[k] = str(sec[k])
    except (RuntimeError, FileNotFoundError, KeyError):
        return

SOURCE_LANG = {
    "id": "Indonesia",
    "en": "Inggris",
}

TARGET_LANG = {"id": "Indonesia", "en": "Inggris"}
MAX_INPUT_WORDS = 50


def _init_session() -> None:
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None


def _effective_gemini_model(ui_choice: str) -> str:
    """Prioritas: `GEMINI_MODEL` di environment / `.env`, lalu pilihan UI."""

    return (os.environ.get("GEMINI_MODEL") or "").strip() or ui_choice


def _count_words(text: str) -> int:
    """Hitung jumlah kata non-kosong untuk validasi input."""

    return len(re.findall(r"\S+", text.strip()))


def _run_pipeline(text: str, src: str, tgt_code: str, gemini_model: str) -> dict[str, Any]:
    lexer = Lexer(text)
    tokens = lexer.tokenize()
    sig = Lexer.significant_tokens(tokens)

    parser = Parser(sig)
    tree = parser.parse()

    sem = SemanticAnalyzer(text, src, tree)
    semantic = sem.analyze()

    tgt_label = TARGET_LANG.get(tgt_code, tgt_code)
    gen = CodeGenerator(
        text,
        tgt_label,
        semantic,
        src,
        model_name=gemini_model,
    )
    codegen = gen.generate()

    return {
        "tokens": tokens,
        "significant_tokens": sig,
        "parse_tree": tree,
        "parse_tree_str": Parser.tree_to_str(tree),
        "semantic": semantic,
        "codegen": codegen,
    }


def _inside_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


def main() -> None:
    try:
        st.set_page_config(
            page_title="Translator Sederhana",
            page_icon="📦",
            layout="wide",
            initial_sidebar_state="expanded",
        )
    except StreamlitAPIException:
        pass
    ensure_env_loaded()
    _hydrate_env_from_secrets()
    _init_session()

    st.title("Translator Sederhana")
    st.caption(
        "Mini project teknik kompilasi"
    )

    with st.sidebar:
        st.header("Konfigurasi")
        st.markdown(
            "Gunakan variabel environment **GEMINI_API_KEY**"
        )
        key_ok = bool(gemini_api_key())
        st.caption("Status API: **tersedia**" if key_ok else "Status API: **belum diset**")
        env_model = (os.environ.get("GEMINI_MODEL") or "").strip()
        if env_model:
            st.caption(f"Model aktif: **`{env_model}`** (dari `GEMINI_MODEL`)")
        gemini_ui_model = st.selectbox(
            "Model Gemini",
            options=["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"],
            index=0,
            disabled=bool(env_model),
            help="Jika muncul error 429 pada satu model, coba model lain atau tunggu beberapa menit.",
        )

    inp, cfg = st.columns((3, 2), gap="large")

    with inp:
        text = st.text_area(
            "Teks sumber",
            height=220,
            placeholder="Masukkan teks yang akan diterjemahkan…",
        )
        word_count = _count_words(text)
        st.caption(f"Jumlah kata: **{word_count}/{MAX_INPUT_WORDS}**")
    with cfg:
        src = st.selectbox("Bahasa sumber", options=list(SOURCE_LANG.keys()), format_func=lambda k: SOURCE_LANG[k])
        tgt = st.selectbox(
            "Bahasa target",
            options=list(TARGET_LANG.keys()),
            format_func=lambda k: TARGET_LANG[k],
        )

    run = st.button("Jalankan pipeline kompilasi", type="primary", use_container_width=True)

    if run:
        if not text.strip():
            st.error("Masukkan teks terlebih dahulu.")
        elif word_count > MAX_INPUT_WORDS:
            st.error(f"Input melebihi batas. Maksimal {MAX_INPUT_WORDS} kata.")
        elif src == tgt:
            st.error("Bahasa sumber dan target tidak boleh sama.")
        else:
            st.session_state.pipeline = _run_pipeline(
                text,
                src,
                tgt,
                _effective_gemini_model(gemini_ui_model),
            )

    pipeline = st.session_state.pipeline

    if pipeline is None:
        st.info("Klik **Jalankan pipeline kompilasi** untuk menerjemahkan.")
    else:
        st.divider()
        sem = pipeline["semantic"]
        if not sem.input_language_valid:
            st.error("Validasi bahasa: input ditolak. Hanya teks **Indonesia** atau **Inggris** yang sesuai pilihan bahasa sumber.")
            for m in sem.messages:
                st.caption(m)
        g = pipeline["codegen"]
        if g.ok and g.translated_text:
            st.success("Terjemahan")
            st.write(g.translated_text)
        else:
            st.warning("Terjemahan belum tersedia atau gagal.")
        st.caption(f"Model: `{g.model or '—'}`")
        if g.notes:
            for n in g.notes:
                st.caption(n)

        with st.expander("Detail tahapan kompilasi (lexer, parser, semantic)", expanded=False):
            st.markdown("**Lexer** — aliran token")
            tok = pipeline["tokens"]
            st.caption(f"Jumlah token: {len(tok)}")
            rows = [
                {"#": i + 1, "jenis": t.type.name, "nilai": t.value, "baris": t.line, "kolom": t.column}
                for i, t in enumerate(tok)
            ]
            st.dataframe(rows, use_container_width=True, height=240)
            with st.expander("Token signifikan (tanpa whitespace)"):
                st.code("\n".join(repr(t) for t in pipeline["significant_tokens"]), language="text")

            st.markdown("**Parser** — pohon sintaks (NP / VP)")
            st.code(pipeline["parse_tree_str"], language="text")

            st.markdown("**Semantic** — analisis makna")
            s = pipeline["semantic"]
            st.json(
                {
                    "ok": s.ok,
                    "content_lang_detected": s.content_lang_detected,
                    "input_language_valid": s.input_language_valid,
                    "detected_lang": s.detected_lang,
                    "detected_label": s.detected_lang_label,
                    "confidence": s.confidence,
                    "context_summary": s.context_summary,
                    "structure_hint": s.structure_hint,
                    "messages": s.messages,
                },
                expanded=False,
            )

    with st.expander("Arsitektur singkat"):
        st.markdown(
            """
1. **Lexer** memecah input menjadi token bertipe (kata, angka, tanda baca, baris baru).
2. **Parser** membangun pohon **S** dengan anak **NP** / **VP** berdasarkan heuristik kata kerja.
3. **Semantic** mendeteksi bahasa (`langdetect`), ringkasan konteks, dan validasi input.
4. **Code Generator** memanggil **Gemini** untuk menghasilkan teks target (**Indonesia** atau **Inggris**).
"""
        )


if __name__ == "__main__":
    import subprocess
    import sys

    if _inside_streamlit():
        main()
    else:
        raise SystemExit(
            subprocess.call([sys.executable, "-m", "streamlit", "run", __file__, *sys.argv[1:]])
        )
