"""
=============================================================
ANTARMUKA STREAMLIT — RAG UTS Data Engineering
=============================================================

Jalankan dengan: streamlit run ui/app.py
=============================================================
"""

import sys
import os
from pathlib import Path

# Agar bisa import dari folder src/
sys.path.append(str(Path(__file__).parent.parent / "src"))

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ─── Konfigurasi Halaman ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG System — UTS Data Engineering",
    page_icon="🤖",
    layout="wide"
)

# ─── Custom Theme ───────────────────────────────────────────────
st.markdown("""
<style>
/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #2b8be0 0%, #43a3f7 100%);
}
[data-testid="stSidebar"] * {
    color: #ffffff !important;
}
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stCheckbox label,
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown strong,
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4 {
    color: #ffffff !important;
}
[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.25) !important;
}
[data-testid="stSidebar"] [data-testid="stAlert"] {
    background-color: rgba(255,255,255,0.12) !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
}
[data-testid="stSidebar"] .stSlider [data-testid="stThumbValue"],
[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] div {
    color: #ffffff !important;
}

/* ── Main Area ── */
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.main .block-container {
    background-color: #ffffff !important;
}
[data-testid="stAppViewContainer"] h1,
[data-testid="stAppViewContainer"] h2,
[data-testid="stAppViewContainer"] h3 {
    color: #1a3a6b !important;
}
[data-testid="stAppViewContainer"] p,
[data-testid="stAppViewContainer"] span,
[data-testid="stAppViewContainer"] label,
[data-testid="stAppViewContainer"] .stMarkdown {
    color: #2c3e50 !important;
}
[data-testid="stAppViewContainer"] .stCaption p {
    color: #5a7baa !important;
}
[data-testid="stAppViewContainer"] hr {
    border-color: #d6e4f0 !important;
}

/* ── Header bar ── */
[data-testid="stHeader"] {
    background-color: #ffffff !important;
}

/* ── Tombol ── */
.stButton > button {
    background-color: #1e5099 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    transition: background-color 0.3s ease;
}
.stButton > button:hover {
    background-color: #1a3a6b !important;
}

/* ── Chat input ── */
[data-testid="stChatInput"] textarea {
    border-color: #1e5099 !important;
    color: #2c3e50 !important;
    background-color: #f0f5fc !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #1a3a6b !important;
    box-shadow: 0 0 0 2px rgba(30,80,153,0.2) !important;
}

/* ── Success / Error alerts di main ── */
[data-testid="stAppViewContainer"] [data-testid="stAlert"] {
    border-radius: 8px !important;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background-color: #f0f5fc !important;
    border: 1px solid #d6e4f0 !important;
    border-radius: 10px !important;
}

/* ── Expander ── */
[data-testid="stAppViewContainer"] .streamlit-expanderHeader {
    color: #1e5099 !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Header ───────────────────────────────────────────────────────────────────
st.title("Sistem Tanya-Jawab RAG")
st.caption("UTS Data Engineering — Retrieval-Augmented Generation")
st.divider()

# ─── Sidebar: Info & Konfigurasi ─────────────────────────────────────────────
with st.sidebar:
    st.header("Konfigurasi")
    
    top_k = st.slider(
        "Jumlah dokumen relevan (top-k)",
        min_value=1, max_value=10, value=3,
        help="Berapa banyak chunk yang diambil dari vector database"
    )
    
    show_context = st.checkbox("Tampilkan konteks yang digunakan", value=True)
    show_prompt = st.checkbox("Tampilkan prompt ke LLM", value=False)
    
    st.divider()
    st.header("Info Sistem")
    
    # TODO: Isi informasi kelompok kalian di sini
    st.markdown("""
    **Kelompok:** *(nama kelompok)*  
    **Domain:** *(domain dokumen)*  
    **LLM:** *(provider LLM)*  
    **Vector DB:** ChromaDB  
    **Embedding:** multilingual-MiniLM
    """)
    
    st.divider()
    st.info("💡 Tip: Mulai dengan pertanyaan spesifik yang jawabannya ada di dalam dokumen Anda.")


# ─── Load Vector Store (cached agar tidak reload setiap query) ───────────────
@st.cache_resource
def load_vs():
    """Load vector store sekali saja, di-cache untuk performa."""
    try:
        from query import load_vectorstore
        return load_vectorstore(), None
    except FileNotFoundError as e:
        return None, str(e)
    except Exception as e:
        return None, f"Error: {e}"


# ─── Main Content ──────────────────────────────────────────────────────────────
vectorstore, error = load_vs()

if error:
    st.error(f" {error}")
    st.info("Jalankan terlebih dahulu: `python src/indexing.py`")
    st.stop()

st.success("Vector database berhasil dimuat dan siap digunakan!")

# ─── Chat Interface ───────────────────────────────────────────────────────────
# Simpan riwayat chat di session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Tampilkan riwayat chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and show_context and "contexts" in msg:
            with st.expander("Konteks yang digunakan"):
                for i, ctx in enumerate(msg["contexts"], 1):
                    st.markdown(f"**[{i}] Skor: {ctx['score']:.4f}** | `{ctx['source']}`")
                    st.text(ctx["content"][:300] + "...")
                    st.divider()

# Input pertanyaan baru
if question := st.chat_input("Ketik pertanyaan Anda di sini..."):
    
    # Tampilkan pertanyaan user
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)
    
    # Generate jawaban
    with st.chat_message("assistant"):
        with st.spinner("Mencari informasi relevan dan menghasilkan jawaban..."):
            try:
                from query import answer_question
                result = answer_question(question, vectorstore)
                
                st.write(result["answer"])
                
                # Tampilkan konteks jika diaktifkan
                if show_context:
                    with st.expander("📚 Konteks yang digunakan"):
                        for i, ctx in enumerate(result["contexts"], 1):
                            st.markdown(f"**[{i}] Skor relevansi: {ctx['score']:.4f}** | `{ctx['source']}`")
                            st.text(ctx["content"][:300] + "...")
                            st.divider()
                
                # Tampilkan prompt jika diaktifkan
                if show_prompt:
                    with st.expander("🔧 Prompt yang dikirim ke LLM"):
                        st.code(result["prompt"], language="text")
                
                # Simpan ke riwayat
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "contexts": result["contexts"]
                })
                
            except Exception as e:
                error_msg = f"Error: {e}\n\nPastikan API key sudah diatur di file .env"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# ─── Tombol Reset Chat ────────────────────────────────────────────────────────
if st.session_state.messages:
    if st.button("Hapus Riwayat Chat"):
        st.session_state.messages = []
        st.rerun()
