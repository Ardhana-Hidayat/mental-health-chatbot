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

# --- Path Configuration ---
# Ensure the 'src' directory and project root are in the Python path
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
from dotenv import load_dotenv

from src.query import load_vectorstore, answer_question

load_dotenv()

# ─── Konfigurasi Halaman ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Teman Cerita - Asisten Kesehatan Mental",
    page_icon="🌱",
    layout="wide"
)

# ─── Kustomisasi CSS (Aura Tenang & Ramah) ───────────────────────────────────
st.markdown("""
<style>
    /* Mengubah warna teks utama dan latar belakang agar lebih lembut */
    .stApp {
        background-color: #F8FBF8;
    }
    
    /* Membuat warna sidebar lebih menonjol tapi tetap kalem */
    [data-testid="stSidebar"] {
        background-color: #E8F2E8 !important;
        border-right: 2px solid #D5E5D5;
    }
    
    h1, h2, h3, h4, p, span, div {
        color: #2F4F4F;
    }
    
    /* Warna tombol utama */
    .stButton>button {
        background-color: #A3C9A8;
        color: white;
        border: none;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #84B589;
        color: white;
        box-shadow: 0px 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Warna area chat pengguna */
    [data-testid="stChatMessage"] {
        background-color: #FFFFFF;
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.03);
        margin-bottom: 15px;
        border: 1px solid #F0F5F0;
    }
    
    /* Mempercantik kotak input chat di bawah */
    .stChatInputContainer {
        border: 1px solid #A3C9A8 !important;
        border-radius: 12px !important;
        background-color: #FFFFFF !important;
    }
</style>
""", unsafe_allow_html=True)


# ─── Header ───────────────────────────────────────────────────────────────────
st.title("🌱 Teman Cerita")
st.caption("Ruang aman untuk bertanya, bercerita, dan memahami kesehatan mentalmu.")
st.divider()

# ─── Sidebar: Info & Konfigurasi ─────────────────────────────────────────────
with st.sidebar:
    st.header("Pengaturan Sesi")
    
    top_k = st.slider(
        "Tingkat Kedalaman Memori",
        min_value=1, max_value=10, value=3,
        help="Semakin tinggi, semakin banyak konteks masa lalu/referensi yang diingat."
    )
    
    show_context = st.checkbox("Tampilkan sumber referensi balasan", value=True)
    show_prompt = st.checkbox("Mode Pengembang (Tampilkan Prompt)", value=False)
    
    st.divider()
    st.header("Tentang Sistem Ini")
    
    st.markdown(f"""
    **Kelompok:** Ardhana & Tim
    **Fokus:** Dukungan Psikologis Awal (PFA) & Edukasi Mental
    **Teknologi Utama:** RAG Pipeline (React/Go API - Opsional)
    **Mesin Pemikir:** ChromaDB + multilingual-MiniLM
    """)
    
    st.divider()
    st.info("💚 **Peringatan:** Sistem ini dibuat untuk tujuan edukasi dan dukungan awal, bukan pengganti diagnosis medis profesional. Jika kamu merasa sangat kewalahan, mohon hubungi profesional terdekat.")


# ─── Load Vector Store (cached agar tidak reload setiap query) ───────────────
@st.cache_resource
def load_vs():
    """Load vector store sekali saja, di-cache untuk performa."""
    try:
        return load_vectorstore(), None
    except FileNotFoundError as e:
        return None, str(e)
    except Exception as e:
        return None, f"Error: {e}"


# ─── Main Content ──────────────────────────────────────────────────────────────
vectorstore, error = load_vs()

if error:
    st.error(f"Sepertinya memori sistem belum siap: {error}")
    st.info("🔧 Mohon jalankan `python src/indexing.py` terlebih dahulu di terminalmu.")
    st.stop()

# ─── Chat Interface ───────────────────────────────────────────────────────────
# Simpan riwayat chat di session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Halo! Aku di sini untuk mendengarkan. Ada yang ingin kamu ceritakan atau tanyakan hari ini?"}
    ]

# Tampilkan riwayat chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🌱" if msg["role"] == "assistant" else "👤"):
        st.write(msg["content"])
        if msg["role"] == "assistant" and show_context and "contexts" in msg:
            with st.expander("Buku Catatanku (Referensi Jawaban)"):
                for i, ctx in enumerate(msg["contexts"], 1):
                    st.markdown(f"**💡 Insight {i} (Relevansi: {ctx['score']:.2f})**")
                    st.text(ctx["content"][:200] + "...")
                    st.divider()

# Input pertanyaan baru
if question := st.chat_input("Ketik di sini apa yang sedang kamu rasakan..."):
    
    # Tampilkan pertanyaan user
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="👤"):
        st.write(question)
    
    # Generate jawaban
    with st.chat_message("assistant", avatar="🌱"):
        with st.spinner("Membaca dan merangkai kata untukmu..."):
            try:
                # Pastikan fungsi answer_question menerima parameter top_k
                result = answer_question(question, vectorstore, k=top_k)
                
                st.write(result["answer"])
                
                # Tampilkan konteks jika diaktifkan
                if show_context and "contexts" in result:
                    with st.expander("📖 Referensi Pustaka"):
                        for i, ctx in enumerate(result["contexts"], 1):
                            st.markdown(f"**Catatan {i} (Akurasi: {ctx['score']:.2f})** | *{ctx['source']}*")
                            st.text(ctx["content"][:200] + "...")
                            st.divider()
                
                # Tampilkan prompt jika diaktifkan
                if show_prompt and "prompt" in result:
                    with st.expander("🔧 Log Pengembang"):
                        st.code(result["prompt"], language="text")
                
                # Simpan ke riwayat
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "contexts": result.get("contexts", [])
                })
                
            except Exception as e:
                error_msg = f"Maaf, sepertinya aku sedang kesulitan berpikir. (Error: {e})\n\nPastikan konfigurasi API sudah benar."
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# ─── Tombol Reset Chat ────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if len(st.session_state.messages) > 1: # Tampilkan tombol jika ada obrolan selain sapaan awal
        if st.button("Mulai Cerita Baru 🍃", use_container_width=True):
            st.session_state.messages = [
                {"role": "assistant", "content": "Halo! Aku di sini untuk mendengarkan. Ada yang ingin kamu ceritakan atau tanyakan hari ini?"}
            ]
            st.rerun()