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

# Mencari jalur folder utama (Project Root)
# Karena file ini ada di 'ui/app.py', parents[1] adalah root proyek
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]

# Masukkan folder root ke dalam urutan pertama sys.path
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Sekarang baru panggil library lainnya
import streamlit as st
from src.query import load_vectorstore, answer_question
from dotenv import load_dotenv

load_dotenv()

# ─── Konfigurasi Halaman ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Nara — Asisten Kesehatan Mental",
    page_icon="🌊",
    layout="wide"
)

# ─── Kustomisasi CSS (Aura Tenang Biru Pastel) ───────────────────────────────
st.markdown("""
<style>
    /* Latar belakang utama (Biru sangat pucat) */
    .stApp {
        background-color: #F4F9FD;
    }
    
    /* Warna Sidebar (Biru pastel kalem) */
    [data-testid="stSidebar"] {
        background-color: #E8F1F8 !important;
        border-right: 2px solid #D6E4F0;
    }
    
    /* Warna teks (Biru gelap keabu-abuan agar nyaman dibaca) */
    h1, h2, h3, h4, p, span, div {
        color: #2C3E50;
    }
    
    /* Warna tombol utama */
    .stButton>button {
        background-color: #8CB8D9;
        color: white;
        border: none;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #72A3C9;
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
        border: 1px solid #E6F0F9;
    }
    
    /* Mempercantik kotak input chat di bawah */
    .stChatInputContainer {
        border: 1px solid #8CB8D9 !important;
        border-radius: 12px !important;
        background-color: #FFFFFF !important;
    }
</style>
""", unsafe_allow_html=True)


# ─── Header ───────────────────────────────────────────────────────────────────
st.title("🌊 Nara")
st.caption("Ruang aman untuk bertanya, bercerita, dan memahami kesehatan mentalmu.")
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
    
    st.markdown("""
    **Kelompok:** Ardhana & Tim  
    **Domain:** Kesehatan Mental (PFA)  
    **LLM:** Groq (Llama-3.3-70b-versatile)  
    **Vector DB:** FAISS  
    **Embedding:** multilingual-MiniLM
    """)
    
    st.divider()
    st.info("💡 Tip: Mulai dengan menceritakan apa yang sedang kamu rasakan hari ini kepada Nara.")


# ─── Load Vector Store (cached agar tidak reload setiap query) ───────────────
@st.cache_resource
def load_vs():
    """Load vector store sekali saja, di-cache untuk performa."""
    try:
        from src.query import load_vectorstore
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

# ─── Chat Interface ───────────────────────────────────────────────────────────
# Simpan riwayat chat di session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Halo! Aku Nara. Aku di sini siap mendengarkan cerita dan keluh kesahmu. Ada yang ingin dibagikan hari ini?"}
    ]

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
if question := st.chat_input("Ketik pesan untuk Nara di sini..."):
    
    # Tampilkan pertanyaan user
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)
    
    # Generate jawaban
    with st.chat_message("assistant"):
        with st.spinner("Nara sedang merangkai kata..."):
            try:
                from src.query import answer_question
                # Kirim parameter top_k dari slider ke fungsi RAG
                result = answer_question(question, vectorstore, k=top_k)
                
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