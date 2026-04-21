"""
=============================================================
PIPELINE QUERY — RAG UTS Data Engineering
=============================================================

Pipeline ini dijalankan setiap kali user mengajukan pertanyaan:
1. Ubah pertanyaan user ke vektor (query embedding)
2. Cari chunk paling relevan dari vector database (retrieval)
3. Gabungkan konteks + pertanyaan ke dalam prompt
4. Kirim ke LLM untuk mendapatkan jawaban

Jalankan CLI dengan: python src/query.py
=============================================================
"""

import os
import warnings
from pathlib import Path
from dotenv import load_dotenv

# Membersihkan log terminal dari peringatan library
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
warnings.filterwarnings("ignore")

load_dotenv()

CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
TOP_K         = int(os.getenv("TOP_K", 3))
VS_DIR        = Path(os.getenv("VECTORSTORE_DIR", "./vectorstore"))
LLM_MODEL     = os.getenv("LLM_MODEL_NAME", "llama-3.3-70b-versatile")


# =============================================================
# TODO MAHASISWA:
# Pilih implementasi yang sesuai dengan pilihan LLM kalian
# =============================================================


def load_vectorstore():
    """Memuat vector database yang sudah dibuat oleh indexing.py"""
    import json
    import faiss
    from sentence_transformers import SentenceTransformer

    path_faiss = VS_DIR / "index.faiss"
    path_json = VS_DIR / "chunks.json"

    if not path_faiss.exists() or not path_json.exists():
        raise FileNotFoundError(
            f"Vector store tidak ditemukan di '{VS_DIR}'.\n"
            "Jalankan dulu: python src/indexing.py"
        )

    # Load FAISS index
    index = faiss.read_index(str(path_faiss))
    
    # Load JSON chunks
    with open(path_json, "r", encoding="utf-8") as f:
        chunks = json.load(f)
        
    # Load model embedding
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    vectorstore = {
        "index": index,
        "chunks": chunks,
        "model": model
    }
    return vectorstore


def retrieve_context(vectorstore, question: str, top_k: int = TOP_K) -> list:
    """
    LANGKAH 1 & 2: Query embedding + Similarity search.
    
    Fungsi ini:
    - Mengubah pertanyaan ke vektor
    - Mencari top_k chunk paling relevan
    - Mengembalikan list dokumen relevan
    """
    index = vectorstore["index"]
    chunks = vectorstore["chunks"]
    model = vectorstore["model"]
    
    # Ubah pertanyaan ke vektor
    q_vec = model.encode([question]).astype("float32")
    
    # Mencari top_k chunk paling relevan di FAISS
    distances, indices = index.search(q_vec, top_k)
    
    contexts = []
    for i, idx in enumerate(indices[0]):
        if idx != -1 and idx < len(chunks):
            chunk_data = chunks[idx]
            contexts.append({
                "content": chunk_data["content"],
                "source": chunk_data.get("source", "unknown"),
                "score": round(float(distances[0][i]), 4)
            })
    
    return contexts


def build_prompt(question: str, contexts: list) -> str:
    """
    LANGKAH 3: Membangun prompt untuk LLM.
    
    Prompt yang baik untuk RAG harus:
    - Memberikan instruksi jelas ke LLM
    - Menyertakan konteks yang sudah diambil
    - Menanyakan pertanyaan user
    - Meminta LLM untuk jujur jika tidak tahu
    
    TODO: Modifikasi prompt ini sesuai domain dan bahasa proyek kalian!
    """
    context_text = "\n\n---\n\n".join(
        [f"[Sumber: {c['source']}]\n{c['content']}" for c in contexts]
    )

    prompt = f"""Kamu adalah teman ngobrol sekaligus asisten kesehatan mental bernama "Nara" yang hangat, suportif, dan penuh empati. Posisikan dirimu seperti sahabat atau konselor yang selalu siap mendengarkan dengan peka, memvalidasi perasaan, dan pastinya tidak pernah menghakimi.

PANDUAN MENJAWAB:
1. Gunakan bahasa Indonesia yang santai, ramah, dan mengalir seperti sedang ngobrol santai, tapi tetap sopan dan jelas dipahami.
2. Selalu validasi dulu perasaan pengguna biar mereka merasa didengar. Hindari nada menggurui atau toxic positivity (seperti "jangan sedih dong" atau "semua pasti baik-baik saja").
3. Jawab pertanyaan MURNI berdasarkan informasi dari kolom KONTEKS di bawah ini. Jangan pernah mengarang informasi atau menambahkan opini pribadi di luar konteks.
4. Kalau informasinya memang tidak ada di dalam konteks, jujurlah dengan cara yang lembut. Misalnya: "Maaf banget ya, aku belum punya info soal itu di catatanku..." lalu arahkan mereka dengan sopan untuk bercerita ke tenaga profesional (psikolog/konselor).
5. Ingat, kamu adalah AI pendamping, bukan dokter. Jangan pernah memberikan diagnosis medis atau resep penanganan klinis.
6. Panggil dirimu sendiri dengan "Aku" atau namamu sendiri sesuai konteks ataupun yang menurutmu waktunya cocok, jangan sebut dirimu dengan "Saya" supaya terkesan tidak kaku.

KONTEKS:
{context_text}

PERTANYAAN:
{question}

JAWABAN:"""
    
    return prompt


# ─────────────────────────────────────────────────────────────
# OPSI LLM A: Groq (gratis, cepat) — REKOMENDASI
# ─────────────────────────────────────────────────────────────
def get_answer_groq(prompt: str) -> str:
    """Menggunakan Groq API (gratis, sangat cepat)."""
    from groq import Groq
    
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model=LLM_MODEL,  # "llama-3.3-70b-versatile"
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,   # Disesuaikan agar respons lebih hangat dan empatik
        max_tokens=1024
    )
    return response.choices[0].message.content


# ─────────────────────────────────────────────────────────────
# OPSI LLM B: Google Gemini (gratis tier)
# ─────────────────────────────────────────────────────────────
# def get_answer_gemini(prompt: str) -> str:
#     import google.generativeai as genai
#     genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
#     model = genai.GenerativeModel("gemini-1.5-flash")
#     response = model.generate_content(prompt)
#     return response.text


# ─────────────────────────────────────────────────────────────
# OPSI LLM C: Ollama (100% offline, gratis)
# Pastikan Ollama sudah diinstall dan model sudah di-pull:
# ollama pull llama3
# ─────────────────────────────────────────────────────────────
# def get_answer_ollama(prompt: str) -> str:
#     import requests
#     response = requests.post(
#         "http://localhost:11434/api/generate",
#         json={"model": "llama3", "prompt": prompt, "stream": False}
#     )
#     return response.json()["response"]


def answer_question(question: str, vectorstore=None, k: int = TOP_K) -> dict:
    """
    Fungsi utama: menerima pertanyaan, mengembalikan jawaban + konteks.
    
    Returns:
        dict dengan keys: answer, contexts, prompt
    """
    if vectorstore is None:
        vectorstore = load_vectorstore()
    
    # Retrieve
    print(f"🔍 Mencari konteks relevan untuk: '{question}'")
    contexts = retrieve_context(vectorstore, question, top_k=k)
    print(f"   ✅ {len(contexts)} chunk relevan ditemukan")
    
    # Build prompt
    prompt = build_prompt(question, contexts)
    
    # Generate answer
    print("🤖 Mengirim ke LLM...")
    
    # TODO: Ganti sesuai LLM yang kalian pilih
    answer = get_answer_groq(prompt)
    # answer = get_answer_gemini(prompt)
    # answer = get_answer_ollama(prompt)
    
    return {
        "question": question,
        "answer": answer,
        "contexts": contexts,
        "prompt": prompt
    }


# ─── CLI Interface ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  🤖 RAG System — UTS Data Engineering")
    print("  Ketik 'keluar' untuk mengakhiri")
    print("=" * 55)

    try:
        vs = load_vectorstore()
        print("✅ Vector database berhasil dimuat\n")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        exit(1)

    while True:
        print()
        question = input("❓ Pertanyaan Anda: ").strip()
        
        if question.lower() in ["keluar", "exit", "quit", "q"]:
            print("👋 Sampai jumpa!")
            break
        
        if not question:
            print("⚠️  Pertanyaan tidak boleh kosong.")
            continue
        
        try:
            result = answer_question(question, vs)
            
            print("\n" + "─" * 55)
            print("💬 JAWABAN:")
            print(result["answer"])
            
            print("\n📚 SUMBER KONTEKS:")
            for i, ctx in enumerate(result["contexts"], 1):
                print(f"  [{i}] Skor: {ctx['score']:.4f} | {ctx['source']}")
                print(f"      {ctx['content'][:100]}...")
            print("─" * 55)
            
        except Exception as e:
            print(f"❌ Error: {e}")
            print("Pastikan API key sudah diatur di file .env")