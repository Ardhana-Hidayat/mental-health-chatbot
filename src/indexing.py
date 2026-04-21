"""
=============================================================
PIPELINE INDEXING — RAG UTS Data Engineering
=============================================================

Pipeline ini dijalankan SEKALI untuk:
1. Memuat dokumen dari folder data/
2. Memecah dokumen menjadi chunk-chunk kecil
3. Mengubah setiap chunk menjadi vektor (embedding)
4. Menyimpan vektor ke dalam vector database

Jalankan dengan: python src/indexing.py
=============================================================
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# import fungsi pembacaan dokumen
from utils import baca_pdf, baca_csv, baca_json, potong_teks

load_dotenv()

CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
DATA_DIR      = Path(os.getenv("DATA_DIR", "./data"))
VS_DIR        = Path(os.getenv("VECTORSTORE_DIR", "./vectorstore")) # Sekarang masuk ke folder vectorstore

def build_index_scratch():
    import json
    import numpy as np
    from sentence_transformers import SentenceTransformer
    import faiss

    print(" Memulai Pipeline Indexing (From Scratch)")

    # 1. Load Dokumen & Chunking
    print("\n[1 & 2] Membaca dan memotong dokumen (Chunking)...")
    file_paths = [
        DATA_DIR / "p3k_psikologis.pdf",
        DATA_DIR / "p3k-psikologis-jarak-jauh.pdf",
        DATA_DIR / "Buku-Panduan-PFA.pdf",
        DATA_DIR / "Mental_Health_FAQ.csv",
        DATA_DIR / "faq.json",
        DATA_DIR / "conversational.json"
    ]

    chunks = []
    for path_file in file_paths:
        if not path_file.exists():
            print(f"  ⚠️ File tidak ditemukan: {path_file.name}")
            continue
            
        print(f"  📄 Memproses: {path_file.name}")
        ekstensi = path_file.suffix.lower()
        teks = ""
        
        if ekstensi == ".pdf":
            teks = baca_pdf(str(path_file))
        elif ekstensi == ".csv":
            teks = baca_csv(str(path_file))
        elif ekstensi == ".json":
            teks = baca_json(str(path_file))
            
        if teks.strip():
            potongan_file = potong_teks(teks, ukuran=CHUNK_SIZE, tumpang_tindih=CHUNK_OVERLAP)
            for p in potongan_file:
                if len(p.strip()) > 10:
                    # Simpan data ke dictionary (nanti jadi JSON)
                    chunks.append({
                        "id": len(chunks),
                        "source": path_file.name,
                        "content": p
                    })
    print(f"  ✅ {len(chunks)} chunk berhasil dibuat.")

    # 3. Embedding (Ubah teks jadi angka)
    print("\n[3] Membuat Vektor Embedding...")
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    texts = [c["content"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    print(f"  ✅ Embedding selesai. Dimensi vektor: {embeddings.shape}")

    # 4. Simpan ke FAISS (vektor) dan JSON (teks asli)
    print(f"\n[4] Menyimpan data ke folder '{VS_DIR}'...")
    VS_DIR.mkdir(parents=True, exist_ok=True)
    
    # A. Simpan Index FAISS
    dimensi = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimensi)
    index.add(embeddings.astype("float32"))
    faiss.write_index(index, str(VS_DIR / "index.faiss"))
    
    # B. Simpan Teks ke JSON
    with open(VS_DIR / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print("  ✅ File index.faiss dan chunks.json berhasil disimpan!")
    print("\n" + "=" * 55)
    print("  Indexing selesai! Kamu bisa mengintip isinya di chunks.json")
    print("=" * 55)

if __name__ == "__main__":
    build_index_scratch()