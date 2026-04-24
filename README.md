# 🤖 RAG Starter Pack — UTS Data Engineering

> **Retrieval-Augmented Generation** — Sistem Tanya-Jawab Cerdas Berbasis Dokumen

Starter pack ini adalah **kerangka awal** proyek RAG untuk UTS Data Engineering D3/D4.
Mahasiswa mengisi, memodifikasi, dan mengembangkan kode ini sesuai topik kelompok masing-masing.

---

## 👥 Identitas Kelompok

| Nama                    | NIM       | Tugas Utama      |
|-------------------------|-----------|------------------|
| Muhammad Adam Al Kidri  | 244311050 | Project Manager  |
| Iqbal Abdullah          | 244311044 | Data Analyst     |
| Ardhana Syah Hidayat    | 244311037 | Data Engineer    |

**Topik Domain:** *Kesehatan (Kesehatan Mental)*  
**Stack yang Dipilih:** *From Scratch*  
**LLM yang Digunakan:** *Grok API*  
**Vector DB yang Digunakan:** *ChromaDB*

---

## 🗂️ Struktur Proyek

```
rag-uts-kelompok-3/
├── data/                    # Dokumen sumber Anda (PDF, TXT, dll.)
│   └── sample.txt           # Contoh dokumen (ganti dengan dokumen Anda)
├── src/
│   ├── indexing.py          # 🔧 WAJIB DIISI: Pipeline indexing
│   ├── query.py             # 🔧 WAJIB DIISI: Pipeline query & retrieval
│   ├── embeddings.py        # 🔧 WAJIB DIISI: Konfigurasi embedding
│   └── utils.py             # Helper functions
├── ui/
│   └── app.py               # 🔧 WAJIB DIISI: Antarmuka Streamlit
├── docs/
│   └── arsitektur.png       # 📌 Diagram arsitektur (buat sendiri)
├── evaluation/
│   └── hasil_evaluasi.xlsx  # 📌 Tabel evaluasi 10 pertanyaan
├── notebooks/
│   └── 01_demo_rag.ipynb    # Notebook demo dari hands-on session
├── .env.example             # Template environment variables
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ⚡ Cara Memulai (Quickstart)

### 1. Clone & Setup

```bash
# Clone repository ini
git clone https://github.com/[username]/rag-uts-[kelompok].git
cd rag-uts-[kelompok]

# Buat virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
# atau: venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Konfigurasi API Key

```bash
# Salin template env
cp .env.example .env

# Edit .env dan isi API key Anda
# JANGAN commit file .env ke GitHub!
```

### 3. Siapkan Dokumen

Letakkan dokumen sumber Anda di folder `data/`:
```bash
# Contoh: salin PDF atau TXT ke folder data
cp dokumen-saya.pdf data/
```

### 4. Jalankan Indexing (sekali saja)

```bash
python src/indexing.py
```

### 5. Jalankan Sistem RAG

```bash
# Dengan Streamlit UI
streamlit run ui/app.py

# Atau via CLI
python src/query.py
```

---

## 🔧 Konfigurasi

Semua konfigurasi utama ada di `src/config.py` (atau langsung di setiap file):

| Parameter       | Default                        | Keterangan                          |
|-----------------|--------------------------------|-------------------------------------|
| `CHUNK_SIZE`    | 500                            | Ukuran setiap chunk teks (karakter) |
| `CHUNK_OVERLAP` | 50                             | Overlap antar chunk                 |
| `TOP_K`         | 3                              | Jumlah dokumen relevan yang diambil |
| `MODEL_NAME`    | Groq - llama-3.3-70b-versatile | Nama model LLM yang digunakan       |

---

## 📊 Hasil Evaluasi

| METRIK           | NILAI |
| ---------------- | ----- |
| TOTAL PERTANYAAN | 10    |
| RATA-RATA SKOR   | 10/10 |
| SKOR TERTINGGI   | 5     |
| SKOR TERENDAH    | -     |

---

## 🏗️ Arsitektur Sistem

<img width="1455" height="1401" alt="mermaid-diagram-2026-04-23-190646" src="https://github.com/user-attachments/assets/0e8fa88b-10fe-4901-ba7b-993f761feb7e" />

---

## 📚 Referensi & Sumber

- Framework: *-*
- LLM: *Groq - llama-3.3-70b-versatile*
- Vector DB: *ChromaDB*

---

## 👨‍🏫 Informasi UTS

- **Mata Kuliah:** Data Engineering
- **Program Studi:** D4 Teknologi Rekayasa Perangkat Lunak
- **Deadline:** *24 April 2026*
"# mental-health-chatbot" 
