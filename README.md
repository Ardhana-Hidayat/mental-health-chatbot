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
git clone https://github.com/Ardhana-Hidayat/mental-health-chatbot.git
cd mental-health-chatbot

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

## 📊 10 Pertanyaan Evaluasi

| # | Pertanyaan | Jawaban Sistem | Jawaban Ideal | Skor (1-5) |
|---|-----------|----------------|---------------|-----------|
| 1 | Apa yang dimaksud dengan Psychological First Aid (PFA)? | PFA adalah serangkaian keterampilan untuk mengurangi dampak negatif stres dan mencegah gangguan kesehatan mental dalam situasi krisis. Bantuan praktis dan dukungan kemanusiaan bagi sesama yang terpapar peristiwa krisis serius. | Bantuan praktis dan dukungan kemanusiaan bagi sesama yang terpapar peristiwa krisis serius. | 5 |
| 2 | Siapa saja orang yang membutuhkan dukungan PFA? | Orang-orang yang sedang mengalami situasi krisis psikologis, stres negatif, atau kondisi distress akibat bencana. | Orang yang sangat tertekan setelah terpapar peristiwa krisis yang baru saja terjadi. | 5 |
| 3 | Kapan waktu yang paling tepat untuk memberikan PFA? | Segera setelah kejadian atau beberapa hari/minggu setelahnya, tergantung durasi dan keparahan peristiwa. | Saat pertama kali melakukan kontak dengan orang yang mengalami trauma, biasanya segera setelah kejadian. | 5 |
| 4 | Di mana sebaiknya PFA dilakukan agar aman? | Di tempat yang aman dari bahaya fisik dan memberikan privasi yang cukup agar orang tersebut merasa nyaman. | Di mana saja yang cukup aman bagi pemberi bantuan dan orang yang dibantu, idealnya dengan privasi. | 5 |
| 5 | Apa saja tiga prinsip aksi utama dalam PFA (Look, Listen, Link)? | Prinsip aksi utamanya adalah Amati (Look), Dengar (Listen), dan Hubungkan (Link). | Tiga prinsip aksi dasar: Look (Amati), Listen (Dengar), dan Link (Hubungkan). | 5 |
| 6 | Apa yang harus diperiksa saat kita menerapkan prinsip "Look"? | Memeriksa keamanan lingkungan, mencari orang dengan kebutuhan mendasar, dan mencari orang dengan reaksi distres hebat. | Memeriksa keamanan, orang dengan kebutuhan mendasar yang mendesak, dan orang dengan reaksi distres yang serius. | 5 |
| 7 | Bagaimana cara mendengarkan yang baik agar penyintas merasa didukung? | Memberikan perhatian penuh, tenang, menunjukkan empati, dan tidak memaksa orang untuk bercerita. | Mendengarkan dengan mata, telinga, dan hati; tetap tenang, dan tunjukkan empati. | 5 |
| 8 | Apa saja tanda-tanda umum bahwa seseorang sedang mengalami stres berat? | Reaksi fisik (gemetar, pusing), kecemasan, rasa takut, hingga reaksi emosional yang intens. | Gejala fisik, menangis, cemas berlebih, hingga reaksi menarik diri dari lingkungan. | 5 |
| 9 | Apa langkah pertama jika menemukan seseorang ingin menyakiti diri sendiri? | Mencari bantuan dari orang lain/profesional yang lebih siap dan tidak menangani situasi sendirian. | Jaga keamanan mereka, tetap bersama mereka, dan segera hubungi bantuan profesional/medis. | 5 |
| 10 | Apa yang harus dilakukan jika seseorang ingin menyakiti diri sendiri? | Hubungi layanan darurat segera. | Jaga keamanan mereka dan segera hubungi tenaga medis atau layanan darurat terdekat. | 5 |
| | | | **Rata-rata Skor** | **5** |


## 📊 Hasil Evaluasi

| METRIK           | NILAI |
| ---------------- | ----- |
| TOTAL PERTANYAAN | 10    |
| RATA-RATA SKOR   | 10/10 |
| SKOR TERTINGGI   | 5     |
| SKOR TERENDAH    | -     |

---

## 🔍 Analisis Kelemahan dan Saran Perbaikan

### ⚠️ Kelemahan Sistem

| # | Kelemahan | Deskripsi |
|---|-----------|-----------|
| 1 | Ketergantungan API | Sistem sangat bergantung pada API Groq. Jika kuota habis atau server down, chatbot tidak berfungsi. |
| 2 | Kualitas Chunking | Beberapa potongan teks (chunks) terkadang terpotong di tengah kalimat, sehingga konteks yang diberikan ke AI kurang utuh. |
| 3 | Memori Terbatas | Chatbot saat ini lebih fokus pada dokumen daripada mengingat riwayat percakapan yang sangat panjang. |

### 💡 Saran Perbaikan

| # | Saran | Deskripsi |
|---|-------|-----------|
| 1 | Implementasi Hybrid Search | Menggabungkan pencarian vektor dengan pencarian Keyword Search untuk meningkatkan akurasi istilah medis spesifik. |
| 2 | Optimasi Metadata | Menambahkan metadata yang lebih detail pada dokumen agar chatbot bisa menyebutkan halaman spesifik dari PDF yang dikutip. |
| 3 | Local LLM | Mencoba menggunakan model lokal seperti Ollama agar sistem bisa berjalan tanpa koneksi internet dan menjaga privasi data pengguna lebih baik. |


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
