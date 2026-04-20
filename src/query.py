import os
import warnings

os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
warnings.filterwarnings("ignore")

import chromadb
from groq import Groq
from dotenv import load_dotenv
from embeddings import embedding

# 1. Muat kunci API dari file .env
load_dotenv()

def cari_jawaban(pertanyaan):
    # 2. Hubungkan ke pangkalan data ChromaDB
    # Pastikan path folder database-nya benar
    client_db = chromadb.PersistentClient(path="./chroma_db")
    
    # Ambil koleksi yang sudah di-indexing sebelumnya
    koleksi = client_db.get_collection(
        name="kesehatan_mental", 
        embedding_function=embedding()
    )

    # 3. Proses Retrieval (Mencari 3 potongan teks paling relevan)
    hasil_pencarian = koleksi.query(
        query_texts=[pertanyaan], 
        n_results=3
    )
    
    # Gabungkan potongan-potongan teks menjadi satu konteks
    konteks = "\n\n".join(hasil_pencarian['documents'][0])

    # 4. Proses Generation (Meminta Groq merangkai jawaban)
    client_groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    # Instruksi untuk AI (Prompt Engineering)
    system_prompt = (
        "Kamu adalah asisten kesehatan mental yang suportif, hangat, dan empatik, bertindak selayaknya seorang konselor psikologi. "
        "Tugas utamamu adalah mendengarkan dengan penuh perhatian, memvalidasi emosi pengguna, dan memberikan respons yang tidak menghakimi (non-judgmental).\n\n"
        "Panduan Interaksi:\n"
        "1. Selalu mulai dengan memvalidasi perasaan pengguna sebelum memberikan saran atau informasi.\n"
        "2. Gunakan teknik 'active listening', seperti menanyakan pertanyaan terbuka yang lembut untuk membantu pengguna merefleksikan perasaan mereka.\n"
        "3. Hindari nada menggurui, toxic positivity (seperti 'jangan sedih, semua akan baik-baik saja'), atau memberikan diagnosis medis/psikiatri.\n"
        "4. Gunakan informasi yang diberikan di DOKUMEN REFERENSI di bawah ini untuk menjawab pertanyaan pengguna.\n"
        "5. Jika jawaban tidak ada di dalam dokumen, jawablah dengan bijak berdasarkan prinsip pertolongan pertama psikologis (psychological first aid) dan selalu ingatkan pengguna dengan lembut bahwa kamu adalah AI, serta sarankan mereka untuk menghubungi tenaga profesional (psikolog/psikiater) untuk penanganan lebih lanjut.\n\n"
        f"DOKUMEN REFERENSI:\n{konteks}"
    )

    try:
        chat_completion = client_groq.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": pertanyaan},
            ],
            model="llama-3.3-70b-versatile", # Model cepat dan pas buat tugas UTS
            temperature=0.7, # Biar jawabannya lebih luwes dan tidak kaku
        )
        
        return chat_completion.choices[0].message.content
        
    except Exception as e:
        return f"Waduh, ada masalah pas hubungi Groq: {e}"

# Bagian untuk mencoba menjalankan lewat terminal
if __name__ == "__main__":
    print("=== 🤖 Asisten Kesehatan Mental ===")
    print("(Ketik 'keluar' untuk berhenti)\n")
    
    while True:
        tanya = input("Apa yang ingin kamu ceritakan? ")
        if tanya.lower() == 'keluar':
            break
            
        print("\nSedang mencari jawaban...")
        hasil = cari_jawaban(tanya)
        print(f"\nAsisten: {hasil}\n")
        print("-" * 30)