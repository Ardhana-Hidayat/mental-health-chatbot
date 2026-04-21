import os
import warnings

os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
warnings.filterwarnings("ignore")

import chromadb
from groq import Groq
from dotenv import load_dotenv
from src.embeddings import embedding

# Muat kunci API dari file .env
load_dotenv()

def load_vectorstore():
    """Memuat koleksi ChromaDB sekali saja agar bisa di-cache oleh UI Streamlit"""
    client_db = chromadb.PersistentClient(path="./chroma_db")
    koleksi = client_db.get_collection(
        name="kesehatan_mental", 
        embedding_function=embedding()
    )
    return koleksi

def answer_question(question, vectorstore=None, k=3):
    """Menerima pertanyaan, mencari konteks, dan mengembalikan jawaban beserta datanya"""
    
    # 1. Pastikan database tersedia
    if vectorstore is None:
        vectorstore = load_vectorstore()
        
    # 2. Proses Retrieval (Mencari potongan teks paling relevan)
    hasil_pencarian = vectorstore.query(
        query_texts=[question], 
        n_results=k
    )
    
    # 3. Rapikan format konteks agar bisa dibaca oleh UI Streamlit
    contexts = []
    konteks_gabungan = ""
    
    # Memeriksa apakah ada dokumen yang ditemukan
    if hasil_pencarian['documents'] and len(hasil_pencarian['documents'][0]) > 0:
        for i in range(len(hasil_pencarian['documents'][0])):
            doc_text = hasil_pencarian['documents'][0][i]
            
            # Ambil nilai kemiripan (distance) jika ChromaDB mengembalikannya
            skor = hasil_pencarian['distances'][0][i] if 'distances' in hasil_pencarian and hasil_pencarian['distances'] else 0.0
            
            # Ambil nama sumber file jika ada di metadata
            sumber = "Dokumen Pengetahuan"
            if 'metadatas' in hasil_pencarian and hasil_pencarian['metadatas'] and hasil_pencarian['metadatas'][0][i]:
                sumber = hasil_pencarian['metadatas'][0][i].get('source', sumber)
                 
            contexts.append({
                "content": doc_text,
                "source": sumber,
                "score": skor
            })
            konteks_gabungan += f"{doc_text}\n\n"

    # 4. Proses Generation (Meminta Groq merangkai jawaban)
    client_groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    system_prompt = (
        "Kamu adalah asisten kesehatan mental yang suportif, hangat, dan empatik, bertindak selayaknya seorang konselor psikologi. "
        "Tugas utamamu adalah mendengarkan dengan penuh perhatian, memvalidasi emosi pengguna, dan memberikan respons yang tidak menghakimi (non-judgmental).\n\n"
        "Panduan Interaksi:\n"
        "1. Selalu mulai dengan memvalidasi perasaan pengguna sebelum memberikan saran atau informasi.\n"
        "2. Gunakan teknik 'active listening', seperti menanyakan pertanyaan terbuka yang lembut untuk membantu pengguna merefleksikan perasaan mereka.\n"
        "3. Hindari nada menggurui, toxic positivity (seperti 'jangan sedih, semua akan baik-baik saja'), atau memberikan diagnosis medis/psikiatri.\n"
        "4. Gunakan informasi yang diberikan di DOKUMEN REFERENSI di bawah ini untuk menjawab pertanyaan pengguna.\n"
        "5. Jika jawaban tidak ada di dalam dokumen, jawablah dengan bijak berdasarkan prinsip pertolongan pertama psikologis (psychological first aid) dan selalu ingatkan pengguna dengan lembut bahwa kamu adalah AI, serta sarankan mereka untuk menghubungi tenaga profesional (psikolog/psikiater) untuk penanganan lebih lanjut.\n\n"
        f"DOKUMEN REFERENSI:\n{konteks_gabungan}"
    )

    try:
        chat_completion = client_groq.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.7, 
        )
        
        jawaban = chat_completion.choices[0].message.content
        
    except Exception as e:
        jawaban = f"Waduh, ada masalah pas hubungi Groq: {e}"
         
    # 5. Kembalikan format dictionary yang diharapkan oleh app.py
    return {
        "answer": jawaban,
        "contexts": contexts,
        "prompt": f"{system_prompt}\n\nPertanyaan User: {question}"
    }

# Bagian untuk mencoba menjalankan lewat terminal
if __name__ == "__main__":
    print("=== 🌱 Teman Cerita (Mode Terminal) ===")
    print("(Ketik 'keluar' untuk berhenti)\n")
    
    koleksi = load_vectorstore()
    
    while True:
        tanya = input("Apa yang ingin kamu ceritakan? ")
        if tanya.lower() == 'keluar':
            break
            
        print("\nSedang merangkai kata...")
        hasil = answer_question(tanya, koleksi)
        print(f"\nAsisten: {hasil['answer']}\n")
        print("-" * 30)