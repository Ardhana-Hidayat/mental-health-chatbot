from src.utils import baca_pdf, baca_csv, baca_json, potong_teks
from src.embeddings import embedding
import chromadb

def build_index_scratch():
    print("Memulai proses indexing...")
    
    # Siapkan pangkalan data
    client = chromadb.PersistentClient(path="./chroma_db")
    koleksi = client.get_or_create_collection(
        name="kesehatan_mental",
        embedding_function=embedding()
    )

    # Baca file dari folder data dengan nama penampung yang berbeda-beda
    teks_pdf_1 = baca_pdf("data/p3k_psikologis.pdf")
    teks_pdf_2 = baca_pdf("data/p3k-psikologis-jarak-jauh.pdf")
    teks_pdf_3 = baca_pdf("data/Buku-Panduan-PFA.pdf")
    
    teks_csv = baca_csv("data/Mental_Health_FAQ.csv")
    
    teks_json_1 = baca_json("data/faq.json")
    teks_json_2 = baca_json("data/conversational.json")
    
    # Gabungkan semuanya secara aman
    teks_gabungan = (
        teks_pdf_1 + "\n" + 
        teks_pdf_2 + "\n" + 
        teks_pdf_3 + "\n" + 
        teks_csv + "\n" + 
        teks_json_1 + "\n" + 
        teks_json_2
    )

    # Potong teks menjadi bagian kecil
    potongan = potong_teks(teks_gabungan)

    # Simpan ke pangkalan data
    daftar_id = [f"id_{i}" for i in range(len(potongan))]
    koleksi.add(documents=potongan, ids=daftar_id)
    
    print(f"Berhasil menyimpan {len(potongan)} potongan teks ke database!")

if __name__ == "__main__":
    build_index_scratch()