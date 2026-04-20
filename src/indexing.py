import chromadb
from utils import baca_pdf, baca_csv, potong_teks
from embedding import embedding

def build_index_scratch():
    print("Memulai proses indexing...")
    
    # Siapkan pangkalan data
    client = chromadb.PersistentClient(path="./chroma_db")
    koleksi = client.get_or_create_collection(
        name="kesehatan_mental",
        embedding_function=embedding()
    )

    # Baca file dari folder data
    teks_pdf = baca_pdf("data/p3k_psikologis.pdf")
    teks_csv = baca_csv("data/Mental_Health_FAQ.csv")
    teks_gabungan = teks_pdf + "\n" + teks_csv

    # Potong teks menjadi bagian kecil
    potongan = potong_teks(teks_gabungan)

    # Simpan ke pangkalan data
    daftar_id = [f"id_{i}" for i in range(len(potongan))]
    koleksi.add(documents=potongan, ids=daftar_id)
    
    print(f"Berhasil menyimpan {len(potongan)} potongan teks ke database!")

if __name__ == "__main__":
    build_index_scratch()