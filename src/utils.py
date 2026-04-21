import csv
import json
from PyPDF2 import PdfReader

def baca_pdf(file_path):
    teks = ""
    try:
        reader = PdfReader(file_path)
        for halaman in reader.pages:
            teks += halaman.extract_text() + "\n"
    except Exception as e:
        print(f"Gagal membaca PDF: {e}")
    return teks

def baca_csv(file_path):
    teks = ""
    try:
        with open(file_path, mode='r', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file)
            for baris in csv_reader:
                tanya = baris.get('Pertanyaan', baris.get('Questions', ''))
                jawab = baris.get('Jawaban', baris.get('Answers', ''))
                teks += f"Pertanyaan: {tanya}\nJawaban: {jawab}\n\n"
    except Exception as e:
        print(f"Gagal membaca CSV: {e}")
    return teks

def baca_json(file_path):
    teks = ""
    try:
        with open(file_path, mode='r', encoding='utf-8') as file:
            data = json.load(file)
            
            if isinstance(data, dict) and "intents" in data:
                for intent in data["intents"]:
                    pertanyaan = " atau ".join(intent.get("patterns", []))
                    jawaban = " | ".join(intent.get("responses", []))
                    
                    teks += f"Pertanyaan: {pertanyaan}\nJawaban: {jawaban}\n\n"
                    
            else:
                teks = json.dumps(data, indent=2, ensure_ascii=False)
                
    except Exception as e:
        print(f"Gagal membaca JSON: {e}")
    return teks

def potong_teks(teks, ukuran=500, tumpang_tindih=50):
    potongan = []
    mulai = 0
    while mulai < len(teks):
        selesai = mulai + ukuran
        potongan.append(teks[mulai:selesai])
        mulai = selesai - tumpang_tindih
    return potongan