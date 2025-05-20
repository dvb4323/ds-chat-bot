import os
import json
import re
from tqdm import tqdm

# Đường dẫn dữ liệu
INPUT_FILE = "../data/cleaned_healthcare_qa.jsonl"
OUTPUT_FILE = "../data/chunked_healthcare_qa.jsonl"
CHUNK_SIZE_WORDS = 200
OVERLAP_WORDS = 50

def clean_text(text):
    """Làm sạch văn bản"""
    return re.sub(r"\s+", " ", text).strip()

def split_into_chunks(text, chunk_size=200, overlap=50):
    """Tách văn bản thành các đoạn theo từ, có overlap"""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks

def chunk_dataset(input_path, output_path, chunk_size, overlap):
    chunked_count = 0
    with open(output_path, "w", encoding="utf-8") as fout:
        with open(input_path, "r", encoding="utf-8") as fin:
            for line in tqdm(fin, desc="Chunking responses"):
                entry = json.loads(line)
                prompt = clean_text(entry.get("prompt", ""))
                response = clean_text(entry.get("response", ""))

                response_chunks = split_into_chunks(response, chunk_size, overlap)

                for chunk in response_chunks:
                    new_entry = {
                        "prompt": prompt,    # giữ nguyên
                        "response": chunk    # chia nhỏ
                    }
                    fout.write(json.dumps(new_entry, ensure_ascii=False) + "\n")
                    chunked_count += 1

    print(f"✅ Đã tạo {chunked_count} chunks từ response và lưu vào {output_path}")

if __name__ == "__main__":
    chunk_dataset(
        input_path=INPUT_FILE,
        output_path=OUTPUT_FILE,
        chunk_size=CHUNK_SIZE_WORDS,
        overlap=OVERLAP_WORDS
    )
