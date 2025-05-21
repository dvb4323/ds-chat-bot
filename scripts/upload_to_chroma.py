import json
import chromadb
from tqdm import tqdm
import os

BATCH_SIZE = 5000  # < 5461 để tránh lỗi

# Khởi tạo client lưu trữ local
os.makedirs("../db/chroma_db", exist_ok=True)
chroma_client = chromadb.PersistentClient(path="../db/chroma_db")
collection = chroma_client.get_or_create_collection("healthcare_chunks")

# Đọc toàn bộ dữ liệu
all_documents = []
all_embeddings = []
all_metadatas = []
all_ids = []

with open("../data/chunked_with_embedding.jsonl", "r", encoding="utf-8") as f:
    for idx, line in enumerate(tqdm(f, desc="Đang tải dữ liệu")):
        data = json.loads(line)
        all_documents.append(data["response"])
        all_embeddings.append(data["embedding"])
        all_metadatas.append({"prompt": data["prompt"]})
        all_ids.append(f"chunk_{idx}")

# Chia nhỏ và upload theo batch
print("🔄 Bắt đầu upload theo batch...")

for i in tqdm(range(0, len(all_documents), BATCH_SIZE), desc="Uploading to ChromaDB"):
    end = i + BATCH_SIZE
    collection.add(
        documents=all_documents[i:end],
        embeddings=all_embeddings[i:end],
        metadatas=all_metadatas[i:end],
        ids=all_ids[i:end]
    )

print("✅ Hoàn tất upload toàn bộ dữ liệu vào ChromaDB.")
