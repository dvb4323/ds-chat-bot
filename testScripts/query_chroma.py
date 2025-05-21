import chromadb
from sentence_transformers import SentenceTransformer

def query_chroma(query_text, top_k=5):
    # Load lại ChromaDB local
    client = chromadb.PersistentClient(path="../db/chroma_db")
    collection = client.get_or_create_collection("healthcare_chunks")

    # Load sentence-transformer để tạo embedding cho truy vấn
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = model.encode(query_text).tolist()

    # Truy vấn top_k kết quả gần nhất
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    print(f"\n🔍 Kết quả cho truy vấn: \"{query_text}\"\n")
    for i in range(top_k):
        doc = results["documents"][0][i]
        meta = results["metadatas"][0][i]
        dist = results["distances"][0][i]
        print(f"--- Kết quả #{i+1} ---")
        print(f"📏 Khoảng cách: {dist:.4f}")
        print(f"🧠 Prompt: {meta['prompt']}")
        print(f"📄 Response chunk:\n{doc}\n")

if __name__ == "__main__":
    # Bạn có thể sửa câu hỏi test ở đây
    test_query = input("Nhập câu hỏi cần kiểm tra: ")
    query_chroma(test_query, top_k=5)
