from sentence_transformers import SentenceTransformer
import faiss
import json
import os
import numpy as np

def load_data(jsonl_path):
    with open(jsonl_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def build_faiss_index(data, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    questions = [item["question"] for item in data]
    embeddings = model.encode(questions, show_progress_bar=True)

    index = faiss.IndexFlatL2(embeddings[0].shape[0])
    index.add(np.array(embeddings))

    return index, embeddings, questions, data

def save_index(index, path="models/embedding_model/faiss.index"):
    faiss.write_index(index, path)

if __name__ == "__main__":
    os.makedirs("models/embedding_model", exist_ok=True)
    data = load_data("data/healthcare_qa.jsonl")
    index, embeddings, questions, _ = build_faiss_index(data)
    save_index(index)
