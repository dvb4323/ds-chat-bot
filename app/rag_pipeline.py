import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from app.ollama_client import query_ollama

def load_index(path="models/embedding_model/faiss.index"):
    return faiss.read_index(path)

def load_dataset(path="data/healthcare_qa.jsonl"):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def retrieve_top_k(query, index, data, model, k=3):
    emb = model.encode([query])
    D, I = index.search(np.array(emb), k)
    results = [data[i] for i in I[0]]
    return results

def build_prompt(query, retrieved_docs):
    context = "\n\n".join([f"Q: {item['question']}\nA: {item['answer']}" for item in retrieved_docs])
    return f"""
You are a helpful medical assistant. Based on the following examples, answer the user's question.

{context}

User: {query}
Assistant:"""

def rag_respond(user_query, model="llama3"):
    index = load_index()
    data = load_dataset()
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    docs = retrieve_top_k(user_query, index, data, embedder)
    prompt = build_prompt(user_query, docs)
    return query_ollama(prompt, model=model)
