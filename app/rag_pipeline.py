from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from app.ollama_client import get_ollama_model

def build_rag(model_name="llama3"):
    # Load embedding & vector store
    embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma(
        persist_directory="../db/chroma_db",
        embedding_function=embedding,
        collection_name="healthcare_chunks"
    )
    retriever = vectordb.as_retriever()

    # Load LLM
    llm = get_ollama_model(model_name)

    # Tạo chuỗi RAG
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain
