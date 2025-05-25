from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from app.ollama_client import get_ollama_model
from app.ollama_client import start_model

def build_rag(model_name="llama3"):
    # Load embedding & vector store
    # embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
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

def load_rag_chain(model_name: str):
    start_model(model_name)

    llm = get_ollama_model(model_name)

    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # dùng tạm
    vectordb = Chroma(
        persist_directory="../db/chroma_db",
        embedding_function=embedding,
        collection_name="healthcare_chunks"
    )
    retriever = vectordb.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    return qa_chain