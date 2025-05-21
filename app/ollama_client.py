from langchain_ollama import OllamaLLM

def get_ollama_model(model_name: str):
    return OllamaLLM(model=model_name)
