from langchain_community.llms import Ollama

def get_ollama_model(model_name: str):
    return Ollama(model=model_name)
