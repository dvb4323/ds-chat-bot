import json
from ollama import Client


def load_sample_prompt(jsonl_path, sample_index=0):
    with open(jsonl_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        data = json.loads(lines[sample_index])
        return data["prompt"], data["response"]


def chat_with_model(model_name, prompt):
    client = Client()
    response = client.chat(model=model_name, messages=[
        {"role": "user", "content": prompt}
    ])
    return response["message"]["content"]


if __name__ == "__main__":
    prompt, expected = load_sample_prompt("../data/healthcare_qa.jsonl", sample_index=0)
    print(f"[INPUT PROMPT]\n{prompt}\n")

    model_name = "llama3"  # Hoặc: "mistral", "phi3", "gemma"
    print(f"[RUNNING MODEL] {model_name}...\n")
    response = chat_with_model(model_name, prompt)

    print(f"[MODEL RESPONSE]\n{response}\n")
    print(f"[EXPECTED ANSWER]\n{expected}\n")
