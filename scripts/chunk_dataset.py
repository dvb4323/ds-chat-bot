import json
import tiktoken
from tqdm import tqdm

# Cấu hình
input_path = "../data/cleaned_healthcare_qa.jsonl"
output_path = "../data/chunked_healthcare_qa.jsonl"
max_tokens = 200

# Chọn tokenizer (tùy model bạn dùng, ví dụ: gpt-3.5-turbo)
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

def chunk_text(text, max_tokens):
    tokens = encoding.encode(text)
    chunks = [tokens[i:i+max_tokens] for i in range(0, len(tokens), max_tokens)]
    return [encoding.decode(chunk) for chunk in chunks]

count = 0
with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
    for line in tqdm(infile, desc="🧩 Đang chia nhỏ response"):
        item = json.loads(line)
        prompt = item["prompt"]
        response = item["response"]

        chunks = chunk_text(response, max_tokens)

        for i, chunk in enumerate(chunks):
            json.dump({
                "prompt": prompt,
                "response": chunk,
                "chunk_id": i + 1,
                "total_chunks": len(chunks)
            }, outfile, ensure_ascii=False)
            outfile.write("\n")
            count += 1

print(f"✅ Đã chia nhỏ và lưu {count} response chunk vào {output_path}")
