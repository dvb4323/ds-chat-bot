import nltk
import json
from tqdm import tqdm

nltk.download("punkt")

MAX_TOKENS = 200  # hoặc dùng word count nếu không dùng tokenizer

def chunk_text(text, max_words=MAX_TOKENS):
    from nltk.tokenize import sent_tokenize
    sentences = sent_tokenize(text)
    chunks, current = [], []

    count = 0
    for sent in sentences:
        words = sent.split()
        if count + len(words) > max_words:
            if current:
                chunks.append(" ".join(current))
                current = []
                count = 0
        current.append(sent)
        count += len(words)

    if current:
        chunks.append(" ".join(current))
    return chunks

input_path = "data/cleaned_healthcare_qa.jsonl"
output_path = "data/cleaned_chunked_healthcare_qa.jsonl"

with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
    for line in tqdm(infile):
        item = json.loads(line)
        question = item["question"]
        answer = item["answer"]

        chunks = chunk_text(answer)
        for chunk in chunks:
            json.dump({"instruction": "answer the medical question", "question": question, "answer": chunk}, outfile, ensure_ascii=False)
            outfile.write("\n")

print(f"✅ Đã lưu các đoạn chia nhỏ vào {output_path}")
