import json
from tqdm import tqdm

# Cấu hình các giới hạn
min_prompt_len = 30
min_response_len = 50
max_prompt_len = 500
max_response_len = 1500

input_path = "../data/healthcare_qa.jsonl"
output_path = "../data/cleaned_healthcare_qa.jsonl"

seen_pairs = set()
kept = 0

with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
    for line in tqdm(infile, desc="🔍 Đang lọc dữ liệu"):
        item = json.loads(line)
        prompt = item.get("prompt", "").strip()
        response = item.get("response", "").strip()

        # Kiểm tra điều kiện độ dài
        if len(prompt) < min_prompt_len or len(prompt) > max_prompt_len:
            continue
        if len(response) < min_response_len or len(response) > max_response_len:
            continue

        # Tránh trùng lặp
        key = (prompt, response)
        if key in seen_pairs:
            continue
        seen_pairs.add(key)

        # Ghi dữ liệu hợp lệ
        json.dump({"prompt": prompt, "response": response}, outfile, ensure_ascii=False)
        outfile.write("\n")
        kept += 1

print(f"\n✅ Đã lọc và giữ lại {kept} cặp Q&A từ {input_path} → {output_path}")
