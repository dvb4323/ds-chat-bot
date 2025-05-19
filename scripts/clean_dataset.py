import json

input_path = "../data/healthcare_qa.jsonl"
output_path = "../data/cleaned_healthcare_qa.jsonl"

seen_pairs = set()
total, saved = 0, 0
with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
    for line in infile:
        total += 1
        item = json.loads(line)
        question = item.get("question", "").strip()
        answer = item.get("answer", "").strip()

        if len(question) < 10:
            print(f"❌ Câu hỏi quá ngắn: {question}")
            continue
        if len(answer) < 20:
            print(f"❌ Câu trả lời quá ngắn: {answer}")
            continue
        if not any(q in question.lower() for q in ["what", "how", "why", "can", "should", "?"]):
            print(f"❌ Không có từ nghi vấn: {question}")
            continue

        key = (question, answer)
        if key in seen_pairs:
            print(f"⚠️ Trùng lặp: {question}")
            continue
        seen_pairs.add(key)

        json.dump({"question": question, "answer": answer}, outfile, ensure_ascii=False)
        outfile.write("\n")
        saved += 1

print(f"✅ Đã lọc và lưu {saved}/{total} Q&A vào {output_path}")
