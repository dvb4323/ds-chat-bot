import os
import json
from datasets import load_dataset

def download_dataset(save_path="../data/healthcare_qa.jsonl"):
    print("[INFO] Đang tải dataset từ Hugging Face...")
    dataset = load_dataset("lavita/ChatDoctor-HealthCareMagic-100k")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if os.path.exists(save_path):
        confirm = input(f"[WARNING] File {save_path} đã tồn tại. Ghi đè? (y/n): ")
        if confirm.lower() != 'y':
            print("[INFO] Hủy thao tác ghi file.")
            return

    with open(save_path, "w", encoding="utf-8") as f:
        for i, item in enumerate(dataset["train"]):
            prompt = f"{item['instruction'].strip()}\n\nPatient: {item['input'].strip()}"
            answer = item["output"].strip()

            json.dump({
                "prompt": prompt,
                "response": answer
            }, f)
            f.write("\n")

            if i % 5000 == 0 and i > 0:
                print(f"[INFO] Đã ghi {i} mẫu...")

    print(f"[SUCCESS] Đã lưu {len(dataset['train'])} mẫu vào '{save_path}'.")

if __name__ == "__main__":
    download_dataset()
