import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import numpy as np

def create_embedding(text, model):
    emb = model.encode(text)
    return emb.astype(np.float32)

def main():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    input_file = '../data/chunked_healthcare_qa.jsonl'
    output_file = '../data/chunked_with_embedding.jsonl'

    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:

        for line in tqdm(f_in):
            data = json.loads(line)
            text = data['prompt'] + " " + data['response']
            embedding = create_embedding(text, model).tolist()
            data['embedding'] = embedding
            f_out.write(json.dumps(data, ensure_ascii=False) + '\n')

    print(f"Đã lưu dữ liệu kèm embedding vào {output_file}")

if __name__ == '__main__':
    main()




## Test
# from sentence_transformers import SentenceTransformer
# import numpy as np
#
# # Load model
# model = SentenceTransformer('all-MiniLM-L6-v2')  # model nhẹ, hiệu quả, phổ biến
#
# # Hàm tạo embedding
# def create_embedding(text):
#     embedding = model.encode(text)
#     return embedding.astype(np.float32)  # chuẩn kiểu float32 để dùng FAISS, Chroma
#
# # Test nhanh
# # if __name__ == '__main__':
# #     text = "Triệu chứng sốt và ho kéo dài nên làm gì?"
# #     emb = create_embedding(text)
# #     print(f"Embedding vector shape: {emb.shape}")
# #     print(emb[:5])  # xem 5 chiều đầu tiên
