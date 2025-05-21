from app.rag_pipeline import build_rag
from app.start_model import start_model

def main():
    print("🤖 Chào mừng đến với Healthcare RAG Chatbot")
    print("Chọn mô hình:")
    print("1. LLaMA 3")
    print("2. Mistral")
    print("3. DeepSeek-R1")

    model_choice = input("👉 Nhập số mô hình: ")
    model_map = {"1": "llama3", "2": "mistral", "3": "deepseek-r1"}
    model_name = model_map.get(model_choice.strip(), "llama3")

    # 🧠 Tự động khởi động mô hình nếu chưa chạy
    start_model(model_name)

    # 🧠 Khởi tạo pipeline RAG
    qa = build_rag(model_name)

    while True:
        question = input("\n❓ Nhập câu hỏi của bạn (hoặc 'exit'): ")
        if question.lower() == "exit":
            break
        result = qa.invoke({"query": question})
        print("\n💡 Trả lời:")
        print(result["result"])

if __name__ == "__main__":
    main()