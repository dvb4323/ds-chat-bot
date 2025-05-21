from app.rag_pipeline import build_rag

def main():
    print("🤖 Chào mừng đến với Healthcare RAG Chatbot")
    print("Chọn mô hình:")
    print("1. LLaMA 3")
    print("2. Mistral")
    print("3. DeepSeek")

    model_choice = input("👉 Nhập số mô hình: ")
    model_map = {"1": "llama3", "2": "mistral", "3": "deepseek-coder"}
    model_name = model_map.get(model_choice.strip(), "llama3")

    print(f"🔧 Đang khởi tạo chatbot với mô hình: {model_name}...")
    qa = build_rag(model_name)

    while True:
        question = input("\n❓ Nhập câu hỏi của bạn (hoặc 'exit'): ")
        if question.lower() == "exit":
            break
        result = qa({"query": question})
        print("\n💡 Trả lời:")
        print(result["result"])

if __name__ == "__main__":
    main()
