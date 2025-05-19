from app.rag_pipeline import rag_respond

if __name__ == "__main__":
    while True:
        query = input("\nBạn hỏi gì (y tế): ")
        if query.strip().lower() in ["exit", "quit"]:
            break
        response = rag_respond(query)
        print("\n🤖 Trợ lý y tế:\n", response)
