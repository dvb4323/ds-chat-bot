import gradio as gr
import traceback
import time

# Import với error handling
try:
    from app.rag_pipeline import load_rag_chain

    print("✅ Import RAG pipeline thành công")
    RAG_AVAILABLE = True
except Exception as e:
    print(f"⚠️ Lỗi import RAG pipeline: {e}")
    RAG_AVAILABLE = False


    # Mock function đơn giản
    def load_rag_chain(model_name):
        class MockChain:
            def run(self, query):
                time.sleep(1)
                if "deepseek-r1" in model_name:
                    return """<think>
User is asking about headache. I need to provide helpful medical information while being careful.

Common causes:
- Dehydration (very common)
- Stress and tension
- Lack of sleep
- Eye strain
- Poor posture

I should provide general guidance and recommend consulting healthcare professionals.
</think>

I understand you're dealing with a headache. Here's some general guidance:

IMMEDIATE RELIEF:
• Drink water slowly (dehydration is very common)
• Rest in a quiet, dark room
• Apply cold compress to forehead
• Gentle massage of temples and neck

COMMON TRIGGERS:
• Dehydration
• Lack of sleep 
• Stress and tension
• Eye strain from screens
• Poor posture
• Skipped meals

WHEN TO SEEK MEDICAL HELP:
• Sudden severe headache
• Headache with fever/stiff neck
• After head injury
• Progressively worsening

IMPORTANT: This is general information only. For persistent headaches, consult a healthcare provider."""

                return f"Response từ {model_name}: {query}\n\nĐây là phản hồi test."

        return MockChain()


class SimpleRAGChatbot:
    def __init__(self):
        self.models = ["llama3", "mistral", "deepseek-r1"]
        self.chains = {}
        self.chat_histories = {model: [] for model in self.models}
        self.current_model = "llama3"  # Test với problematic model
        print(f"🚀 Khởi tạo Simple RAG Chatbot")

    def load_model(self, model_name: str) -> str:
        try:
            print(f"🔄 Đang tải model: {model_name}")
            if model_name not in self.chains:
                self.chains[model_name] = load_rag_chain(model_name)
                print(f"✅ Đã tải model: {model_name}")
            return f"✅ Model {model_name} đã sẵn sàng!"
        except Exception as e:
            error_msg = f"❌ Lỗi khi tải model {model_name}: {str(e)}"
            print(error_msg)
            return error_msg

    def change_model(self, model_name: str):
        print(f"🔄 Chuyển sang model: {model_name}")
        self.current_model = model_name
        status = self.load_model(model_name)
        history = self.chat_histories[model_name]
        return history, status

    def chat(self, message: str, history):
        if not message or not message.strip():
            return history, ""

        print(f"💬 Nhận tin nhắn: {message}")

        try:
            # Load model nếu chưa có
            if self.current_model not in self.chains:
                self.load_model(self.current_model)

            # Gọi RAG chain
            chain = self.chains[self.current_model]
            print(f"🤖 Đang xử lý với model: {self.current_model}")

            # Lấy response
            raw_response = chain.run(message)

            # Debug raw response
            print(f"✅ Received response type: {type(raw_response)}")
            print(f"✅ Received response length: {len(str(raw_response)) if raw_response else 0}")
            print(f"🔍 Response starts with: {str(raw_response)[:50] if raw_response else 'None'}...")

            # Convert to string
            if raw_response is None:
                response = "❌ Model không trả về phản hồi"
            else:
                response = str(raw_response).strip()

            # Handle DeepSeek-R1 format - AVOID MARKDOWN
            if "<think>" in response and "</think>" in response:
                try:
                    # Extract thinking and answer parts
                    parts = response.split("</think>", 1)
                    if len(parts) == 2:
                        thinking_part = parts[0].replace("<think>", "").strip()
                        answer_part = parts[1].strip()

                        # Format without markdown to avoid gray background
                        response = f"🧠 QUÁ TRÌNH SUY LUẬN:\n{thinking_part}\n\n📝 CÂU TRẢ LỜI:\n{answer_part}"
                        print(
                            f"🔧 Processed DeepSeek format - thinking: {len(thinking_part)}, answer: {len(answer_part)}")
                    else:
                        # Simple replacement
                        response = response.replace("<think>", "🧠 QUÁ TRÌNH SUY LUẬN:\n")
                        response = response.replace("</think>", "\n\n📝 CÂU TRẢ LỜI:\n")
                except Exception as e:
                    print(f"⚠️ Processing error: {e}")
                    # Fallback - just remove tags
                    response = response.replace("<think>", "").replace("</think>", "")

            # Ensure not empty
            if not response:
                response = "⚠️ Response rỗng từ model"

            print(f"📤 Final response length: {len(response)}")

            # Add to history - simple format
            history.append([message, response])
            self.chat_histories[self.current_model] = history

            return history, ""

        except Exception as e:
            error_msg = f"❌ Lỗi: {str(e)}"
            print(f"❌ Lỗi chat: {e}")
            print(traceback.format_exc())
            history.append([message, error_msg])
            return history, ""

    def clear_history(self):
        print(f"🗑️ Xóa lịch sử chat của model: {self.current_model}")
        self.chat_histories[self.current_model] = []
        return [], f"🗑️ Đã xóa lịch sử chat của {self.current_model}"


# Khởi tạo chatbot
chatbot = SimpleRAGChatbot()

# CSS fixed cho DeepSeek display
simple_css = """
.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
}

.chatbot {
    height: 600px !important;
    font-family: system-ui, -apple-system, sans-serif !important;
}

/* Force full display */
.chatbot .message,
.chatbot .message *,
.chatbot .prose,
.chatbot .prose * {
    max-height: none !important;
    overflow: visible !important;
    white-space: pre-wrap !important;
    word-wrap: break-word !important;
}

.chatbot .user,
.chatbot .bot {
    max-height: none !important;
    overflow: visible !important;
    padding: 15px !important;
    margin: 10px 0 !important;
    line-height: 1.5 !important;
}

/* Fix gray background issue */
.chatbot pre,
.chatbot code {
    background: #f8f9fa !important;
    color: #333 !important;
    border: 1px solid #e9ecef !important;
    border-radius: 6px !important;
    padding: 12px !important;
    white-space: pre-wrap !important;
    word-wrap: break-word !important;
    font-family: 'Consolas', 'Monaco', monospace !important;
    font-size: 14px !important;
    line-height: 1.4 !important;
}

/* Special styling for thinking sections */
.chatbot .bot .prose h2,
.chatbot .bot .prose h3,
.chatbot .bot .prose strong {
    color: #2c3e50 !important;
    background: transparent !important;
}

/* Ensure text is readable */
.chatbot .bot .prose {
    color: #333 !important;
    background: transparent !important;
}

.chatbot .bot .prose p {
    color: #333 !important;
    background: transparent !important;
    margin: 8px 0 !important;
}
"""

# Tạo giao diện Gradio đơn giản
with gr.Blocks(title="Simple RAG Chatbot", css=simple_css) as app:
    # Header
    gr.Markdown("# 🤖 Simple RAG Chatbot")
    gr.Markdown("Medical Assistant")

    if not RAG_AVAILABLE:
        gr.Markdown("⚠️ **Development Mode:** Sử dụng mock responses")

    # Controls
    with gr.Row():
        model_dropdown = gr.Dropdown(
            choices=chatbot.models,
            value=chatbot.current_model,
            label="Chọn Model"
        )

        status_box = gr.Textbox(
            label="Trạng thái",
            value="Đang khởi tạo...",
            interactive=False
        )

    # Chat interface - NO TYPE PARAMETER
    chat_interface = gr.Chatbot(
        label="Chat",
        height=600,
        show_copy_button=True,
        placeholder="Bắt đầu trò chuyện..."
    )

    # Input
    with gr.Row():
        msg_input = gr.Textbox(
            placeholder="Nhập tin nhắn...",
            container=False,
            scale=4
        )
        send_btn = gr.Button("Gửi", variant="primary", scale=1)

    # Buttons
    clear_btn = gr.Button("Xóa lịch sử")

    # Examples
    gr.Examples(
        examples=[
            ["Hello, I'm having a headache, what should I do?"],
            ["What are the benefits of exercise?"],
            ["How to improve sleep quality?"]
        ],
        inputs=msg_input
    )

    # Event handlers
    model_dropdown.change(
        fn=chatbot.change_model,
        inputs=[model_dropdown],
        outputs=[chat_interface, status_box]
    )

    msg_input.submit(
        fn=chatbot.chat,
        inputs=[msg_input, chat_interface],
        outputs=[chat_interface, msg_input]
    )

    send_btn.click(
        fn=chatbot.chat,
        inputs=[msg_input, chat_interface],
        outputs=[chat_interface, msg_input]
    )

    clear_btn.click(
        fn=chatbot.clear_history,
        outputs=[chat_interface, status_box]
    )

    # Load model khi khởi động
    app.load(
        fn=lambda: chatbot.load_model(chatbot.current_model),
        outputs=[status_box]
    )

if __name__ == "__main__":
    print("🚀 Khởi chạy Simple RAG Chatbot...")
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        debug=True
    )