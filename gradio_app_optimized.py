import gradio as gr
import asyncio
import threading
import time
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import traceback

# Import với error handling
try:
    from app.rag_pipeline import load_rag_chain
    from app.ollama_client import is_model_running, start_model

    print("✅ Import RAG pipeline thành công")
    RAG_AVAILABLE = True
except Exception as e:
    print(f"⚠️ Lỗi import RAG pipeline: {e}")
    RAG_AVAILABLE = False


class ModelManager:
    """Quản lý model với caching và async loading"""

    def __init__(self):
        self.models = {
            "llama3": {"name": "Llama 3", "status": "unloaded", "chain": None},
            "mistral": {"name": "Mistral", "status": "unloaded", "chain": None},
            "deepseek-r1": {"name": "DeepSeek R1", "status": "unloaded", "chain": None}
        }
        self.loading_lock = threading.Lock()

    def get_model_status(self, model_name: str) -> str:
        """Lấy trạng thái model"""
        if model_name not in self.models:
            return "unknown"
        return self.models[model_name]["status"]

    def is_model_loaded(self, model_name: str) -> bool:
        """Kiểm tra model đã load chưa"""
        return (model_name in self.models and
                self.models[model_name]["status"] == "loaded" and
                self.models[model_name]["chain"] is not None)

    def load_model_sync(self, model_name: str, progress_callback=None) -> Tuple[bool, str]:
        """Load model đồng bộ với progress tracking"""
        with self.loading_lock:
            try:
                if progress_callback:
                    progress_callback(f"🔄 Đang kiểm tra model {model_name}...")

                self.models[model_name]["status"] = "loading"

                if progress_callback:
                    progress_callback(f"🚀 Đang khởi động Ollama service...")

                # Load RAG chain
                if RAG_AVAILABLE:
                    if progress_callback:
                        progress_callback(f"📚 Đang load embeddings và vector database...")

                    chain = load_rag_chain(model_name)
                    self.models[model_name]["chain"] = chain
                else:
                    # Mock chain for development
                    class MockChain:
                        def __call__(self, inputs):
                            query = inputs.get("query", "")
                            return {
                                "result": f"Mock response từ {model_name}: {query}",
                                "source_documents": []
                            }

                    self.models[model_name]["chain"] = MockChain()

                self.models[model_name]["status"] = "loaded"

                if progress_callback:
                    progress_callback(f"✅ Model {model_name} đã sẵn sàng!")

                return True, f"✅ Model {model_name} loaded successfully"

            except Exception as e:
                self.models[model_name]["status"] = "error"
                error_msg = f"❌ Lỗi loading model {model_name}: {str(e)}"
                print(error_msg)
                print(traceback.format_exc())
                return False, error_msg

    def get_chain(self, model_name: str):
        """Lấy chain của model"""
        if self.is_model_loaded(model_name):
            return self.models[model_name]["chain"]
        return None


class ChatGPTLikeRAGBot:
    """RAG Chatbot với giao diện giống ChatGPT"""

    def __init__(self):
        self.model_manager = ModelManager()
        self.conversations = {}  # {conversation_id: {"messages": [], "model": ""}}
        self.current_conversation_id = self.create_new_conversation()
        self.current_model = "llama3"

    def create_new_conversation(self) -> str:
        """Tạo cuộc hội thoại mới"""
        conv_id = f"conv_{int(time.time())}"
        self.conversations[conv_id] = {
            "messages": [],
            "model": self.current_model,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "title": "New Chat"
        }
        return conv_id

    def get_conversation_list(self) -> List[Tuple[str, str]]:
        """Lấy danh sách cuộc hội thoại"""
        conversations = []
        for conv_id, data in self.conversations.items():
            title = data.get("title", "New Chat")
            if len(title) > 30:
                title = title[:30] + "..."
            conversations.append((conv_id, f"{title} ({data['created_at']})"))
        return sorted(conversations, key=lambda x: x[0], reverse=True)

    def switch_conversation(self, conv_id: str):
        """Chuyển cuộc hội thoại"""
        if conv_id in self.conversations:
            self.current_conversation_id = conv_id
            self.current_model = self.conversations[conv_id]["model"]
            messages = self.conversations[conv_id]["messages"]

            # Convert to Gradio format
            history = []
            for msg in messages:
                if msg["role"] == "user":
                    user_msg = msg["content"]
                elif msg["role"] == "assistant":
                    assistant_msg = msg["content"]
                    history.append([user_msg, assistant_msg])

            return history, f"✅ Switched to conversation: {conv_id}"
        return [], f"❌ Conversation not found: {conv_id}"

    def load_model_with_progress(self, model_name: str):
        """Load model với progress indicator"""

        def progress_generator():
            status_messages = []

            def progress_callback(message):
                status_messages.append(message)

            # Load model trong thread riêng
            success, final_message = self.model_manager.load_model_sync(
                model_name, progress_callback
            )

            # Yield từng status message
            for msg in status_messages:
                yield msg
                time.sleep(0.5)  # Simulate progress

            yield final_message

        return progress_generator()

    def change_model(self, model_name: str):
        """Chuyển model"""
        self.current_model = model_name
        self.conversations[self.current_conversation_id]["model"] = model_name

        # Check if model is loaded
        if self.model_manager.is_model_loaded(model_name):
            return f"✅ Switched to {model_name} (already loaded)"
        else:
            return f"🔄 Switched to {model_name} (will load on first use)"

    def chat(self, message: str, history: List) -> Tuple[List, str]:
        """Xử lý chat với RAG"""
        if not message or not message.strip():
            return history, ""

        try:
            # Load model nếu chưa có
            if not self.model_manager.is_model_loaded(self.current_model):
                success, status_msg = self.model_manager.load_model_sync(self.current_model)
                if not success:
                    history.append([message, f"❌ Cannot load model: {status_msg}"])
                    return history, ""

            # Get chain
            chain = self.model_manager.get_chain(self.current_model)
            if not chain:
                history.append([message, "❌ Model not available"])
                return history, ""

            # Call RAG chain properly
            start_time = time.time()

            if RAG_AVAILABLE:
                # Proper RAG call
                result = chain({"query": message})
                response = result.get("result", "No response")
                sources = result.get("source_documents", [])

                # Add source information if available
                if sources:
                    source_info = f"\n\n📚 **Sources:** {len(sources)} documents retrieved"
                    response += source_info
            else:
                # Mock response for development
                result = chain({"query": message})
                response = result.get("result", "Mock response")

            processing_time = time.time() - start_time

            # Handle DeepSeek-R1 special format
            if "deepseek-r1" in self.current_model.lower() and "<think>" in response:
                response = self._format_deepseek_response(response)

            # Add timing info
            response += f"\n\n⏱️ *Processed in {processing_time:.2f}s*"

            # Update conversation
            conv = self.conversations[self.current_conversation_id]
            conv["messages"].extend([
                {"role": "user", "content": message},
                {"role": "assistant", "content": response}
            ])

            # Update conversation title if it's the first message
            if conv["title"] == "New Chat" and len(conv["messages"]) == 2:
                conv["title"] = message[:50] + ("..." if len(message) > 50 else "")

            # Add to history
            history.append([message, response])

            return history, ""

        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            print(f"Chat error: {e}")
            print(traceback.format_exc())
            history.append([message, error_msg])
            return history, ""

    def _format_deepseek_response(self, response: str) -> str:
        """Format DeepSeek R1 response với thinking process"""
        try:
            if "<think>" in response and "</think>" in response:
                parts = response.split("</think>", 1)
                if len(parts) == 2:
                    thinking = parts[0].replace("<think>", "").strip()
                    answer = parts[1].strip()

                    return f"""🧠 **Reasoning Process:**
{thinking}

💡 **Answer:**
{answer}"""
                else:
                    return response.replace("<think>", "🧠 **Thinking:** ").replace("</think>", "\n\n💡 **Answer:** ")
        except Exception as e:
            print(f"Format error: {e}")

        return response

    def clear_current_conversation(self):
        """Xóa cuộc hội thoại hiện tại"""
        self.conversations[self.current_conversation_id]["messages"] = []
        return [], "🗑️ Conversation cleared"

    def delete_conversation(self, conv_id: str):
        """Xóa cuộc hội thoại"""
        if conv_id in self.conversations and len(self.conversations) > 1:
            del self.conversations[conv_id]
            if conv_id == self.current_conversation_id:
                self.current_conversation_id = list(self.conversations.keys())[0]
            return "✅ Conversation deleted"
        return "❌ Cannot delete conversation"


# Initialize chatbot
chatbot = ChatGPTLikeRAGBot()

# CSS cho giao diện ChatGPT-like
chatgpt_css = """
/* Import Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Global Variables */
:root {
    --primary-color: #10a37f;
    --secondary-color: #19c37d;
    --bg-color: #f7f7f8;
    --sidebar-color: #ffffff;
    --text-color: #374151;
    --border-color: #e5e7eb;
    --hover-color: #f3f4f6;
    --shadow: 0 1px 3px rgba(0,0,0,0.1);
    --radius: 8px;
}

/* Dark Mode */
.dark {
    --bg-color: #1a1a1a;
    --sidebar-color: #2d2d2d;
    --text-color: #ffffff;
    --border-color: #404040;
    --hover-color: #404040;
}

/* Main Container */
.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    max-width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
    background: var(--bg-color) !important;
    min-height: 100vh;
}

/* Header */
.header {
    background: var(--sidebar-color);
    border-bottom: 1px solid var(--border-color);
    padding: 1rem 2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: var(--shadow);
}

.header h1 {
    color: var(--text-color);
    font-size: 1.5rem;
    font-weight: 600;
    margin: 0;
}

/* Layout */
.main-layout {
    display: flex;
    height: calc(100vh - 80px);
}

/* Sidebar */
.sidebar {
    width: 280px;
    background: var(--sidebar-color);
    border-right: 1px solid var(--border-color);
    padding: 1rem;
    overflow-y: auto;
}

.sidebar .model-selector {
    margin-bottom: 1.5rem;
}

.sidebar .conversations {
    margin-top: 1rem;
}

.conversation-item {
    padding: 0.75rem;
    margin: 0.25rem 0;
    border-radius: var(--radius);
    cursor: pointer;
    transition: all 0.2s;
    border: 1px solid transparent;
}

.conversation-item:hover {
    background: var(--hover-color);
    border-color: var(--border-color);
}

.conversation-item.active {
    background: var(--primary-color);
    color: white;
}

/* Chat Area */
.chat-container {
    flex: 1;
    display: flex;
    flex-direction: column;
    background: var(--bg-color);
}

.chatbot {
    flex: 1;
    border: none !important;
    background: transparent !important;
    font-family: inherit !important;
}

.chatbot .message {
    max-width: 768px;
    margin: 0 auto 1.5rem auto;
    padding: 0 2rem;
}

.chatbot .user {
    background: var(--primary-color) !important;
    color: white !important;
    border-radius: 18px !important;
    padding: 12px 18px !important;
    margin-left: auto !important;
    margin-right: 0 !important;
    max-width: 70% !important;
    word-wrap: break-word;
}

.chatbot .bot {
    background: var(--sidebar-color) !important;
    color: var(--text-color) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 12px !important;
    padding: 16px 20px !important;
    margin-right: auto !important;
    margin-left: 0 !important;
    max-width: 85% !important;
    box-shadow: var(--shadow);
}

/* Input Area */
.input-container {
    padding: 1rem 2rem 2rem 2rem;
    background: var(--bg-color);
    border-top: 1px solid var(--border-color);
}

.input-row {
    max-width: 768px;
    margin: 0 auto;
    display: flex;
    gap: 0.5rem;
    align-items: flex-end;
}

.message-input {
    flex: 1;
    border: 1px solid var(--border-color) !important;
    border-radius: 24px !important;
    padding: 12px 20px !important;
    background: var(--sidebar-color) !important;
    color: var(--text-color) !important;
    font-size: 14px;
    resize: none;
    max-height: 120px;
    transition: all 0.2s;
}

.message-input:focus {
    outline: none !important;
    border-color: var(--primary-color) !important;
    box-shadow: 0 0 0 3px rgba(16, 163, 127, 0.1) !important;
}

.send-button {
    background: var(--primary-color) !important;
    border: none !important;
    border-radius: 50% !important;
    width: 40px !important;
    height: 40px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
}

.send-button:hover {
    background: var(--secondary-color) !important;
    transform: scale(1.05);
}

/* Controls */
.controls {
    display: flex;
    gap: 0.5rem;
    margin-bottom: 1rem;
}

.control-button {
    padding: 0.5rem 1rem !important;
    border: 1px solid var(--border-color) !important;
    border-radius: var(--radius) !important;
    background: var(--sidebar-color) !important;
    color: var(--text-color) !important;
    font-size: 0.875rem !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
}

.control-button:hover {
    background: var(--hover-color) !important;
    border-color: var(--primary-color) !important;
}

/* Status indicators */
.status-indicator {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 1rem;
    border-radius: 20px;
    font-size: 0.875rem;
    font-weight: 500;
}

.status-loading {
    background: #fef3c7;
    color: #92400e;
}

.status-ready {
    background: #d1fae5;
    color: #065f46;
}

.status-error {
    background: #fee2e2;
    color: #991b1b;
}

/* Examples */
.examples {
    max-width: 768px;
    margin: 2rem auto;
    padding: 0 2rem;
}

.example-item {
    background: var(--sidebar-color);
    border: 1px solid var(--border-color);
    border-radius: var(--radius);
    padding: 1rem;
    margin: 0.5rem 0;
    cursor: pointer;
    transition: all 0.2s;
}

.example-item:hover {
    border-color: var(--primary-color);
    box-shadow: var(--shadow);
}

/* Responsive */
@media (max-width: 768px) {
    .main-layout {
        flex-direction: column;
    }

    .sidebar {
        width: 100%;
        height: auto;
        border-right: none;
        border-bottom: 1px solid var(--border-color);
    }

    .chatbot .message {
        padding: 0 1rem;
    }

    .input-container {
        padding: 1rem;
    }
}

/* Loading Animation */
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

.loading {
    animation: pulse 2s infinite;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 6px;
}

::-webkit-scrollbar-track {
    background: var(--bg-color);
}

::-webkit-scrollbar-thumb {
    background: var(--border-color);
    border-radius: 3px;
}

::-webkit-scrollbar-thumb:hover {
    background: var(--text-color);
}
"""


# Tạo giao diện Gradio
def create_interface():
    with gr.Blocks(
            title="ChatGPT-like RAG Assistant",
            css=chatgpt_css,
            theme=gr.themes.Soft(
                primary_hue="emerald",
                secondary_hue="emerald",
                neutral_hue="gray"
            )
    ) as app:

        # State variables
        conversation_state = gr.State(chatbot.current_conversation_id)

        # Header
        with gr.Row(elem_classes=["header"]):
            gr.Markdown("# 🤖 RAG Medical Assistant")
            if not RAG_AVAILABLE:
                gr.Markdown("⚠️ **Development Mode**")

        # Main layout
        with gr.Row(elem_classes=["main-layout"]):
            # Sidebar
            with gr.Column(scale=1, elem_classes=["sidebar"]):
                gr.Markdown("### 🎯 Model Selection")

                model_dropdown = gr.Dropdown(
                    choices=[
                        ("🦙 Llama 3", "llama3"),
                        ("🌟 Mistral", "mistral"),
                        ("🧠 DeepSeek R1", "deepseek-r1")
                    ],
                    value="llama3",
                    label="Select Model",
                    elem_classes=["model-selector"]
                )

                model_status = gr.Textbox(
                    label="Model Status",
                    value="🔄 Ready to load",
                    interactive=False,
                    elem_classes=["status-indicator"]
                )

                gr.Markdown("### 💬 Conversations")

                new_chat_btn = gr.Button(
                    "➕ New Chat",
                    variant="primary",
                    elem_classes=["control-button"]
                )

                conversation_list = gr.Dropdown(
                    choices=chatbot.get_conversation_list(),
                    label="Select Conversation",
                    value=chatbot.current_conversation_id
                )

                with gr.Row():
                    clear_btn = gr.Button("🗑️ Clear", elem_classes=["control-button"])
                    delete_btn = gr.Button("❌ Delete", elem_classes=["control-button"])

            # Chat area
            with gr.Column(scale=3, elem_classes=["chat-container"]):
                # Chat interface
                chat_interface = gr.Chatbot(
                    label="",
                    height=500,
                    show_copy_button=True,
                    elem_classes=["chatbot"],
                    avatar_images=("🧑‍💻", "🤖")
                )

                # Input area
                with gr.Row(elem_classes=["input-container"]):
                    with gr.Column(scale=1, elem_classes=["input-row"]):
                        msg_input = gr.Textbox(
                            placeholder="Type your message here...",
                            container=False,
                            lines=1,
                            max_lines=4,
                            elem_classes=["message-input"]
                        )

                        with gr.Row():
                            send_btn = gr.Button(
                                "🚀",
                                variant="primary",
                                elem_classes=["send-button"]
                            )

                # Examples
                with gr.Column(elem_classes=["examples"]):
                    gr.Examples(
                        examples=[
                            ["Hello, I'm having a headache, what should I do?"],
                            ["What are the benefits of regular exercise?"],
                            ["How can I improve my sleep quality?"],
                            ["What should I know about diabetes management?"],
                            ["Tell me about healthy eating habits"]
                        ],
                        inputs=msg_input,
                        elem_classes=["example-item"]
                    )

        # Event handlers
        def handle_model_change(model_name):
            status = chatbot.change_model(model_name)
            return status

        def handle_new_chat():
            new_conv_id = chatbot.create_new_conversation()
            chatbot.current_conversation_id = new_conv_id
            updated_list = chatbot.get_conversation_list()
            return [], "✨ New conversation started", updated_list, new_conv_id

        def handle_conversation_switch(conv_id):
            if conv_id:
                history, status = chatbot.switch_conversation(conv_id)
                return history, status, conv_id
            return [], "No conversation selected", conv_id

        def handle_chat(message, history):
            return chatbot.chat(message, history)

        def handle_clear():
            return chatbot.clear_current_conversation()

        # Wire up events
        model_dropdown.change(
            fn=handle_model_change,
            inputs=[model_dropdown],
            outputs=[model_status]
        )

        new_chat_btn.click(
            fn=handle_new_chat,
            outputs=[chat_interface, model_status, conversation_list, conversation_state]
        )

        conversation_list.change(
            fn=handle_conversation_switch,
            inputs=[conversation_list],
            outputs=[chat_interface, model_status, conversation_state]
        )

        msg_input.submit(
            fn=handle_chat,
            inputs=[msg_input, chat_interface],
            outputs=[chat_interface, msg_input]
        )

        send_btn.click(
            fn=handle_chat,
            inputs=[msg_input, chat_interface],
            outputs=[chat_interface, msg_input]
        )

        clear_btn.click(
            fn=handle_clear,
            outputs=[chat_interface, model_status]
        )

        # Load initial model on startup
        app.load(
            fn=lambda: chatbot.model_manager.load_model_sync("llama3")[1],
            outputs=[model_status]
        )

    return app


if __name__ == "__main__":
    print("🚀 Starting ChatGPT-like RAG Assistant...")
    app = create_interface()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        debug=True,
        show_error=True
    )