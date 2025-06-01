#!/usr/bin/env python3
"""
Optimized RAG Chatbot Main Runner - FIXED VERSION
ChatGPT-like interface with enhanced performance and Windows compatibility
"""

import os
import sys
import signal
import atexit
import shutil
from pathlib import Path

# Fix Unicode encoding for Windows
if sys.platform.startswith('win'):
    import codecs

    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import gradio as gr
import threading
import time
from typing import Dict, List, Tuple, Optional
import logging

# Import configurations and components
from config import get_config

# Safe imports with fallbacks
try:
    from utils.model_manager import AsyncModelManager

    MODEL_MANAGER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Model manager not available: {e}")
    MODEL_MANAGER_AVAILABLE = False

try:
    from app.enhanced_rag_pipeline import get_rag_pipeline, shutdown_rag_pipeline

    ENHANCED_RAG_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Enhanced RAG not available: {e}")
    ENHANCED_RAG_AVAILABLE = False
    try:
        from app.rag_pipeline import load_rag_chain

        RAG_AVAILABLE = True
    except ImportError:
        RAG_AVAILABLE = False


# Setup logging with Windows compatibility
def setup_safe_logging():
    """Setup logging that works on Windows"""
    try:
        config = get_config()
        log_config = config.get("logging", {})
        level = getattr(logging, log_config.get("level", "INFO").upper())

        # Create formatter without emojis for Windows compatibility
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # Console handler with UTF-8 encoding
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level)

        # File handler if specified
        log_file = log_config.get("file")
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(
                log_file, encoding='utf-8', mode='a'
            )
            file_handler.setFormatter(formatter)
            file_handler.setLevel(level)

        # Setup root logger
        logger = logging.getLogger()
        logger.setLevel(level)
        logger.addHandler(console_handler)
        if log_file:
            logger.addHandler(file_handler)

        return logging.getLogger(__name__)

    except Exception as e:
        # Fallback to basic logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)


logger = setup_safe_logging()


def fix_chromadb_corruption(db_path: str) -> bool:
    """Fix corrupted ChromaDB by recreating it"""
    try:
        db_path = Path(db_path)
        if db_path.exists():
            logger.warning(f"ChromaDB appears corrupted at {db_path}. Attempting to fix...")

            # Backup corrupted DB
            backup_path = db_path.parent / f"{db_path.name}_corrupted_backup_{int(time.time())}"
            shutil.move(str(db_path), str(backup_path))
            logger.info(f"Corrupted DB backed up to: {backup_path}")

            # Create new directory
            db_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created new ChromaDB directory at: {db_path}")

            return True

    except Exception as e:
        logger.error(f"Failed to fix ChromaDB corruption: {e}")
        return False

    return True


class SimplifiedChatGPTRAG:
    """Simplified application class with Windows compatibility"""

    def __init__(self):
        self.config = get_config()

        # Initialize components with fallbacks
        if MODEL_MANAGER_AVAILABLE:
            self.model_manager = AsyncModelManager(
                max_concurrent_loads=self.config.get("performance.max_concurrent_model_loads", 2)
            )
        else:
            self.model_manager = None

        # Try to initialize RAG with error handling
        self.rag_pipeline = None
        self.rag_available = False

        try:
            if ENHANCED_RAG_AVAILABLE:
                self.rag_pipeline = get_rag_pipeline()
                self.rag_available = True
                logger.info("Enhanced RAG pipeline initialized successfully")
            elif RAG_AVAILABLE:
                logger.info("Using fallback RAG pipeline")
                self.rag_available = True
            else:
                logger.warning("RAG pipeline not available, using mock responses")

        except Exception as e:
            logger.error(f"RAG initialization failed: {e}")

            # Try to fix ChromaDB corruption
            if "file is not a database" in str(e):
                db_path = self.config.get("database.chroma_db_path", "./db/chroma_db")
                if fix_chromadb_corruption(db_path):
                    try:
                        if ENHANCED_RAG_AVAILABLE:
                            self.rag_pipeline = get_rag_pipeline()
                            self.rag_available = True
                            logger.info("RAG pipeline recovered after fixing ChromaDB")
                    except Exception as retry_e:
                        logger.error(f"RAG recovery failed: {retry_e}")

        # UI state - Fix thứ tự khởi tạo
        self.current_model = self._get_default_model()  # ✅ Di chuyển lên trước
        self.conversations = {}
        self.current_conversation_id = self._create_new_conversation()  # ✅ Giờ OK

        # Performance monitoring
        self.start_time = time.time()
        self.total_queries = 0
        self.total_response_time = 0

        # Register cleanup
        atexit.register(self.cleanup)

        logger.info("Simplified ChatGPT RAG initialized successfully")

    def _get_default_model(self) -> str:
        """Get default model from config"""
        enabled_models = self.config.get_enabled_models()
        return enabled_models[0] if enabled_models else "llama3"

    def _create_new_conversation(self) -> str:
        """Create new conversation"""
        conv_id = f"conv_{int(time.time())}"
        self.conversations[conv_id] = {
            "messages": [],
            "model": self.current_model,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "title": "New Chat",
            "query_count": 0
        }
        return conv_id

    def get_conversation_list(self) -> List[Tuple[str, str]]:
        """Get formatted conversation list"""
        conversations = []
        max_conversations = self.config.get("ui.max_conversations", 50)

        # Sort by creation time (newest first)
        sorted_convs = sorted(
            self.conversations.items(),
            key=lambda x: x[1]["created_at"],
            reverse=True
        )

        for conv_id, data in sorted_convs[:max_conversations]:
            title = data.get("title", "New Chat")
            if len(title) > 35:
                title = title[:35] + "..."

            query_count = data.get("query_count", 0)
            display_text = f"{title} ({query_count} msgs)"
            conversations.append((conv_id, display_text))

        return conversations

    def switch_conversation(self, conv_id: str) -> Tuple[List, str]:
        """Switch to different conversation"""
        if conv_id and conv_id in self.conversations:
            self.current_conversation_id = conv_id
            conv_data = self.conversations[conv_id]
            self.current_model = conv_data.get("model", self.current_model)

            # Convert messages to Gradio format
            history = []
            messages = conv_data.get("messages", [])

            for i in range(0, len(messages), 2):
                if i + 1 < len(messages):
                    user_msg = messages[i]["content"]
                    assistant_msg = messages[i + 1]["content"]
                    history.append([user_msg, assistant_msg])

            return history, f"Switched to: {conv_data.get('title', 'Chat')}"

        return [], "Conversation not found"

    def start_new_conversation(self) -> Tuple[List, str, List, str]:
        """Start new conversation"""
        self.current_conversation_id = self._create_new_conversation()
        updated_list = self.get_conversation_list()
        return [], "New conversation started", updated_list, self.current_conversation_id

    def change_model(self, model_name: str) -> str:
        """Change current model"""
        if model_name in self.config.get_enabled_models():
            self.current_model = model_name

            # Update current conversation model
            if self.current_conversation_id in self.conversations:
                self.conversations[self.current_conversation_id]["model"] = model_name

            return f"Switched to {model_name}"

        return f"Model {model_name} not available"

    def _mock_rag_response(self, message: str) -> str:
        """Generate mock response when RAG is not available"""
        time.sleep(1)  # Simulate processing

        if "deepseek-r1" in self.current_model.lower():
            return f"""<think>
The user is asking: {message}

I should provide a helpful response based on medical knowledge while noting this is a development/demo mode.
</think>

I understand your question about: {message}

This is a development mode response since the RAG system is not fully initialized. In production mode, I would search through medical documents to provide evidence-based answers.

For medical questions, I always recommend:
- Consulting healthcare professionals for personalized advice
- Seeking emergency care for urgent symptoms
- Following evidence-based medical guidelines

Please note: This response is generated in development mode. The full RAG system would provide more comprehensive, document-backed answers.

Response time: simulated"""
        else:
            return f"""Thank you for your question: {message}

I'm currently running in development mode since the RAG pipeline is not fully available. 

In full mode, I would:
- Search through medical documents
- Provide evidence-based information
- Include relevant source citations
- Offer comprehensive health guidance

For now, I recommend consulting healthcare professionals for medical advice.

Model: {self.current_model} (mock mode)
Response time: simulated"""

    def process_chat(self, message: str, history: List) -> Tuple[List, str]:
        """Process chat message"""
        if not message or not message.strip():
            return history, ""

        start_time = time.time()

        try:
            response = ""

            if self.rag_available and self.rag_pipeline:
                # Use real RAG pipeline
                try:
                    if hasattr(self.rag_pipeline, 'query'):
                        result = self.rag_pipeline.query(
                            model_name=self.current_model,
                            question=message,
                            include_sources=self.config.get("ui.show_source_documents", True)
                        )
                        response = result.get("response", "No response generated")
                    else:
                        # Fallback to basic RAG
                        chain = load_rag_chain(self.current_model)
                        result = chain({"query": message})
                        response = result.get("result", "No response generated")

                        # Add timing info
                        query_time = time.time() - start_time
                        response += f"\n\nResponse time: {query_time:.2f}s"

                except Exception as rag_error:
                    logger.error(f"RAG query failed: {rag_error}")
                    response = f"RAG system error: {str(rag_error)}\n\nFalling back to mock response:\n\n"
                    response += self._mock_rag_response(message)
            else:
                # Use mock response
                response = self._mock_rag_response(message)

            # Handle DeepSeek-R1 special format
            if "<think>" in response and "</think>" in response:
                response = self._format_deepseek_response(response)

            # Update conversation
            conv = self.conversations[self.current_conversation_id]
            conv["messages"].extend([
                {"role": "user", "content": message, "timestamp": start_time},
                {"role": "assistant", "content": response, "timestamp": time.time()}
            ])
            conv["query_count"] += 1

            # Update conversation title if first message
            if conv["title"] == "New Chat" and conv["query_count"] == 1:
                conv["title"] = message[:50] + ("..." if len(message) > 50 else "")

            # Update performance stats
            self.total_queries += 1
            total_time = time.time() - start_time
            self.total_response_time += total_time

            # Add to history
            history.append([message, response])

            logger.info(f"Query processed in {total_time:.2f}s")

            return history, ""

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            logger.error(f"Chat processing error: {e}")
            history.append([message, error_msg])
            return history, ""

    def _format_deepseek_response(self, response: str) -> str:
        """Format DeepSeek R1 response with thinking process"""
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
            logger.error(f"Error formatting DeepSeek response: {e}")

        return response

    def clear_conversation(self) -> Tuple[List, str]:
        """Clear current conversation"""
        if self.current_conversation_id in self.conversations:
            self.conversations[self.current_conversation_id]["messages"] = []
            self.conversations[self.current_conversation_id]["query_count"] = 0
            return [], "Conversation cleared"
        return [], "No active conversation"

    def delete_conversation(self, conv_id: str) -> Tuple[str, List]:
        """Delete conversation"""
        if conv_id in self.conversations and len(self.conversations) > 1:
            del self.conversations[conv_id]

            # Switch to another conversation if current was deleted
            if conv_id == self.current_conversation_id:
                remaining_convs = list(self.conversations.keys())
                self.current_conversation_id = remaining_convs[0]

            updated_list = self.get_conversation_list()
            return "Conversation deleted", updated_list

        return "Cannot delete conversation", self.get_conversation_list()

    def get_system_status(self) -> str:
        """Get comprehensive system status"""
        try:
            # Basic status
            uptime = time.time() - self.start_time
            avg_response_time = (self.total_response_time / self.total_queries
                                 if self.total_queries > 0 else 0)

            status_text = f"""**System Status**

**RAG System:** {'Available' if self.rag_available else 'Mock Mode'}
**Model Manager:** {'Available' if MODEL_MANAGER_AVAILABLE else 'Basic Mode'}

**Performance:**
• Uptime: {uptime / 3600:.1f} hours
• Total queries: {self.total_queries}
• Avg response time: {avg_response_time:.2f}s

**Current:**
• Active model: {self.current_model}
• Conversations: {len(self.conversations)}
• Current conversation: {self.conversations[self.current_conversation_id].get('title', 'New Chat')}

**Database Path:** {self.config.get('database.chroma_db_path', './db/chroma_db')}
**Server:** {self.config.get('server.host')}:{self.config.get('server.port')}"""

            return status_text

        except Exception as e:
            return f"Error getting system status: {str(e)}"

    def cleanup(self):
        """Cleanup resources on shutdown"""
        logger.info("Starting application cleanup...")

        try:
            # Shutdown components if available
            if self.model_manager and hasattr(self.model_manager, 'shutdown'):
                self.model_manager.shutdown()

            if ENHANCED_RAG_AVAILABLE:
                shutdown_rag_pipeline()

            logger.info("Cleanup completed successfully")

        except Exception as e:
            logger.error(f"Cleanup error: {e}")


def create_gradio_interface(app: SimplifiedChatGPTRAG) -> gr.Blocks:
    """Create simplified but functional Gradio interface"""

    # Simplified CSS without complex animations
    simple_css = """
    .gradio-container {
        font-family: system-ui, -apple-system, sans-serif !important;
        max-width: 1200px !important;
        margin: 0 auto !important;
    }

    .chatbot {
        height: 500px !important;
        border-radius: 8px !important;
        border: 1px solid #e5e7eb !important;
    }

    .chatbot .user {
        background: #10a37f !important;
        color: white !important;
        border-radius: 18px !important;
        padding: 12px 18px !important;
        margin-left: auto !important;
        max-width: 70% !important;
    }

    .chatbot .bot {
        background: #f8f9fa !important;
        color: #333 !important;
        border: 1px solid #e9ecef !important;
        border-radius: 12px !important;
        padding: 16px 20px !important;
        margin-right: auto !important;
        max-width: 85% !important;
    }

    .input-row {
        display: flex;
        gap: 0.5rem;
        align-items: flex-end;
    }

    .message-input {
        flex: 1;
        min-height: 44px;
        border-radius: 22px;
        padding: 12px 20px;
        border: 1px solid #d1d5db;
    }

    .send-button {
        width: 44px;
        height: 44px;
        border-radius: 22px;
        background: #10a37f;
        color: white;
        border: none;
    }
    """

    # Create interface
    with gr.Blocks(
            title="RAG Medical Assistant",
            css=simple_css,
            theme=gr.themes.Soft(primary_hue="emerald")
    ) as interface:

        # State management
        conversation_state = gr.State(app.current_conversation_id)

        # Header
        gr.Markdown("# 🤖 RAG Medical Assistant")

        if not app.rag_available:
            gr.Markdown("⚠️ **Running in Mock Mode** - RAG system not fully available")

        # Layout
        with gr.Row():
            # Sidebar
            with gr.Column(scale=1):
                gr.Markdown("### Model Selection")

                enabled_models = app.config.get_enabled_models()
                model_choices = [(app.config.get_model_config(m).get("name", m), m)
                                 for m in enabled_models]

                model_dropdown = gr.Dropdown(
                    choices=model_choices,
                    value=app.current_model,
                    label="Select Model"
                )

                model_status = gr.Textbox(
                    label="Status",
                    value="Ready",
                    interactive=False
                )

                gr.Markdown("### Conversations")

                new_chat_btn = gr.Button("➕ New Chat", variant="primary")

                conversation_dropdown = gr.Dropdown(
                    choices=app.get_conversation_list(),
                    value=app.current_conversation_id,
                    label="Select Conversation"
                )

                with gr.Row():
                    clear_btn = gr.Button("🗑️ Clear")
                    delete_btn = gr.Button("❌ Delete")

                gr.Markdown("### System")
                status_btn = gr.Button("📋 Show Status")

                system_status = gr.Textbox(
                    label="System Status",
                    value="Click 'Show Status' for details",
                    interactive=False,
                    max_lines=15
                )

            # Chat Area
            with gr.Column(scale=3):
                # Chat Interface - Fix deprecation warning
                chat_interface = gr.Chatbot(
                    label="Chat",
                    height=500,
                    show_copy_button=True,
                    avatar_images=("👤", "🤖"),
                    type="messages"  # Fix deprecation warning
                )

                # Input Area
                with gr.Row(elem_classes=["input-row"]):
                    msg_input = gr.Textbox(
                        placeholder="Type your message here...",
                        container=False,
                        lines=1,
                        max_lines=4,
                        elem_classes=["message-input"]
                    )

                    send_btn = gr.Button("🚀", elem_classes=["send-button"])

                # Examples
                gr.Examples(
                    examples=[
                        ["Hello, I'm having a headache, what should I do?"],
                        ["What are the benefits of regular exercise?"],
                        ["How can I improve my sleep quality?"],
                        ["What should I know about diabetes management?"],
                        ["Tell me about healthy eating habits"]
                    ],
                    inputs=msg_input
                )

        # Event Handlers
        def handle_model_change(model_name):
            return app.change_model(model_name)

        def handle_new_chat():
            history, status, conv_list, conv_id = app.start_new_conversation()
            return history, status, conv_list, conv_id

        def handle_conversation_switch(conv_id):
            history, status = app.switch_conversation(conv_id)
            return history, status, conv_id

        def handle_chat(message, history):
            return app.process_chat(message, history)

        def handle_clear():
            history, status = app.clear_conversation()
            conv_list = app.get_conversation_list()
            return history, status, conv_list

        def handle_delete(conv_id):
            status, conv_list = app.delete_conversation(conv_id)
            if conv_list:
                new_conv_id = conv_list[0][0]
                history, _ = app.switch_conversation(new_conv_id)
                return history, status, conv_list, new_conv_id
            else:
                return handle_new_chat()

        def handle_status_check():
            return app.get_system_status()

        # Wire up events
        model_dropdown.change(
            fn=handle_model_change,
            inputs=[model_dropdown],
            outputs=[model_status]
        )

        new_chat_btn.click(
            fn=handle_new_chat,
            outputs=[chat_interface, model_status, conversation_dropdown, conversation_state]
        )

        conversation_dropdown.change(
            fn=handle_conversation_switch,
            inputs=[conversation_dropdown],
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
            outputs=[chat_interface, model_status, conversation_dropdown]
        )

        delete_btn.click(
            fn=handle_delete,
            inputs=[conversation_state],
            outputs=[chat_interface, model_status, conversation_dropdown, conversation_state]
        )

        status_btn.click(
            fn=handle_status_check,
            outputs=[system_status]
        )

        # Auto-load default model on startup
        interface.load(
            fn=lambda: app.change_model(app.current_model),
            outputs=[model_status]
        )

    return interface


def main():
    """Main entry point with error handling"""
    try:
        # Initialize application
        print("Starting RAG Medical Assistant...")

        # Validate configuration
        config = get_config()
        is_valid, errors = config.validate_config()
        if not is_valid:
            print("Configuration validation failed:")
            for error in errors:
                print(f"  • {error}")
            print("Continuing with default configuration...")

        # Create application
        app = SimplifiedChatGPTRAG()

        # Create interface
        interface = create_gradio_interface(app)

        # Get server configuration
        server_config = config.get_server_config()

        # Launch application
        print(f"Launching server on {server_config.get('host')}:{server_config.get('port')}")

        interface.launch(
            server_name=server_config.get("host", "127.0.0.1"),
            server_port=server_config.get("port", 7860),
            inbrowser=True,
            debug=server_config.get("debug", False),
            show_error=True,
            share=False
        )

    except KeyboardInterrupt:
        print("Received keyboard interrupt")
    except Exception as e:
        print(f"Application startup failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Application shutdown complete")


if __name__ == "__main__":
    main()