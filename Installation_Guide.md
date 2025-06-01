# 🚀 RAG Medical Chatbot - Hướng dẫn cài đặt

## 📋 Yêu cầu hệ thống
- **Python:** 3.9 - 3.11
- **RAM:** 8GB minimum (16GB recommended)
- **Storage:** 4GB free space
- **OS:** Windows 10+, macOS 10.15+, Ubuntu 18.04+

## 🔧 Quy trình cài đặt

### **Bước 1: Cài đặt môi trường**

**Python 3.10:**
- Windows: Download from [python.org](https://python.org)
- macOS: `brew install python@3.10`
- Linux: `sudo apt install python3.10 python3.10-pip`

**Ollama:**
- Tải ollama tại [ollama.ai](https://ollama.ai) và cài đặt

### **Bước 2: Clone Project**
```bash
git clone https://github.com/dvb4323/ds-chat-bot
cd ds-chat-bot
```

### **Bước 3: Khởi tạo môi trường ảo**
```bash
# Create virtual environment
python -m venv .venv

# Activate environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### **Bước 4: Tải mô hình LLM**
```bash
# Download required models (will take 5-10 minutes)
ollama pull llama3
ollama pull mistral
ollama pull deepseek-r1
```

### **Bước 5: Chạy ứng dụng**
```bash
# Start the chatbot
python main_optimized.py
```

## 🌐 Truy cập ứng dụng
- Ứng dụng chạy trên local: `http://127.0.0.1:7860`

