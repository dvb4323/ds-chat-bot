import subprocess
import psutil

def is_model_running(model_name: str) -> bool:
    """
    Kiểm tra xem mô hình có đang chạy bằng cách tìm tiến trình Ollama đang sử dụng tên mô hình.
    """
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'ollama' in proc.info['name'].lower() and model_name in ' '.join(proc.info['cmdline']):
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return False

def start_model(model_name: str):
    if not is_model_running(model_name):
        print(f"🟢 Đang khởi động mô hình {model_name} bằng Ollama...")
        subprocess.Popen(["ollama", "run", model_name])
    else:
        print(f"✅ Mô hình {model_name} đã đang chạy.")
