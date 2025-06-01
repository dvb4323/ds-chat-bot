import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional, Callable, Tuple
import psutil

try:
    from app.rag_pipeline import load_rag_chain
    from app.ollama_client import is_model_running, start_model

    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False


class AsyncModelManager:
    """
    Advanced model manager với async loading, caching và health monitoring
    """

    def __init__(self, max_concurrent_loads: int = 2):
        self.models = {
            "llama3": {
                "name": "Llama 3",
                "status": "unloaded",
                "chain": None,
                "last_used": None,
                "load_time": None,
                "memory_usage": 0
            },
            "mistral": {
                "name": "Mistral",
                "status": "unloaded",
                "chain": None,
                "last_used": None,
                "load_time": None,
                "memory_usage": 0
            },
            "deepseek-r1": {
                "name": "DeepSeek R1",
                "status": "unloaded",
                "chain": None,
                "last_used": None,
                "load_time": None,
                "memory_usage": 0
            }
        }

        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_loads)
        self.loading_futures = {}
        self.lock = threading.RLock()

        # Start health monitoring
        self._start_health_monitor()

    def _start_health_monitor(self):
        """Bắt đầu monitor sức khỏe của models"""

        def monitor_loop():
            while True:
                try:
                    self._check_model_health()
                    time.sleep(30)  # Check every 30 seconds
                except Exception as e:
                    print(f"Health monitor error: {e}")
                    time.sleep(60)

        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()

    def _check_model_health(self):
        """Kiểm tra sức khỏe của các models"""
        with self.lock:
            for model_name, info in self.models.items():
                if info["status"] == "loaded" and info["chain"]:
                    try:
                        # Check if Ollama process is still running
                        if RAG_AVAILABLE and not is_model_running(model_name):
                            print(f"⚠️ Model {model_name} process died, marking as unloaded")
                            info["status"] = "unloaded"
                            info["chain"] = None

                        # Update memory usage
                        info["memory_usage"] = self._get_model_memory_usage(model_name)

                    except Exception as e:
                        print(f"Health check failed for {model_name}: {e}")

    def _get_model_memory_usage(self, model_name: str) -> float:
        """Ước tính memory usage của model"""
        try:
            total_memory = 0
            for proc in psutil.process_iter(['pid', 'name', 'memory_info', 'cmdline']):
                try:
                    if ('ollama' in proc.info['name'].lower() and
                            model_name in ' '.join(proc.info['cmdline'])):
                        total_memory += proc.info['memory_info'].rss / 1024 / 1024  # MB
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            return total_memory
        except Exception:
            return 0

    async def load_model_async(self, model_name: str,
                               progress_callback: Optional[Callable] = None) -> Tuple[bool, str]:
        """Load model bất đồng bộ với progress tracking"""

        if model_name not in self.models:
            return False, f"Unknown model: {model_name}"

        with self.lock:
            # Check if already loaded
            if self.models[model_name]["status"] == "loaded":
                self.models[model_name]["last_used"] = time.time()
                return True, f"✅ Model {model_name} already loaded"

            # Check if already loading
            if model_name in self.loading_futures:
                if progress_callback:
                    progress_callback(f"🔄 Model {model_name} is already loading...")

                # Wait for existing load to complete
                try:
                    result = await asyncio.wrap_future(self.loading_futures[model_name])
                    return result
                except Exception as e:
                    return False, f"❌ Failed to wait for loading: {str(e)}"

        # Start new loading process
        loop = asyncio.get_event_loop()
        future = self.executor.submit(
            self._load_model_sync, model_name, progress_callback
        )

        with self.lock:
            self.loading_futures[model_name] = future

        try:
            result = await asyncio.wrap_future(future)
            return result
        finally:
            with self.lock:
                if model_name in self.loading_futures:
                    del self.loading_futures[model_name]

    def _load_model_sync(self, model_name: str,
                         progress_callback: Optional[Callable] = None) -> Tuple[bool, str]:
        """Load model đồng bộ (chạy trong thread pool)"""

        start_time = time.time()

        try:
            with self.lock:
                self.models[model_name]["status"] = "loading"

            if progress_callback:
                progress_callback(f"🔍 Checking {model_name} availability...")

            if RAG_AVAILABLE:
                if progress_callback:
                    progress_callback(f"🚀 Starting Ollama service for {model_name}...")

                # Start Ollama model
                start_model(model_name)

                if progress_callback:
                    progress_callback(f"📚 Loading embeddings and vector database...")

                # Load RAG chain
                chain = load_rag_chain(model_name)

                if progress_callback:
                    progress_callback(f"🔧 Initializing RAG pipeline...")

                # Test the chain
                test_result = chain({"query": "test"})
                if not test_result:
                    raise Exception("Chain test failed")

            else:
                # Mock chain for development
                if progress_callback:
                    progress_callback(f"🧪 Creating mock chain for {model_name}...")

                class MockChain:
                    def __call__(self, inputs):
                        query = inputs.get("query", "")
                        time.sleep(0.5)  # Simulate processing
                        return {
                            "result": f"Mock response from {model_name}: {query}",
                            "source_documents": []
                        }

                chain = MockChain()

            load_time = time.time() - start_time

            with self.lock:
                self.models[model_name].update({
                    "status": "loaded",
                    "chain": chain,
                    "last_used": time.time(),
                    "load_time": load_time,
                    "memory_usage": self._get_model_memory_usage(model_name)
                })

            success_msg = f"✅ {model_name} loaded successfully in {load_time:.1f}s"
            if progress_callback:
                progress_callback(success_msg)

            return True, success_msg

        except Exception as e:
            error_msg = f"❌ Failed to load {model_name}: {str(e)}"

            with self.lock:
                self.models[model_name]["status"] = "error"

            if progress_callback:
                progress_callback(error_msg)

            print(f"Model loading error: {e}")
            return False, error_msg

    def load_model_sync(self, model_name: str,
                        progress_callback: Optional[Callable] = None) -> Tuple[bool, str]:
        """Synchronous wrapper for loading models"""
        return self._load_model_sync(model_name, progress_callback)

    def get_model_chain(self, model_name: str):
        """Lấy chain của model, tự động load nếu cần"""
        with self.lock:
            if model_name not in self.models:
                return None

            model_info = self.models[model_name]

            # Auto-load if not loaded
            if model_info["status"] != "loaded":
                print(f"Auto-loading {model_name}...")
                success, msg = self._load_model_sync(model_name)
                if not success:
                    return None

            # Update last used time
            model_info["last_used"] = time.time()
            return model_info["chain"]

    def unload_model(self, model_name: str) -> bool:
        """Unload model để giải phóng memory"""
        with self.lock:
            if model_name in self.models:
                self.models[model_name].update({
                    "status": "unloaded",
                    "chain": None,
                    "memory_usage": 0
                })
                return True
            return False

    def get_model_status(self, model_name: str) -> Dict:
        """Lấy thông tin chi tiết về model"""
        with self.lock:
            if model_name not in self.models:
                return {"status": "unknown"}

            info = self.models[model_name].copy()

            # Add runtime info
            if info["last_used"]:
                info["idle_time"] = time.time() - info["last_used"]
            else:
                info["idle_time"] = None

            return info

    def get_all_models_status(self) -> Dict:
        """Lấy status của tất cả models"""
        with self.lock:
            return {
                name: self.get_model_status(name)
                for name in self.models.keys()
            }

    def cleanup_idle_models(self, max_idle_time: int = 3600):
        """Dọn dẹp models không sử dụng lâu (default: 1 hour)"""
        current_time = time.time()
        unloaded_count = 0

        with self.lock:
            for model_name, info in self.models.items():
                if (info["status"] == "loaded" and
                        info["last_used"] and
                        current_time - info["last_used"] > max_idle_time):
                    print(f"🧹 Unloading idle model: {model_name}")
                    self.unload_model(model_name)
                    unloaded_count += 1

        return unloaded_count

    def preload_models(self, model_names: list,
                       progress_callback: Optional[Callable] = None):
        """Preload multiple models concurrently"""

        async def preload_async():
            tasks = []
            for model_name in model_names:
                if model_name in self.models:
                    task = self.load_model_async(model_name, progress_callback)
                    tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)
            return results

        # Run in new event loop if needed
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(preload_async())

    def get_memory_usage_summary(self) -> Dict:
        """Lấy tóm tắt memory usage"""
        total_memory = 0
        loaded_models = 0

        with self.lock:
            for model_name, info in self.models.items():
                if info["status"] == "loaded":
                    total_memory += info["memory_usage"]
                    loaded_models += 1

        return {
            "total_memory_mb": total_memory,
            "loaded_models": loaded_models,
            "avg_memory_per_model": total_memory / loaded_models if loaded_models > 0 else 0
        }

    def shutdown(self):
        """Graceful shutdown"""
        print("🔄 Shutting down model manager...")

        # Unload all models
        with self.lock:
            for model_name in list(self.models.keys()):
                self.unload_model(model_name)

        # Shutdown executor
        self.executor.shutdown(wait=True)
        print("✅ Model manager shutdown complete")