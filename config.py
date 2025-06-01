"""
Simplified Configuration for RAG Chatbot
Windows-compatible, fallback-friendly version
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List

class SimpleConfig:
    """Simplified configuration management"""

    # Default configuration
    DEFAULT_CONFIG = {
        "models": {
            "llama3": {
                "name": "Llama 3",
                "enabled": True,
                "max_tokens": 2048,
                "temperature": 0.7
            },
            "mistral": {
                "name": "Mistral",
                "enabled": True,
                "max_tokens": 2048,
                "temperature": 0.7
            },
            "deepseek-r1": {
                "name": "DeepSeek R1",
                "enabled": True,
                "max_tokens": 4096,
                "temperature": 0.3
            }
        },

        "database": {
            "chroma_db_path": "./db/chroma_db",
            "collection_name": "healthcare_chunks",
            "embedding_model": "all-MiniLM-L6-v2"
        },

        "server": {
            "host": "127.0.0.1",
            "port": 7860,
            "debug": True,
            "auto_reload": True
        },

        "ui": {
            "theme": "soft",
            "dark_mode": False,
            "max_conversations": 50,
            "max_messages_per_conversation": 100,
            "show_source_documents": True,
            "show_timing_info": True
        },

        "performance": {
            "max_concurrent_model_loads": 2,
            "model_idle_timeout": 3600,
            "auto_cleanup_enabled": True,
            "memory_monitoring": True,
            "health_check_interval": 30
        },

        "rag": {
            "retrieval_top_k": 5,
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "similarity_threshold": 0.7
        },

        "logging": {
            "level": "INFO",
            "file": "./logs/chatbot.log",
            "max_file_size": "10MB",
            "backup_count": 5
        }
    }

    def __init__(self, config_file: str = "config.json"):
        self.config_file = Path(config_file)
        self.config = self.DEFAULT_CONFIG.copy()
        self._load_config_safe()

    def _load_config_safe(self):
        """Load configuration with error handling"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)

                # Simple merge
                self._update_config(self.config, user_config)
                print(f"Configuration loaded from {self.config_file}")
            else:
                print(f"Creating default config at {self.config_file}")
                self._save_config_safe()

        except Exception as e:
            print(f"Warning: Error loading config: {e}. Using defaults.")
            self.config = self.DEFAULT_CONFIG.copy()

    def _save_config_safe(self):
        """Save configuration with error handling"""
        try:
            # Create directory if not exists
            self.config_file.parent.mkdir(parents=True, exist_ok=True)

            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)

            print(f"Configuration saved to {self.config_file}")

        except Exception as e:
            print(f"Warning: Error saving config: {e}")

    def _update_config(self, base: Dict, update: Dict):
        """Simple recursive config update"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._update_config(base[key], value)
            else:
                base[key] = value

    def get(self, key_path: str, default=None):
        """Get configuration value using dot notation"""
        keys = key_path.split('.')
        value = self.config

        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key_path: str, value: Any):
        """Set configuration value using dot notation"""
        keys = key_path.split('.')
        config = self.config

        # Navigate to parent
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]

        # Set final value
        config[keys[-1]] = value

    def get_model_config(self, model_name: str) -> Dict:
        """Get configuration for specific model"""
        return self.get(f"models.{model_name}", {})

    def get_enabled_models(self) -> List[str]:
        """Get list of enabled models"""
        enabled = []
        models = self.get("models", {})
        for model_name, config in models.items():
            if config.get("enabled", True):
                enabled.append(model_name)
        return enabled

    def get_database_config(self) -> Dict:
        """Get database configuration"""
        return self.get("database", {})

    def get_server_config(self) -> Dict:
        """Get server configuration"""
        return self.get("server", {})

    def get_ui_config(self) -> Dict:
        """Get UI configuration"""
        return self.get("ui", {})

    def get_performance_config(self) -> Dict:
        """Get performance configuration"""
        return self.get("performance", {})

    def get_rag_config(self) -> Dict:
        """Get RAG configuration"""
        return self.get("rag", {})

    def validate_config(self) -> tuple:
        """Basic configuration validation"""
        errors = []

        try:
            # Validate database path
            db_path = Path(self.get("database.chroma_db_path", "./db/chroma_db"))
            if not db_path.parent.exists():
                try:
                    db_path.parent.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    errors.append(f"Cannot create database directory: {e}")

            # Validate models
            enabled_models = self.get_enabled_models()
            if not enabled_models:
                errors.append("No models enabled")

            # Validate server config
            port = self.get("server.port", 7860)
            if not isinstance(port, int) or port < 1 or port > 65535:
                errors.append(f"Invalid port number: {port}")

        except Exception as e:
            errors.append(f"Validation error: {e}")

        return len(errors) == 0, errors

    def export_config(self, file_path: str):
        """Export configuration to file"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            print(f"Configuration exported to {file_path}")
        except Exception as e:
            print(f"Error exporting config: {e}")

    def import_config(self, file_path: str):
        """Import configuration from file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                imported_config = json.load(f)

            self._update_config(self.config, imported_config)
            self._save_config_safe()
            print(f"Configuration imported from {file_path}")

        except Exception as e:
            print(f"Error importing config: {e}")


# Global configuration instance
_config_instance = None

def get_config() -> SimpleConfig:
    """Get global configuration instance"""
    global _config_instance
    if _config_instance is None:
        _config_instance = SimpleConfig()
    return _config_instance

def reload_config():
    """Reload configuration from file"""
    global _config_instance
    if _config_instance:
        _config_instance._load_config_safe()

def setup_logging():
    """Simple logging setup"""
    import logging

    try:
        config = get_config()
        log_config = config.get("logging", {})
        level = getattr(logging, log_config.get("level", "INFO").upper(), logging.INFO)

        # Basic logging setup
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
            ]
        )

        # Add file handler if specified
        log_file = log_config.get("file")
        if log_file:
            try:
                log_path = Path(log_file)
                log_path.parent.mkdir(parents=True, exist_ok=True)

                file_handler = logging.FileHandler(log_file, encoding='utf-8')
                file_handler.setFormatter(
                    logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                )

                logger = logging.getLogger()
                logger.addHandler(file_handler)

            except Exception as e:
                print(f"Warning: Could not setup file logging: {e}")

        return logging.getLogger(__name__)

    except Exception as e:
        print(f"Warning: Logging setup failed: {e}")
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)