import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging

from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.schema import Document

from app.ollama_client import get_ollama_model, start_model, is_model_running
from config import get_config

logger = logging.getLogger(__name__)


class EnhancedRAGPipeline:
    """Enhanced RAG Pipeline với caching, monitoring và optimization"""

    def __init__(self):
        self.config = get_config()
        self.embeddings = None
        self.vectordb = None
        self.chains = {}
        self.chain_stats = {}

        # Initialize embeddings once
        self._init_embeddings()
        self._init_vectordb()

    def _init_embeddings(self):
        """Initialize embeddings model"""
        try:
            embedding_model = self.config.get("database.embedding_model", "all-MiniLM-L6-v2")
            logger.info(f"Loading embeddings model: {embedding_model}")

            self.embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )

            logger.info("✅ Embeddings model loaded successfully")

        except Exception as e:
            logger.error(f"❌ Failed to load embeddings: {e}")
            raise

    def _init_vectordb(self):
        """Initialize vector database"""
        try:
            db_config = self.config.get_database_config()
            db_path = db_config.get("chroma_db_path", "./db/chroma_db")
            collection_name = db_config.get("collection_name", "healthcare_chunks")

            # Ensure database directory exists
            Path(db_path).mkdir(parents=True, exist_ok=True)

            logger.info(f"Connecting to ChromaDB at: {db_path}")

            # Try direct ChromaDB client first to check collection
            import chromadb
            direct_client = chromadb.PersistentClient(path=db_path)
            collections = direct_client.list_collections()

            logger.info(f"Available collections: {[col.name for col in collections]}")

            # Check if target collection exists
            target_collection = None
            for col in collections:
                if col.name == collection_name:
                    target_collection = col
                    break

            if target_collection:
                doc_count = target_collection.count()
                logger.info(f"Found collection '{collection_name}' with {doc_count} documents")

                # Use LangChain wrapper for existing collection
                self.vectordb = Chroma(
                    persist_directory=db_path,
                    embedding_function=self.embeddings,
                    collection_name=collection_name
                )

                # Verify connection
                test_count = self.vectordb._collection.count()
                logger.info(f"✅ Connected to ChromaDB. Documents: {test_count}")

                if test_count == 0:
                    logger.warning("⚠️ Vector database appears empty through LangChain wrapper")
                    # Try to access direct collection
                    logger.info("Attempting direct collection access...")

            else:
                logger.warning(
                    f"Collection '{collection_name}' not found. Available: {[col.name for col in collections]}")

                # Create new collection via LangChain
                self.vectordb = Chroma(
                    persist_directory=db_path,
                    embedding_function=self.embeddings,
                    collection_name=collection_name
                )
                logger.info(f"✅ Created new ChromaDB collection: {collection_name}")

        except Exception as e:
            logger.error(f"❌ Failed to initialize vector database: {e}")
            raise

    def _create_custom_prompt(self, model_name: str) -> PromptTemplate:
        """Create custom prompt template for each model"""

        if "deepseek-r1" in model_name.lower():
            # Special prompt for DeepSeek R1 with thinking process
            template = """<think>
The user is asking: {question}

Let me analyze the provided context to give an accurate and helpful answer.

Context analysis:
{context}

I should provide a comprehensive answer based on this medical/healthcare information while being careful not to give specific medical advice that requires professional consultation.
</think>

Based on the provided medical information, here's what I can tell you:

Context: {context}

Question: {question}

Answer: I'll provide you with evidence-based information from medical sources. However, please remember that this is for educational purposes and you should consult healthcare professionals for personalized medical advice.

"""
        else:
            # Standard prompt for other models
            template = """You are a helpful medical assistant. Use the following context to answer the question accurately and helpfully.

Context: {context}

Question: {question}

Instructions:
- Provide evidence-based information from the context
- If the context doesn't contain relevant information, say so clearly
- Always recommend consulting healthcare professionals for personalized advice
- Be clear, concise, and helpful

Answer:"""

        return PromptTemplate(
            input_variables=["context", "question"],
            template=template
        )

    def _get_retriever(self, model_name: str):
        """Get retriever with model-specific configuration"""
        rag_config = self.config.get_rag_config()
        top_k = rag_config.get("retrieval_top_k", 5)
        similarity_threshold = rag_config.get("similarity_threshold", 0.3)  # Giảm từ 0.7 xuống 0.3

        return self.vectordb.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": top_k,
                "score_threshold": similarity_threshold
            }
        )

    def build_rag_chain(self, model_name: str):
        """Build RAG chain for specific model"""
        try:
            start_time = time.time()

            # Ensure model is running
            if not is_model_running(model_name):
                logger.info(f"Starting Ollama model: {model_name}")
                start_model(model_name)

            # Get LLM
            llm = get_ollama_model(model_name)
            model_config = self.config.get_model_config(model_name)

            # Configure LLM parameters
            if hasattr(llm, 'temperature'):
                llm.temperature = model_config.get("temperature", 0.7)
            if hasattr(llm, 'max_tokens'):
                llm.max_tokens = model_config.get("max_tokens", 2048)

            # Get retriever
            retriever = self._get_retriever(model_name)

            # Create custom prompt
            prompt = self._create_custom_prompt(model_name)

            # Build RAG chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={
                    "prompt": prompt,
                    "verbose": self.config.get("server.debug", False)
                },
                return_source_documents=True
            )

            # Cache the chain
            self.chains[model_name] = qa_chain

            # Track stats
            load_time = time.time() - start_time
            self.chain_stats[model_name] = {
                "load_time": load_time,
                "created_at": time.time(),
                "query_count": 0,
                "total_query_time": 0,
                "avg_query_time": 0
            }

            logger.info(f"✅ RAG chain for {model_name} built in {load_time:.2f}s")
            return qa_chain

        except Exception as e:
            logger.error(f"❌ Failed to build RAG chain for {model_name}: {e}")
            raise

    def get_or_create_chain(self, model_name: str):
        """Get existing chain or create new one"""
        if model_name not in self.chains:
            return self.build_rag_chain(model_name)
        return self.chains[model_name]

    def query(self, model_name: str, question: str,
              include_sources: bool = None) -> Dict[str, Any]:
        """Query RAG system with enhanced response"""

        if include_sources is None:
            include_sources = self.config.get("ui.show_source_documents", True)

        try:
            start_time = time.time()

            # Get or create chain
            chain = self.get_or_create_chain(model_name)

            # Execute query
            result = chain({"query": question})

            # Process response
            response = result.get("result", "")
            source_docs = result.get("source_documents", [])

            query_time = time.time() - start_time

            # Update stats
            if model_name in self.chain_stats:
                stats = self.chain_stats[model_name]
                stats["query_count"] += 1
                stats["total_query_time"] += query_time
                stats["avg_query_time"] = stats["total_query_time"] / stats["query_count"]

            # Format response
            formatted_response = self._format_response(
                response, source_docs, query_time, include_sources
            )

            logger.info(f"Query processed for {model_name} in {query_time:.2f}s")

            return {
                "response": formatted_response,
                "source_documents": source_docs,
                "query_time": query_time,
                "model_name": model_name,
                "timestamp": time.time()
            }

        except Exception as e:
            logger.error(f"❌ Query failed for {model_name}: {e}")
            return {
                "response": f"❌ Error processing query: {str(e)}",
                "source_documents": [],
                "query_time": 0,
                "model_name": model_name,
                "timestamp": time.time(),
                "error": True
            }

    def _format_response(self, response: str, source_docs: List[Document],
                         query_time: float, include_sources: bool) -> str:
        """Format response with sources and timing info"""

        # Handle DeepSeek R1 special format
        if "<think>" in response and "</think>" in response:
            response = self._format_deepseek_response(response)

        # Add timing info if enabled
        if self.config.get("ui.show_timing_info", True):
            response += f"\n\n⏱️ *Response generated in {query_time:.2f}s*"

        # Add source information if enabled and available
        if include_sources and source_docs:
            sources_text = self._format_sources(source_docs)
            response += f"\n\n{sources_text}"

        return response

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

    def _format_sources(self, source_docs: List[Document]) -> str:
        """Format source documents information"""
        if not source_docs:
            return ""

        sources_text = f"📚 **Sources ({len(source_docs)} documents):**\n"

        for i, doc in enumerate(source_docs[:3], 1):  # Show max 3 sources
            content_preview = doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content

            # Get metadata if available
            metadata = doc.metadata
            source_info = metadata.get("source", f"Document {i}")

            sources_text += f"\n**{i}. {source_info}**\n{content_preview}\n"

        if len(source_docs) > 3:
            sources_text += f"\n*... and {len(source_docs) - 3} more sources*"

        return sources_text

    def get_chain_stats(self, model_name: str = None) -> Dict:
        """Get performance statistics for chains"""
        if model_name:
            return self.chain_stats.get(model_name, {})
        return self.chain_stats.copy()

    def health_check(self) -> Dict[str, Any]:
        """Perform health check on RAG system"""
        health_info = {
            "status": "healthy",
            "timestamp": time.time(),
            "issues": []
        }

        try:
            # Check embeddings
            if self.embeddings is None:
                health_info["issues"].append("Embeddings not initialized")
                health_info["status"] = "unhealthy"

            # Check vector database
            if self.vectordb is None:
                health_info["issues"].append("Vector database not initialized")
                health_info["status"] = "unhealthy"
            else:
                try:
                    doc_count = self.vectordb._collection.count()
                    health_info["document_count"] = doc_count
                    if doc_count == 0:
                        health_info["issues"].append("Vector database is empty")
                        health_info["status"] = "warning"
                except Exception as e:
                    health_info["issues"].append(f"Vector database error: {str(e)}")
                    health_info["status"] = "unhealthy"

            # Check active chains
            health_info["active_chains"] = list(self.chains.keys())
            health_info["chain_count"] = len(self.chains)

            # Check enabled models
            enabled_models = self.config.get_enabled_models()
            health_info["enabled_models"] = enabled_models

            # Check Ollama models
            running_models = []
            for model in enabled_models:
                if is_model_running(model):
                    running_models.append(model)

            health_info["running_models"] = running_models

            if len(running_models) == 0:
                health_info["issues"].append("No Ollama models running")
                health_info["status"] = "warning"

        except Exception as e:
            health_info["issues"].append(f"Health check error: {str(e)}")
            health_info["status"] = "unhealthy"

        return health_info

    def cleanup_unused_chains(self, max_idle_time: int = 3600):
        """Clean up unused chains to free memory"""
        current_time = time.time()
        removed_chains = []

        for model_name, stats in list(self.chain_stats.items()):
            # Calculate idle time
            last_activity = max(stats.get("created_at", 0),
                                stats.get("last_query_time", 0))
            idle_time = current_time - last_activity

            if idle_time > max_idle_time:
                # Remove chain
                if model_name in self.chains:
                    del self.chains[model_name]
                del self.chain_stats[model_name]
                removed_chains.append(model_name)
                logger.info(f"🧹 Removed unused chain: {model_name}")

        return removed_chains

    def reload_vectordb(self):
        """Reload vector database (useful after adding new documents)"""
        try:
            logger.info("🔄 Reloading vector database...")
            self._init_vectordb()

            # Clear existing chains to force reload
            self.chains.clear()
            self.chain_stats.clear()

            logger.info("✅ Vector database reloaded successfully")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to reload vector database: {e}")
            return False

    def get_database_info(self) -> Dict[str, Any]:
        """Get information about the vector database"""
        try:
            if not self.vectordb:
                return {"error": "Vector database not initialized"}

            collection = self.vectordb._collection

            return {
                "document_count": collection.count(),
                "collection_name": collection.name,
                "embedding_model": self.config.get("database.embedding_model"),
                "database_path": self.config.get("database.chroma_db_path"),
                "status": "active"
            }

        except Exception as e:
            return {"error": f"Failed to get database info: {str(e)}"}

    def search_similar_documents(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for similar documents in the vector database"""
        try:
            if not self.vectordb:
                return []

            # Perform similarity search
            docs = self.vectordb.similarity_search_with_score(query, k=top_k)

            results = []
            for doc, score in docs:
                results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "similarity_score": score
                })

            return results

        except Exception as e:
            logger.error(f"❌ Similar documents search failed: {e}")
            return []

    def shutdown(self):
        """Graceful shutdown of RAG pipeline"""
        logger.info("🔄 Shutting down RAG pipeline...")

        # Clear chains
        self.chains.clear()
        self.chain_stats.clear()

        # Close vector database connection
        if self.vectordb:
            try:
                # ChromaDB doesn't have explicit close method
                self.vectordb = None
            except Exception as e:
                logger.error(f"Error closing vector database: {e}")

        logger.info("✅ RAG pipeline shutdown complete")


# Factory function for backward compatibility
def load_rag_chain(model_name: str):
    """Factory function to create RAG chain (backward compatibility)"""
    pipeline = EnhancedRAGPipeline()
    return pipeline.get_or_create_chain(model_name)


# Global pipeline instance
_pipeline_instance = None


def get_rag_pipeline() -> EnhancedRAGPipeline:
    """Get global RAG pipeline instance"""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = EnhancedRAGPipeline()
    return _pipeline_instance


def shutdown_rag_pipeline():
    """Shutdown global RAG pipeline"""
    global _pipeline_instance
    if _pipeline_instance:
        _pipeline_instance.shutdown()
        _pipeline_instance = None