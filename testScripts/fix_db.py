#!/usr/bin/env python3
"""
Test ChromaDB Connection
Verify data accessibility through different methods
"""

import json
import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


def test_direct_chromadb():
    """Test direct ChromaDB client access"""
    print("🔍 Testing Direct ChromaDB Access")
    print("=" * 40)

    try:
        # Connect using direct client (same as upload script)
        client = chromadb.PersistentClient(path="../../db/chroma_db")

        # List all collections
        collections = client.list_collections()
        print(f"📋 Found {len(collections)} collections:")

        for col in collections:
            count = col.count()
            print(f"  - {col.name}: {count} documents")

            if count > 0:
                # Test query
                results = col.query(
                    query_texts=["headache pain relief"],
                    n_results=2
                )
                print(f"    Sample query returned {len(results['documents'][0])} results")
                if results['documents'][0]:
                    print(f"    First result: {results['documents'][0][0][:100]}...")

        return collections

    except Exception as e:
        print(f"❌ Direct ChromaDB test failed: {e}")
        return []


def test_langchain_chromadb():
    """Test LangChain ChromaDB wrapper access"""
    print("\n🔗 Testing LangChain ChromaDB Access")
    print("=" * 40)

    try:
        # Connect using LangChain wrapper (same as RAG pipeline)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # Test different paths
        paths_to_test = [
            "../db/chroma_db",
            "./db/chroma_db",
            "db/chroma_db"
        ]

        for db_path in paths_to_test:
            print(f"\n📍 Testing path: {db_path}")

            try:
                vectordb = Chroma(
                    persist_directory=db_path,
                    embedding_function=embeddings,
                    collection_name="healthcare_chunks"
                )

                # Get document count
                count = vectordb._collection.count()
                print(f"  ✅ Connected! Documents: {count}")

                if count > 0:
                    # Test similarity search
                    results = vectordb.similarity_search("headache", k=2)
                    print(f"  🔍 Search test: {len(results)} results")
                    if results:
                        print(f"     First result: {results[0].page_content[:100]}...")
                        print(f"     Metadata: {results[0].metadata}")

                    return vectordb
                else:
                    print(f"  ⚠️ Database empty at {db_path}")

            except Exception as e:
                print(f"  ❌ Failed to connect to {db_path}: {e}")

    except Exception as e:
        print(f"❌ LangChain ChromaDB test failed: {e}")

    return None


def test_embedding_compatibility():
    """Test if embeddings are compatible"""
    print("\n🧪 Testing Embedding Compatibility")
    print("=" * 40)

    try:
        # Load sample data to check embedding format
        with open("../data/chunked_with_embedding.jsonl", "r", encoding="utf-8") as f:
            first_line = f.readline()
            sample_data = json.loads(first_line)

            print(f"📊 Sample embedding dimensions: {len(sample_data['embedding'])}")
            print(f"📝 Sample document: {sample_data['response'][:100]}...")
            print(f"🏷️ Sample metadata: {sample_data['prompt'][:100]}...")

        # Test current embedding model
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        test_embedding = embeddings.embed_query("test query")
        print(f"🔧 Current model embedding dimensions: {len(test_embedding)}")

        if len(sample_data['embedding']) == len(test_embedding):
            print("✅ Embedding dimensions match!")
        else:
            print("❌ Embedding dimensions mismatch!")
            print("💡 This could cause search issues")

    except Exception as e:
        print(f"❌ Embedding compatibility test failed: {e}")


def suggest_fixes():
    """Suggest fixes based on test results"""
    print("\n💡 Suggested Fixes")
    print("=" * 20)

    print("1. If direct ChromaDB works but LangChain doesn't:")
    print("   - Update config path to match working path")
    print("   - Check collection name spelling")

    print("\n2. If embeddings mismatch:")
    print("   - Update embedding model in config")
    print("   - Or re-embed with current model")

    print("\n3. If no data found:")
    print("   - Re-run upload_to_chroma.py")
    print("   - Check data file path")


def main():
    """Run all tests"""
    print("🤖 ChromaDB Connection Test Suite")
    print("=" * 50)

    # Test 1: Direct ChromaDB
    collections = test_direct_chromadb()

    # Test 2: LangChain wrapper
    langchain_db = test_langchain_chromadb()

    # Test 3: Embedding compatibility
    test_embedding_compatibility()

    # Suggestions
    suggest_fixes()

    print("\n🎯 Summary:")
    print(f"- Direct ChromaDB collections: {len(collections)}")
    print(f"- LangChain connection: {'✅ Success' if langchain_db else '❌ Failed'}")


if __name__ == "__main__":
    main()