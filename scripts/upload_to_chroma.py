import json
import chromadb
from tqdm import tqdm
import os
import shutil
import time
from pathlib import Path


def cleanup_corrupted_database(db_path):
    """Clean up corrupted ChromaDB"""
    print(f"🧹 Cleaning up corrupted database at: {db_path}")

    db_path = Path(db_path)

    if db_path.exists():
        # Create backup
        backup_path = db_path.parent / f"chroma_db_backup_{int(time.time())}"
        print(f"📦 Creating backup at: {backup_path}")

        try:
            shutil.move(str(db_path), str(backup_path))
            print(f"✅ Backup created successfully")
        except Exception as e:
            print(f"⚠️ Backup failed, proceeding with deletion: {e}")
            try:
                shutil.rmtree(db_path)
            except Exception as e2:
                print(f"❌ Failed to delete corrupted database: {e2}")
                return False

    # Create fresh directory
    db_path.mkdir(parents=True, exist_ok=True)
    print(f"📁 Created fresh database directory")
    return True


def find_data_file():
    """Tìm file data trong project"""
    possible_paths = [
        "../data/chunked_with_embedding.jsonl",
        "./data/chunked_with_embedding.jsonl",
        "data/chunked_with_embedding.jsonl",
        "../chunked_with_embedding.jsonl",
        "./chunked_with_embedding.jsonl"
    ]

    for path in possible_paths:
        if Path(path).exists():
            file_size = Path(path).stat().st_size
            print(f"✅ Found data file: {path} ({file_size:,} bytes)")
            return path

    print("❌ Data file not found. Searching...")
    # Search in current directory and subdirectories
    for root, dirs, files in os.walk(".."):
        for file in files:
            if file == "chunked_with_embedding.jsonl":
                found_path = os.path.join(root, file)
                file_size = Path(found_path).stat().st_size
                print(f"✅ Found data file: {found_path} ({file_size:,} bytes)")
                return found_path

    return None


def test_data_file(data_file):
    """Test if data file is valid"""
    print(f"🧪 Testing data file: {data_file}")

    try:
        with open(data_file, "r", encoding="utf-8") as f:
            # Read first few lines
            lines_read = 0
            valid_lines = 0

            for line in f:
                lines_read += 1
                if lines_read > 10:  # Test first 10 lines
                    break

                try:
                    data = json.loads(line)
                    # Check required fields
                    if all(key in data for key in ["response", "embedding", "prompt"]):
                        valid_lines += 1
                        if valid_lines == 1:
                            print(f"   📊 Sample embedding dim: {len(data['embedding'])}")
                            print(f"   📝 Sample text: {data['response'][:100]}...")
                    else:
                        print(f"   ⚠️ Missing fields in line {lines_read}")
                except json.JSONDecodeError:
                    print(f"   ❌ Invalid JSON in line {lines_read}")

            print(f"   ✅ Valid lines: {valid_lines}/{lines_read}")
            return valid_lines > 0

    except Exception as e:
        print(f"   ❌ Error reading file: {e}")
        return False


def upload_data_to_chroma():
    """Upload data to ChromaDB with cleanup"""

    print("🔍 Searching for data file...")
    data_file = find_data_file()

    if not data_file:
        print("❌ Data file 'chunked_with_embedding.jsonl' not found!")
        return False

    # Test data file
    if not test_data_file(data_file):
        print("❌ Data file validation failed!")
        return False

    # Setup database path
    BATCH_SIZE = 1000  # Smaller batch size for stability
    db_path = "../db/chroma_db"

    print(f"📁 Target database path: {Path(db_path).absolute()}")

    # Clean up any existing corrupted database
    if not cleanup_corrupted_database(db_path):
        return False

    # Initialize ChromaDB with retry
    max_retries = 3
    chroma_client = None

    for attempt in range(max_retries):
        try:
            print(f"🔄 Attempting to connect to ChromaDB (attempt {attempt + 1}/{max_retries})")
            chroma_client = chromadb.PersistentClient(path=db_path)
            print("✅ ChromaDB client created successfully")
            break

        except Exception as e:
            print(f"❌ Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print("🔧 Cleaning up and retrying...")
                cleanup_corrupted_database(db_path)
                time.sleep(2)
            else:
                print("❌ All connection attempts failed")
                return False

    if not chroma_client:
        return False

    # Create collection
    try:
        collection_name = "healthcare_chunks"
        print(f"📋 Creating collection: {collection_name}")

        # Delete if exists
        try:
            existing_collection = chroma_client.get_collection(collection_name)
            print(f"🗑️ Deleting existing collection...")
            chroma_client.delete_collection(collection_name)
        except:
            pass

        collection = chroma_client.create_collection(collection_name)
        print(f"✅ Collection created successfully")

    except Exception as e:
        print(f"❌ Failed to create collection: {e}")
        return False

    # Load and upload data
    print(f"📖 Loading data from: {data_file}")

    try:
        all_documents = []
        all_embeddings = []
        all_metadatas = []
        all_ids = []

        with open(data_file, "r", encoding="utf-8") as f:
            for idx, line in enumerate(tqdm(f, desc="Loading data")):
                try:
                    data = json.loads(line)

                    # Validate required fields
                    if not all(key in data for key in ["response", "embedding", "prompt"]):
                        print(f"⚠️ Skipping line {idx}: missing required fields")
                        continue

                    all_documents.append(str(data["response"]))
                    all_embeddings.append(data["embedding"])
                    all_metadatas.append({"prompt": str(data["prompt"])})
                    all_ids.append(f"chunk_{idx}")

                except json.JSONDecodeError as e:
                    print(f"⚠️ Skipping invalid JSON at line {idx}: {e}")
                except Exception as e:
                    print(f"⚠️ Error processing line {idx}: {e}")

        if not all_documents:
            print("❌ No valid documents found in data file!")
            return False

        print(f"✅ Loaded {len(all_documents)} valid documents")

        # Upload in smaller batches
        print("🔄 Uploading to ChromaDB...")
        uploaded_count = 0

        for i in tqdm(range(0, len(all_documents), BATCH_SIZE), desc="Uploading batches"):
            end = min(i + BATCH_SIZE, len(all_documents))

            try:
                batch_docs = all_documents[i:end]
                batch_embeddings = all_embeddings[i:end]
                batch_metadata = all_metadatas[i:end]
                batch_ids = all_ids[i:end]

                collection.add(
                    documents=batch_docs,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadata,
                    ids=batch_ids
                )

                uploaded_count += len(batch_docs)

            except Exception as e:
                print(f"❌ Error uploading batch {i}-{end}: {e}")
                print("🔄 Continuing with next batch...")
                continue

        # Verify upload
        try:
            final_count = collection.count()
            print(f"✅ Upload complete!")
            print(f"   📊 Expected: {len(all_documents)} documents")
            print(f"   📊 Uploaded: {uploaded_count} documents")
            print(f"   📊 In database: {final_count} documents")

            if final_count > 0:
                # Test search
                test_results = collection.query(
                    query_texts=["headache pain relief treatment"],
                    n_results=2
                )

                if test_results['documents'][0]:
                    print("🔍 Search test successful!")
                    print(f"   📝 Sample result: {test_results['documents'][0][0][:100]}...")
                    return True
                else:
                    print("⚠️ Search test returned no results")
                    return final_count > 0
            else:
                print("❌ No documents in database after upload")
                return False

        except Exception as e:
            print(f"❌ Error verifying upload: {e}")
            return False

    except Exception as e:
        print(f"❌ Upload process failed: {e}")
        return False


def create_sample_data():
    """Create sample data if original data missing"""
    print("🧪 Creating sample medical data for testing...")

    try:
        from langchain_huggingface import HuggingFaceEmbeddings

        print("📚 Loading embedding model...")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        sample_docs = [
            {
                "prompt": "What should I do for a headache?",
                "response": "For headaches, try resting in a quiet, dark room. Apply a cold compress to your forehead or temples. Stay hydrated by drinking plenty of water. Over-the-counter pain relievers like ibuprofen or acetaminophen can help. Gentle neck and shoulder stretches may also provide relief. If headaches are frequent or severe, consult a healthcare professional."
            },
            {
                "prompt": "How can I improve my sleep quality?",
                "response": "To improve sleep quality: maintain a consistent sleep schedule by going to bed and waking up at the same time daily. Create a comfortable sleep environment that's cool, dark, and quiet. Avoid caffeine and large meals close to bedtime. Limit screen time before bed as blue light can interfere with sleep. Exercise regularly but not close to bedtime. Consider relaxation techniques like meditation or deep breathing."
            },
            {
                "prompt": "What are the benefits of regular exercise?",
                "response": "Regular exercise provides numerous health benefits including: improved cardiovascular health and reduced risk of heart disease, stronger bones and muscles, better mental health and reduced anxiety/depression, weight management and improved metabolism, enhanced immune system function, better sleep quality, increased energy levels, and reduced risk of chronic diseases like diabetes and certain cancers."
            },
            {
                "prompt": "How to manage diabetes effectively?",
                "response": "Effective diabetes management involves: monitoring blood sugar levels regularly with a glucose meter, following a healthy diet with controlled carbohydrates and regular meal times, taking medications as prescribed by your doctor, exercising regularly to help control blood sugar, maintaining a healthy weight, staying well hydrated, managing stress levels, and working closely with your healthcare team for regular check-ups."
            },
            {
                "prompt": "What are symptoms of high blood pressure?",
                "response": "High blood pressure often has no noticeable symptoms, earning it the nickname 'silent killer.' However, some people may experience: headaches, dizziness, shortness of breath, nosebleeds, or chest pain. Severe hypertension may cause blurred vision, fatigue, or confusion. It's important to have regular blood pressure checks since symptoms typically don't appear until levels are dangerously high."
            }
        ]

        # Create data directory
        data_dir = Path("../data")
        data_dir.mkdir(exist_ok=True)

        # Create sample file
        sample_file = data_dir / "chunked_with_embedding.jsonl"

        print(f"📝 Creating sample data file: {sample_file}")

        with open(sample_file, "w", encoding="utf-8") as f:
            for i, doc in enumerate(tqdm(sample_docs, desc="Creating embeddings")):
                embedding = embeddings.embed_query(doc["response"])
                json_line = {
                    "prompt": doc["prompt"],
                    "response": doc["response"],
                    "embedding": embedding
                }
                f.write(json.dumps(json_line) + "\n")

        file_size = sample_file.stat().st_size
        print(f"✅ Created sample data: {sample_file} ({file_size:,} bytes)")
        return str(sample_file)

    except Exception as e:
        print(f"❌ Failed to create sample data: {e}")
        return None


def main():
    """Main function"""
    print("🤖 ChromaDB Data Upload Tool (with Cleanup)")
    print("=" * 50)

    success = upload_data_to_chroma()

    if not success:
        print("\n❓ Data upload failed. Would you like to create sample data for testing?")
        choice = input("Create sample medical data? (y/n): ").lower().strip()

        if choice == 'y':
            sample_file = create_sample_data()
            if sample_file:
                print(f"\n🔄 Retrying upload with sample data...")
                success = upload_data_to_chroma()

    if success:
        print("\n🎉 Data upload successful!")
        print("💡 You can now restart the RAG application")
        print("🚀 Run: python main_optimized.py")
    else:
        print("\n❌ Data upload failed")
        print("💡 Please check the error messages above and try again")


if __name__ == "__main__":
    main()