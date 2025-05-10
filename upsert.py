import os
import json
from tqdm import tqdm
from pinecone import Pinecone
from openai import OpenAI

from keys import PINECONE_API_KEY, OPENAI_API_KEY
from config import INDEX_NAME, INDEX_NAMESPACE, OPENAI_EMBEDDING_MODEL

client = OpenAI(api_key=OPENAI_API_KEY)
DATA_DIR = "data/rag"

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

def load_chunks_from_dir(base_dir):
    all_chunks = []
    for source_type in ["fas", "ss"]:
        dir_path = os.path.join(base_dir, source_type)
        for filename in os.listdir(dir_path):
            if not filename.endswith(".json"):
                continue
            file_path = os.path.join(dir_path, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)
                for i, chunk in enumerate(chunks):
                    chunk_id = f"{source_type} {filename.split('_')[1][:-5]} {i}"
                    all_chunks.append({
                        "id": chunk_id,
                        "text": chunk["text"],
                        "metadata": {
                            "source_type": source_type,
                            "source_file": filename[:-5],
                            "type": chunk["type"],
                            "keywords": chunk["keywords"],
                        }
                    })
    return all_chunks

def get_embeddings(texts):
    response = client.embeddings.create(model=OPENAI_EMBEDDING_MODEL,
    input=texts)
    return [d.embedding for d in response.data]

def upsert_chunks(chunks, batch_size=96):
    for i in tqdm(range(0, len(chunks), batch_size)):
        print(f"Upserting chunks {i+1} to {i+batch_size}...")
        batch = chunks[i:i+batch_size]
        texts = [c["text"] for c in batch]
        embeddings = get_embeddings(texts)
        vectors = [
            {
                "id": batch[j]["id"],
                "values": embeddings[j],
                "metadata": batch[j]["metadata"]
            }
            for j in range(len(batch))
        ]
        index.upsert(vectors=vectors, namespace=INDEX_NAMESPACE)

def main():
    print("Loading chunks...")
    chunks = load_chunks_from_dir(DATA_DIR)
    print(f"Loaded {len(chunks)} chunks")

    print(f"Upserting to Pinecone index: {INDEX_NAME}")
    upsert_chunks(chunks)
    print("Done.")

if __name__ == "__main__":
    main()
    
    # texts = ["This is a test sentence.", "Another test sentence."]
    
    # embeddings = get_embeddings(texts)
    
    # for i, embedding in enumerate(embeddings):
    #     print(f"Embedding for text {i+1}: {len(embedding)} dimensions")
