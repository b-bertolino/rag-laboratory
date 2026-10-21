import json


cfg = {
    "llm": {
        "model": "llama3.2:3b",
        "base_url": "http://localhost:11434",
        "temperature": 0.0,
        "request_timeout": 120.0,
    },
    "embedding": {
        "model": "nomic-embed-text",
        "base_url": "http://localhost:11434",
    },
    "splitter": {
        "chunk_size": 512,
        "chunk_overlap": 50,
    },
    "chroma": {
        "persist_dir": "./chroma_db",
        "collection_name": "ai_safety_rag",
    },
    "retrieval": {
        "similarity_top_k": 3,
    },
}

# print(json.dumps(cfg, indent=2))