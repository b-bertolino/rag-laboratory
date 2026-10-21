from pathlib import Path
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

from rag_laboratory.utils.config import cfg
from rag_laboratory.utils.logger import get_configured_logger
from rag_laboratory.generation import EMBED_MODEL, LLM
from rag_laboratory.chunking import nodes

logger = get_configured_logger(__name__)


PERSIST_DIR = Path(cfg["chroma"]["persist_dir"])
COLLECTION = cfg["chroma"]["collection_name"]

chroma_client = chromadb.PersistentClient(path=str(PERSIST_DIR))
existing = [c.name for c in chroma_client.list_collections()]

if COLLECTION in existing:
    logger.info(
        "Cache hit — loading existing collection '%s' from %s",
        COLLECTION,
        PERSIST_DIR,
    )
    chroma_collection = chroma_client.get_collection(COLLECTION)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=EMBED_MODEL,
    )
else:
    logger.info(
        "Cache miss — embedding %d nodes into new collection '%s'",
        len(nodes),
        COLLECTION,
    )
    chroma_collection = chroma_client.create_collection(COLLECTION)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embed_model=EMBED_MODEL,
    )
    logger.info("Index built and persisted to %s", PERSIST_DIR)

print(f"Collection '{COLLECTION}' has {chroma_collection.count()} vectors.")

query_engine = index.as_query_engine(
    llm=LLM,
    similarity_top_k=cfg["retrieval"]["similarity_top_k"],
)

QUERY = "What is Constitutional AI and who developed it?"
logger.info("Running query: %s", QUERY)

response = query_engine.query(QUERY)

# display(Markdown(f"**Query:** {QUERY}\n\n**Answer:** {response}"))
print(f"**Query:** {QUERY}\n\n**Answer:** {response}")


print("\n--- Retrieved source nodes ---")
for i, node in enumerate(response.source_nodes, 1):
    score = getattr(node, "score", "n/a")
    print(f"[{i}] score={score:.4f}  |  {node.text[:120].strip()}...")


from llama_index.core.retrievers import VectorIndexRetriever

gold_qa = [
    {
        "id": "q1",
        "question": "What is alignment in the context of AI safety?",
        "expected_keywords": ["alignment", "goals"],
    },
    {
        "id": "q2",
        "question": "What is RLHF and how does it work?",
        "expected_keywords": ["rlhf", "reward"],
    },
    {
        "id": "q3",
        "question": "What is reward hacking?",
        "expected_keywords": ["reward hacking", "unintended"],
    },
    {
        "id": "q4",
        "question": "What is Constitutional AI and who developed it?",
        "expected_keywords": ["constitutional ai", "anthropic"],
    },
    {
        "id": "q5",
        "question": "What is red teaming in AI?",
        "expected_keywords": ["red teaming", "failure"],
    },
    {
        "id": "q6",
        "question": "What is deceptive alignment?",
        "expected_keywords": ["deceptive alignment", "training"],
    },
    {
        "id": "q7",
        "question": "What is interpretability in AI systems?",
        "expected_keywords": ["interpretability", "neural"],
    },
    {
        "id": "q8",
        "question": "Which organizations are working on AI safety?",
        "expected_keywords": ["anthropic", "openai", "deepmind"],
    },
]

retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=cfg["retrieval"]["similarity_top_k"],
    embed_model=EMBED_MODEL,
)

hits = 0
reciprocal_ranks = []
results = []

for item in gold_qa:
    retrieved_nodes = retriever.retrieve(item["question"])
    keywords = [kw.lower() for kw in item["expected_keywords"]]

    first_hit_rank = None
    for rank, node in enumerate(retrieved_nodes, 1):
        text_lower = node.text.lower()
        if all(kw in text_lower for kw in keywords):
            first_hit_rank = rank
            break

    hit = first_hit_rank is not None
    hits += int(hit)
    reciprocal_ranks.append(1 / first_hit_rank if hit else 0.0)

    results.append(
        {
            "id": item["id"],
            "hit": hit,
            "rank": first_hit_rank,
            "question": item["question"][:60],
        }
    )
    logger.info(
        "[%s] hit=%s rank=%s | %s",
        item["id"],
        hit,
        first_hit_rank,
        item["question"][:50],
    )

hit_rate = hits / len(gold_qa)
mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)

print("\n" + "=" * 50)
print(
    f"  Retrieval Evaluation Results (top-k={cfg['retrieval']['similarity_top_k']})"
)
print("=" * 50)
print(f"  Hit-Rate : {hit_rate:.2%}  ({hits}/{len(gold_qa)} questions)")
print(f"  MRR      : {mrr:.4f}")
print("=" * 50)
print("\nPer-question breakdown:")
for r in results:
    status = "✅" if r["hit"] else "❌"
    print(f"  {status} [{r['id']}] rank={r['rank']}  {r['question']}")