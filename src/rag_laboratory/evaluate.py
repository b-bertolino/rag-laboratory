import json
import logging
from pathlib import Path

logger = logging.getLogger("local_rag")


def load_gold_qa(path: Path) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def first_hit_rank(texts: list[str], keywords: list[str]) -> int | None:
    """Rank (da 1) del primo testo che contiene tutte le keyword, None se nessuno."""
    keywords = [kw.lower() for kw in keywords]
    for rank, text in enumerate(texts, 1):
        text_lower = text.lower()
        if all(kw in text_lower for kw in keywords):
            return rank
    return None


def evaluate_retrieval(retriever, gold_qa: list[dict]) -> dict:
    results = []
    for item in gold_qa:
        retrieved = retriever.retrieve(item["question"])
        rank = first_hit_rank([n.text for n in retrieved], item["expected_keywords"])
        results.append({"id": item["id"], "question": item["question"], "hit": rank is not None, "rank": rank})
        logger.info("[%s] hit=%s rank=%s | %s", item["id"], rank is not None, rank, item["question"][:50])

    hits = sum(r["hit"] for r in results)
    reciprocal_ranks = [1 / r["rank"] if r["hit"] else 0.0 for r in results]
    return {
        "hits": hits,
        "total": len(results),
        "hit_rate": hits / len(results),
        "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks),
        "results": results,
    }
