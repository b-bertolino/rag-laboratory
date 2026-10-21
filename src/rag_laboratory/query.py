import logging

from llama_index.core import VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.llms import LLM
from llama_index.core.retrievers import VectorIndexRetriever

logger = logging.getLogger("local_rag")


def build_query_engine(index: VectorStoreIndex, llm: LLM, cfg: dict):
    """Retrieval top-k + prompt + LLM in un solo oggetto."""
    return index.as_query_engine(llm=llm, similarity_top_k=cfg["retrieval"]["similarity_top_k"])


def build_retriever(index: VectorStoreIndex, embed_model: BaseEmbedding, cfg: dict) -> VectorIndexRetriever:
    """Solo il retrieval, senza LLM: serve per la valutazione e il guardrail."""
    return VectorIndexRetriever(
        index=index,
        similarity_top_k=cfg["retrieval"]["similarity_top_k"],
        embed_model=embed_model,
    )


def safe_query(query_engine, retriever, question: str, threshold: float) -> str:
    """Risponde solo se il chunk migliore supera la soglia, altrimenti rifiuta."""
    nodes = retriever.retrieve(question)
    if not nodes:
        return "[GUARDRAIL] No documents retrieved."

    best_score = max(n.score for n in nodes if n.score is not None)
    logger.info("[safe_query] best_score=%.4f threshold=%.2f", best_score, threshold)

    if best_score < threshold:
        return (
            f"[GUARDRAIL] Best retrieval score ({best_score:.4f}) is below "
            f"threshold ({threshold}). Cannot answer reliably."
        )
    return str(query_engine.query(question))
