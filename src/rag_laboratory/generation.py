from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from rag_laboratory.utils.config import cfg
# import logging
from rag_laboratory.utils.logger import get_configured_logger


logger = get_configured_logger(__name__)

LLM = Ollama(
    model=cfg["llm"]["model"],
    base_url=cfg["llm"]["base_url"],
    temperature=cfg["llm"]["temperature"],
    request_timeout=cfg["llm"]["request_timeout"],
)

EMBED_MODEL = OllamaEmbedding(
    model_name=cfg["embedding"]["model"],
    base_url=cfg["embedding"]["base_url"],
)

if __name__ == "__main__":
    # logger.info("LLM: %s | Embedding: %s", cfg["llm"]["model"], cfg["embedding"]["model"])
    logger.info(f"LLM: {cfg['llm']['model']} | Embedding: {cfg["embedding"]["model"]}")
