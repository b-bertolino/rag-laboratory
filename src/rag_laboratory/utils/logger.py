import logging
import shutil

from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore

def get_configured_logger(name: str = __name__) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        # format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        format="%(asctime)s [%(levelname)s] %(filename)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(name)

if __name__ == "__main__":
    # logger = logging.getLogger("local_rag")
    logger = get_configured_logger("local_rag")
    logger.info("Imports loaded successfully.")