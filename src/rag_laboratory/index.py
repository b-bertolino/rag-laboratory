import logging
import shutil

import chromadb
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.schema import BaseNode
from llama_index.vector_stores.chroma import ChromaVectorStore

logger = logging.getLogger("local_rag")


def get_or_build_index(
    nodes: list[BaseNode], embed_model: BaseEmbedding, cfg: dict
) -> tuple[VectorStoreIndex, chromadb.Collection]:
    """
    Al momento la cache controlla solo che la collezione esista. 
    Se si cambia documenti o chunk_size, l'indice vecchio viene riusato senza avvisi: serve fare reset_index().
    """
    persist_dir = cfg["chroma"]["persist_dir"]
    name = cfg["chroma"]["collection_name"]

    client = chromadb.PersistentClient(path=str(persist_dir))
    existing = [c.name for c in client.list_collections()]

    if name in existing:
        logger.info("Cache hit - loading existing collection '%s' from %s", name, persist_dir)
        collection = client.get_collection(name)
        vector_store = ChromaVectorStore(chroma_collection=collection)
        index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
    else:
        logger.info("Cache miss - embedding %d nodes into new collection '%s'", len(nodes), name)
        collection = client.create_collection(name)
        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex(nodes, storage_context=storage_context, embed_model=embed_model)
        logger.info("Index built and persisted to %s", persist_dir)
    return index, collection


def reset_index(cfg: dict) -> bool:
    """Cancella la cartella di Chroma. True se c'era qualcosa da cancellare."""
    persist_dir = cfg["chroma"]["persist_dir"]
    if not persist_dir.exists():
        logger.info("%s does not exist, nothing to clean up.", persist_dir)
        return False
    shutil.rmtree(persist_dir)
    logger.info("Deleted %s - run `index` to rebuild it.", persist_dir)
    return True
