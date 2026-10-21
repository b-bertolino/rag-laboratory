from rag_laboratory.data.sample import CORPUS_TEXT
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

from rag_laboratory.utils.config import cfg
from rag_laboratory.utils.logger import get_configured_logger


logger = get_configured_logger(__name__)

documents = [
    Document(text=CORPUS_TEXT, metadata={"source": "ai_safety_primer"})
]
logger.info(f"Loaded {len(documents)} document(s) from inline corpus")

splitter = SentenceSplitter(
    chunk_size=cfg["splitter"]["chunk_size"],
    chunk_overlap=cfg["splitter"]["chunk_overlap"],
)
nodes = splitter.get_nodes_from_documents(documents)
logger.info(
    "Split into %d nodes (chunk_size=%d, overlap=%d)",
    len(nodes),
    cfg["splitter"]["chunk_size"],
    cfg["splitter"]["chunk_overlap"],
)

print(f"\nFirst chunk preview ({len(nodes[0].text)} chars):")
print("-" * 60)
print(nodes[0].text[:400], "...")