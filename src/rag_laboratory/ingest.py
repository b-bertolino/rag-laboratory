from pathlib import Path

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode


def load_documents(input_dir: Path) -> list[Document]:
    return [
        Document(text=p.read_text(encoding="utf-8"), metadata={"source": p.stem})
        for p in sorted(Path(input_dir).glob("*.md"))
    ]


def split_documents(documents: list[Document], cfg: dict) -> list[BaseNode]:
    c = cfg["splitter"]
    splitter = SentenceSplitter(chunk_size=c["chunk_size"], chunk_overlap=c["chunk_overlap"])
    return splitter.get_nodes_from_documents(documents)
