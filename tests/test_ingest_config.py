from rag_laboratory.config import ROOT, load_config
from rag_laboratory.ingest import load_documents, split_documents


def test_config_paths_are_absolute_and_relative_to_project():
    cfg = load_config()
    assert cfg["chroma"]["persist_dir"] == ROOT / "chroma_db"
    assert cfg["data"]["input_dir"] == ROOT / "data"
    assert cfg["eval"]["gold_qa_path"].exists()


def test_corpus_splits_into_two_nodes():
    cfg = load_config()
    documents = load_documents(cfg["data"]["input_dir"])

    assert len(documents) == 1
    assert documents[0].metadata == {"source": "ai_safety_primer"}
    # Il notebook riporta 3 nodi, ma i suoi output vengono da un file con a capo
    # diversi (e versione di LlamaIndex ignota). Con CORPUS_TEXT identico al
    # codice pubblicato e llama-index-core 0.14.25 i nodi sono 2 (NOTES.md, 1b).
    assert len(split_documents(documents, cfg)) == 2
