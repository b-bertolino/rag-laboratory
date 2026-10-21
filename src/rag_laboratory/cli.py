import argparse
import logging
import time

from rag_laboratory.config import DEFAULT_CONFIG, load_config
from rag_laboratory.evaluate import evaluate_retrieval, load_gold_qa
from rag_laboratory.index import get_or_build_index, reset_index
from rag_laboratory.ingest import load_documents, split_documents
from rag_laboratory.models import build_embed_model, build_llm
from rag_laboratory.query import build_query_engine, build_retriever, safe_query

logger = logging.getLogger("local_rag")

DEFAULT_QUERY = "What is Constitutional AI and who developed it?"


def _build_index(cfg: dict):
    """Si spezzano i documenti anche quando l'indice è in cache (costa pochi millisecondi)."""
    embed_model = build_embed_model(cfg)
    documents = load_documents(cfg["data"]["input_dir"])
    logger.info("Loaded %d document(s) from %s", len(documents), cfg["data"]["input_dir"])
    nodes = split_documents(documents, cfg)
    logger.info(
        "Split into %d nodes (chunk_size=%d, overlap=%d)",
        len(nodes), cfg["splitter"]["chunk_size"], cfg["splitter"]["chunk_overlap"],
    )
    index, collection = get_or_build_index(nodes, embed_model, cfg)
    return index, collection, embed_model


def _print_sources(response) -> None:
    print("\n--- Retrieved source nodes ---")
    for i, node in enumerate(response.source_nodes, 1):
        preview = " ".join(node.text.split())[:120]
        print(f"[{i}] score={node.score:.4f}  |  {preview}...")


def cmd_index(cfg: dict, args) -> None:
    _, collection, _ = _build_index(cfg)
    print(f"Collection '{collection.name}' has {collection.count()} vectors.")


def cmd_ask(cfg: dict, args) -> None:
    index, _, _ = _build_index(cfg)
    query_engine = build_query_engine(index, build_llm(cfg), cfg)

    if args.show_prompt:
        for key, prompt in query_engine.get_prompts().items():
            print(f"=== {key} ===\n{prompt.get_template()}\n")

    logger.info("Running query: %s", args.question)
    start = time.perf_counter()
    response = query_engine.query(args.question)
    elapsed = time.perf_counter() - start

    print(f"\nQuery : {args.question}\nAnswer: {response}\n({elapsed:.1f} s)")
    _print_sources(response)


def cmd_eval(cfg: dict, args) -> None:
    index, _, embed_model = _build_index(cfg)
    retriever = build_retriever(index, embed_model, cfg)
    report = evaluate_retrieval(retriever, load_gold_qa(cfg["eval"]["gold_qa_path"]))

    top_k = cfg["retrieval"]["similarity_top_k"]
    print("\n" + "=" * 50)
    print(f"  Retrieval Evaluation Results (top-k={top_k})")
    print("=" * 50)
    print(f"  Hit-Rate : {report['hit_rate']:.2%}  ({report['hits']}/{report['total']} questions)")
    print(f"  MRR      : {report['mrr']:.4f}")
    print("=" * 50)
    print("\nPer-question breakdown:")
    for r in report["results"]:
        status = "OK  " if r["hit"] else "MISS"
        print(f"  {status} [{r['id']}] rank={r['rank']}  {r['question'][:60]}")


def cmd_failures(cfg: dict, args) -> None:
    index, _, embed_model = _build_index(cfg)
    query_engine = build_query_engine(index, build_llm(cfg), cfg)
    retriever = build_retriever(index, embed_model, cfg)

    # Failure mode 1: domanda senza contenuto semantico.
    # Failure mode 2: domanda fuori dal dominio del documento.
    for title, query in [
        ("Failure Mode 1 — empty/nonsense query", "asdfjkl qwerty zzz"),
        ("Failure Mode 2 — out-of-domain query", "What is the recipe for making sourdough bread?"),
    ]:
        print(f"\n### {title}")
        response = query_engine.query(query)
        print("Query :", query)
        print("Answer:", str(response))
        top = response.source_nodes[0].score if response.source_nodes else None
        print("Top retrieved node score:", f"{top:.4f}" if top is not None else "none")

    # Failure mode 3: guardrail a soglia sullo score migliore.
    threshold = cfg["guardrail"]["score_threshold"]
    print(f"\n### Failure Mode 3 — score-threshold guardrail (threshold={threshold})")
    for label, question in [("in-domain ", "What is reward hacking?"), ("out-domain", "What is the capital of France?")]:
        print(f"Q ({label}): {question}")
        print(f"A             : {safe_query(query_engine, retriever, question, threshold)}\n")


def cmd_reset(cfg: dict, args) -> None:
    persist_dir = cfg["chroma"]["persist_dir"]
    if not args.yes:
        answer = input(f"Cancellare {persist_dir}? [s/N] ").strip().lower()
        if answer not in ("s", "si", "sì", "y", "yes"):
            print("Annullato.")
            return
    reset_index(cfg)


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m local_rag.cli", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="percorso di config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("index", help="costruisce o riapre l'indice Chroma").set_defaults(func=cmd_index)

    p_ask = sub.add_parser("ask", help="domanda al RAG (retrieval + LLM)")
    p_ask.add_argument("question", nargs="?", default=DEFAULT_QUERY)
    p_ask.add_argument("--show-prompt", action="store_true", help="stampa i prompt del query engine")
    p_ask.set_defaults(func=cmd_ask)

    sub.add_parser("eval", help="hit-rate e MRR sul gold set (senza LLM)").set_defaults(func=cmd_eval)
    sub.add_parser("failures", help="i tre failure mode del tutorial").set_defaults(func=cmd_failures)

    p_reset = sub.add_parser("reset", help="cancella l'indice su disco")
    p_reset.add_argument("--yes", action="store_true", help="non chiedere conferma")
    p_reset.set_defaults(func=cmd_reset)

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    args.func(load_config(args.config), args)


if __name__ == "__main__":
    main()
