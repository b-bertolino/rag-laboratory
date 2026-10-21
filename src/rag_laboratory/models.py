from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama


def build_llm(cfg: dict) -> Ollama:
    c = cfg["llm"]
    return Ollama(
        model=c["model"],
        base_url=c["base_url"],
        temperature=c["temperature"],
        request_timeout=c["request_timeout"],
    )


def build_embed_model(cfg: dict) -> OllamaEmbedding:
    c = cfg["embedding"]
    return OllamaEmbedding(model_name=c["model"], base_url=c["base_url"])
