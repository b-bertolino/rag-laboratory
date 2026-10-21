from types import SimpleNamespace

from rag_laboratory.query import safe_query


class FakeRetriever:
    def __init__(self, scores):
        self.scores = scores

    def retrieve(self, question):
        return [SimpleNamespace(score=s) for s in self.scores]


class FakeQueryEngine:
    def __init__(self):
        self.calls = 0

    def query(self, question):
        self.calls += 1
        return "risposta dell'LLM"


def test_above_threshold_calls_llm():
    engine = FakeQueryEngine()
    assert safe_query(engine, FakeRetriever([0.30, 0.45]), "q", threshold=0.40) == "risposta dell'LLM"
    assert engine.calls == 1


def test_below_threshold_blocks_without_llm():
    engine = FakeQueryEngine()
    answer = safe_query(engine, FakeRetriever([0.34, 0.20]), "q", threshold=0.40)
    assert answer.startswith("[GUARDRAIL] Best retrieval score (0.3400)")
    assert engine.calls == 0


def test_no_nodes():
    engine = FakeQueryEngine()
    assert safe_query(engine, FakeRetriever([]), "q", threshold=0.40) == "[GUARDRAIL] No documents retrieved."
    assert engine.calls == 0
