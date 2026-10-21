from types import SimpleNamespace

import pytest

from rag_laboratory.evaluate import evaluate_retrieval, first_hit_rank


class FakeRetriever:
    """Restituisce per ogni domanda una lista di testi decisa dal test."""

    def __init__(self, answers: dict[str, list[str]]):
        self.answers = answers

    def retrieve(self, question):
        return [SimpleNamespace(text=t) for t in self.answers[question]]


def test_first_hit_rank_needs_all_keywords():
    texts = ["Reward signal only", "Reward hacking is UNINTENDED behaviour"]
    assert first_hit_rank(texts, ["reward hacking", "unintended"]) == 2
    assert first_hit_rank(texts, ["reward", "deepmind"]) is None


def test_evaluate_retrieval_hit_rate_and_mrr():
    gold = [
        {"id": "a", "question": "qa", "expected_keywords": ["x"]},  # rank 1 -> 1
        {"id": "b", "question": "qb", "expected_keywords": ["x"]},  # rank 2 -> 0.5
        {"id": "c", "question": "qc", "expected_keywords": ["x"]},  # miss   -> 0
    ]
    retriever = FakeRetriever({"qa": ["x", "y"], "qb": ["y", "x"], "qc": ["y", "z"]})

    report = evaluate_retrieval(retriever, gold)

    assert report["hits"] == 2
    assert report["hit_rate"] == pytest.approx(2 / 3)
    assert report["mrr"] == pytest.approx((1 + 0.5 + 0) / 3)
    assert [r["rank"] for r in report["results"]] == [1, 2, None]


def test_tutorial_ranks_give_tutorial_mrr():
    # Rank osservati nel notebook: 1,1,2,1,1,1,1,3 -> MRR 0,8542.
    ranks = [1, 1, 2, 1, 1, 1, 1, 3]
    assert sum(1 / r for r in ranks) / len(ranks) == pytest.approx(0.8542, abs=1e-4)
