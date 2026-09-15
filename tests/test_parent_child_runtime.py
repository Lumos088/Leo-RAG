from scripts.pipeline.context_assembly import (
    build_parent_context,
    infer_query_type,
    insert_baseline_rescue,
)
from scripts.pipeline.parent_child_retrieval import ParentChildRetriever, load_parent_child_config


def test_query_type_router_matches_supported_intents():
    assert infer_query_type("什么是并查集Union-Find") == "definition"
    assert infer_query_type("TCP 和 UDP 的区别是什么") == "comparison"
    assert infer_query_type("死锁的四个必要条件是什么") == "enumeration"
    assert infer_query_type("图的两种基本遍历算法是什么") == "enumeration"
    assert infer_query_type("B 树和 B+ 树的主要特性有哪些") == "comparison"


def test_baseline_rescue_is_unique_and_inserted_at_baseline_rank():
    parents = [{"id": "p1", "text": "one"}, {"id": "p3", "text": "three"}]
    baseline = [
        {"id": "p1", "rerank_score": 0.9},
        {"id": "p2", "rerank_score": 0.8},
    ]
    store = {"p1": {"id": "p1", "text": "one"}, "p2": {"id": "p2", "text": "two"}}
    combined, rescue_id = insert_baseline_rescue(parents, baseline, store)
    assert rescue_id == "p2"
    assert [item["id"] for item in combined] == ["p1", "p2", "p3"]
    assert combined[1]["context_role"] == "baseline_rescue"


def test_parent_context_keeps_parent_citations_with_child_highlight():
    parents = [{
        "id": "p1", "parent_id": "p1", "text": "完整上下文", "course": "Data Structure",
        "document_title": "book", "chapter_path": "chapter", "parent_score": 0.5,
    }]
    highlights = {"p1": [{"id": "c1", "text": "关键证据"}]}
    context, citations, retrieval = build_parent_context(parents, max_chars=16000, highlights=highlights)
    assert "关键证据" in context and "完整上下文" in context
    assert citations[0]["id"] == "p1"
    assert citations[0]["highlight_child_ids"] == ["c1"]
    assert retrieval[0]["highlight_child_ids"] == ["c1"]


def test_production_parent_child_config_is_enabled(project_root):
    config = load_parent_child_config(project_root / "config" / "parent_child.json")
    assert config["enabled"] is True
    assert config["parent_rerank_enabled"] is True
    assert config["parent_rerank_mode"] == "rrf"
    assert config["baseline_rescue_enabled"] is True
    assert config["definition_child_highlight_enabled"] is True
    assert config["parent_top_k"] == 6


def test_disable_reranking_is_respected():
    class Dense:
        def search(self, query, top_k):
            return [{"id": "c1", "parent_id": "p1", "text": "evidence", "score": 0.7}]

    class ForbiddenReranker:
        def rerank(self, *args, **kwargs):
            raise AssertionError("reranker must not be called")

    engine = ParentChildRetriever.__new__(ParentChildRetriever)
    engine.config = {
        "variant": "test", "dense_k": 5, "bm25_k": 5, "rerank_candidates": 5,
        "parent_top_k": 1, "parent_rerank_enabled": True, "parent_rerank_candidates": 5,
        "parent_score_aggregation": "max", "parent_rank_bonus_weight": 0.05,
    }
    engine.child_retriever = Dense()
    engine.bm25 = None
    engine.reranker = ForbiddenReranker()
    engine.parents = {"p1": {"id": "p1", "text": "parent"}}
    result = engine.search("query", parent_top_k=1, use_bm25=False, use_rerank=False)
    assert result["parents"][0]["id"] == "p1"


def pytest_generate_tests(metafunc):
    if "project_root" in metafunc.fixturenames:
        from pathlib import Path
        metafunc.parametrize("project_root", [Path(__file__).resolve().parents[1]])
