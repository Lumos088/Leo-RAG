"""Shared Dense + BM25 fusion used by CLI, Web API, and evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


PROJECT_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_DIR / "config" / "retrieval.json"
DEFAULT_CONFIG = {
    "schema_version": "stage1-production-v1",
    "index_dir": "vector_db/stage1/minilm/raw",
    "dense_k": 75,
    "bm25_k": 50,
    "rrf_k": 20,
    "dense_weight": 1.0,
    "bm25_weight": 1.0,
    "rerank_candidates": 50,
    "final_top_k": 10,
    "max_top_k": 10,
    "max_context_chars": 16000,
}


def load_retrieval_config(path: Path = CONFIG_PATH) -> Dict:
    config = dict(DEFAULT_CONFIG)
    if path.exists():
        config.update(json.loads(path.read_text(encoding="utf-8")))
    for key in (
        "dense_k",
        "bm25_k",
        "rrf_k",
        "rerank_candidates",
        "final_top_k",
        "max_top_k",
        "max_context_chars",
    ):
        config[key] = int(config[key])
        if config[key] <= 0:
            raise ValueError(f"{key} must be greater than zero")
    if config["final_top_k"] > config["max_top_k"]:
        raise ValueError("final_top_k must not exceed max_top_k")
    if not str(config["index_dir"]).strip():
        raise ValueError("index_dir must not be empty")
    for key in ("dense_weight", "bm25_weight"):
        config[key] = float(config[key])
        if config[key] <= 0:
            raise ValueError(f"{key} must be greater than zero")
    return config


RETRIEVAL_CONFIG = load_retrieval_config()


def rrf(rank: int, k: int | None = None) -> float:
    return 1.0 / ((k if k is not None else RETRIEVAL_CONFIG["rrf_k"]) + rank)


def fuse_results(
    dense_results: List[Dict],
    bm25_results: List[Dict],
    config: Dict | None = None,
) -> List[Dict]:
    config = config or RETRIEVAL_CONFIG
    merged: Dict[str, Dict] = {}
    for source, results, weight in (
        ("dense", dense_results, config["dense_weight"]),
        ("bm25", bm25_results, config["bm25_weight"]),
    ):
        for rank, result in enumerate(results, 1):
            doc_id = result["id"]
            item = merged.setdefault(doc_id, dict(result))
            item[f"{source}_rank"] = rank
            if source == "dense":
                item["dense_score"] = result.get("score")
            else:
                item["bm25_score"] = result.get("bm25_score", result.get("score"))
            item["rrf_score"] = item.get("rrf_score", 0.0) + weight * rrf(
                rank, config["rrf_k"]
            )
    return sorted(merged.values(), key=lambda item: item["rrf_score"], reverse=True)


def hybrid_search(retriever, bm25_retriever, query: str, config: Dict | None = None):
    config = config or RETRIEVAL_CONFIG
    dense_results = retriever.search(query, top_k=config["dense_k"])
    bm25_results = bm25_retriever.search(query, top_k=config["bm25_k"])
    return fuse_results(dense_results, bm25_results, config)


def hybrid_rerank(
    retriever,
    bm25_retriever,
    reranker,
    query: str,
    top_k: int,
    config: Dict | None = None,
):
    config = config or RETRIEVAL_CONFIG
    fused = hybrid_search(retriever, bm25_retriever, query, config)
    candidates = fused[: config["rerank_candidates"]]
    return reranker.rerank(query, candidates, top_k=top_k)
