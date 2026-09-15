"""Generate isolated Stage-1 dense and hybrid retrieval runs.

The script deliberately does not mutate ``runs/`` or the production index.
It writes each run atomically so interrupted/repeated experiments cannot append
duplicate query blocks.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import jieba
from rank_bm25 import BM25Okapi


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiment.evaluate_retrieval import (
    detect_text_encoding,
    evaluate_run,
    load_qrels,
    load_run,
)
from scripts.pipeline.retriever import FaissRetriever


def load_topics(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding=detect_text_encoding(path), newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"qid", "query"}.issubset(reader.fieldnames):
            raise ValueError("topics must contain qid and query columns")
        return [
            {
                "qid": (row.get("qid") or "").strip(),
                "query": (row.get("query") or "").strip(),
                "type": (row.get("type") or "").strip(),
            }
            for row in reader
            if (row.get("qid") or "").strip() and (row.get("query") or "").strip()
        ]


class Stage1BM25:
    """Stable BM25 baseline over raw Chunk text (independent of embedding model)."""

    def __init__(self, documents: list[dict[str, Any]]) -> None:
        self.documents = documents
        self.model = BM25Okapi([list(jieba.cut(str(doc.get("text") or ""))) for doc in documents])

    def search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        scores = self.model.get_scores(list(jieba.cut(query)))
        order = sorted(range(len(scores)), key=lambda index: float(scores[index]), reverse=True)
        results: list[dict[str, Any]] = []
        for index in order[:top_k]:
            item = dict(self.documents[index])
            item["bm25_score"] = float(scores[index])
            results.append(item)
        return results


def reciprocal_rank_fusion(
    dense: Iterable[dict[str, Any]],
    sparse: Iterable[dict[str, Any]],
    *,
    rrf_k: int,
    dense_weight: float,
    bm25_weight: float,
    top_k: int,
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for rank, result in enumerate(dense, 1):
        doc_id = str(result["id"])
        merged.setdefault(doc_id, dict(result))
        merged[doc_id]["dense_rank"] = rank
        merged[doc_id]["dense_score"] = float(result["score"])
        merged[doc_id]["rrf_score"] = merged[doc_id].get("rrf_score", 0.0) + (
            dense_weight / (rrf_k + rank)
        )
    for rank, result in enumerate(sparse, 1):
        doc_id = str(result["id"])
        merged.setdefault(doc_id, dict(result))
        merged[doc_id]["bm25_rank"] = rank
        merged[doc_id]["bm25_score"] = float(result["bm25_score"])
        merged[doc_id]["rrf_score"] = merged[doc_id].get("rrf_score", 0.0) + (
            bm25_weight / (rrf_k + rank)
        )
    return sorted(merged.values(), key=lambda item: item["rrf_score"], reverse=True)[:top_k]


def write_run(path: Path, rows: list[tuple[str, str, int, float]], tag: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for qid, doc_id, rank, score in rows:
            handle.write(f"{qid} Q0 {doc_id} {rank} {score:.8f} {tag}\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--label", required=True, help="Short experiment label, e.g. minilm-contextual")
    parser.add_argument("--topics", type=Path, default=PROJECT_ROOT / "topics.csv")
    parser.add_argument("--qrels", type=Path, default=PROJECT_ROOT / "qrels.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "experiments" / "stage1" / "runs")
    parser.add_argument("--dense-k", type=int, default=75)
    parser.add_argument("--bm25-k", type=int, default=50)
    parser.add_argument("--rrf-k", type=int, default=20)
    parser.add_argument("--output-k", type=int, default=50)
    parser.add_argument("--dense-weight", type=float, default=1.0)
    parser.add_argument("--bm25-weight", type=float, default=1.0)
    parser.add_argument("--with-rerank", action="store_true")
    parser.add_argument("--rerank-k", type=int, default=50)
    parser.add_argument("--device")
    args = parser.parse_args()

    if min(args.dense_k, args.bm25_k, args.rrf_k, args.output_k, args.rerank_k) <= 0:
        raise ValueError("k values must be positive")

    index_dir = args.index_dir.resolve()
    index_config = json.loads((index_dir / "kb_config.json").read_text(encoding="utf-8"))
    topics = load_topics(args.topics.resolve())
    print(f"topics={len(topics)} index={index_dir}")
    started = time.perf_counter()
    retriever = FaissRetriever(index_dir=index_dir, device=args.device)
    model_loaded = time.perf_counter()
    bm25 = Stage1BM25(retriever.metadatas)
    bm25_loaded = time.perf_counter()
    reranker = None
    reranker_loaded = bm25_loaded
    if args.with_rerank:
        from app.reranker import BGECrossEncoderReranker

        reranker = BGECrossEncoderReranker()
        reranker_loaded = time.perf_counter()

    dense_rows: list[tuple[str, str, int, float]] = []
    hybrid_rows: list[tuple[str, str, int, float]] = []
    rerank_rows: list[tuple[str, str, int, float]] = []
    query_latencies: list[float] = []
    for number, topic in enumerate(topics, 1):
        query_started = time.perf_counter()
        dense = retriever.search(topic["query"], top_k=args.dense_k)
        sparse = bm25.search(topic["query"], top_k=args.bm25_k)
        hybrid = reciprocal_rank_fusion(
            dense,
            sparse,
            rrf_k=args.rrf_k,
            dense_weight=args.dense_weight,
            bm25_weight=args.bm25_weight,
            top_k=args.output_k,
        )
        query_latencies.append(time.perf_counter() - query_started)
        for rank, result in enumerate(dense[: args.output_k], 1):
            dense_rows.append((topic["qid"], str(result["id"]), rank, float(result["score"])))
        for rank, result in enumerate(hybrid, 1):
            hybrid_rows.append((topic["qid"], str(result["id"]), rank, float(result["rrf_score"])))
        if reranker is not None:
            reranked = reranker.rerank(
                topic["query"],
                [dict(item) for item in hybrid[: args.rerank_k]],
                top_k=args.output_k,
            )
            for rank, result in enumerate(reranked, 1):
                rerank_rows.append(
                    (topic["qid"], str(result["id"]), rank, float(result["rerank_score"]))
                )
        if number == 1 or number % 10 == 0 or number == len(topics):
            print(f"[{number:02d}/{len(topics)}] {topic['qid']}")

    dense_path = args.output_dir / f"{args.label}_dense.run"
    hybrid_path = args.output_dir / f"{args.label}_hybrid.run"
    write_run(dense_path, dense_rows, f"{args.label}_dense")
    write_run(hybrid_path, hybrid_rows, f"{args.label}_hybrid")
    rerank_path = args.output_dir / f"{args.label}_rerank.run"
    if reranker is not None:
        write_run(rerank_path, rerank_rows, f"{args.label}_rerank")

    qrels, _ = load_qrels(args.qrels.resolve())
    systems = {
        "dense": evaluate_run(load_run(dense_path), qrels),
        "hybrid": evaluate_run(load_run(hybrid_path), qrels),
    }
    if reranker is not None:
        systems["rerank"] = evaluate_run(load_run(rerank_path), qrels)
    total_seconds = time.perf_counter() - started
    report = {
        "label": args.label,
        "index_dir": str(index_dir),
        "embedding": retriever.embedding_spec.to_dict(),
        "index": {
            "text_mode": index_config.get("text_mode", "legacy"),
            "dimension": index_config.get("dim", retriever.index.d),
            "index_bytes": (index_dir / "kb.index").stat().st_size,
            "offline_encode_seconds": index_config.get("encode_seconds"),
        },
        "parameters": {
            "dense_k": args.dense_k,
            "bm25_k": args.bm25_k,
            "rrf_k": args.rrf_k,
            "output_k": args.output_k,
            "dense_weight": args.dense_weight,
            "bm25_weight": args.bm25_weight,
            "rerank_k": args.rerank_k if args.with_rerank else None,
        },
        "topics": len(topics),
        "timing_seconds": {
            "model_load": round(model_loaded - started, 3),
            "bm25_build": round(bm25_loaded - model_loaded, 3),
            "reranker_load": round(reranker_loaded - bm25_loaded, 3),
            "queries_total": round(sum(query_latencies), 3),
            "query_mean": round(sum(query_latencies) / max(len(query_latencies), 1), 4),
            "total": round(total_seconds, 3),
        },
        "systems": systems,
        "run_files": {
            "dense": str(dense_path.resolve()),
            "hybrid": str(hybrid_path.resolve()),
            **({"rerank": str(rerank_path.resolve())} if reranker is not None else {}),
        },
    }
    report_path = args.output_dir / f"{args.label}_metrics.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, metrics in systems.items():
        print(
            f"{name:7s} MRR@10={metrics['MRR@10']:.4f} "
            f"NDCG@10={metrics['NDCG@10']:.4f} Recall@50={metrics['Recall@50']:.4f}"
        )
    print(f"saved={report_path.resolve()}")


if __name__ == "__main__":
    main()
