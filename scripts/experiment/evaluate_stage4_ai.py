"""Evaluate Stage 4 Parent-Child retrieval against parent-level qrels."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiment.evaluate_retrieval import detect_text_encoding
from scripts.pipeline.parent_child_retrieval import ParentChildRetriever


def load_topics(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding=detect_text_encoding(path), newline="") as handle:
        return [
            {"qid": row["qid"].strip(), "query": row["query"].strip()}
            for row in csv.DictReader(handle)
            if row.get("qid") and row.get("query")
        ]


def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = defaultdict(dict)
    with path.open("r", encoding=detect_text_encoding(path), newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("qid") and row.get("doc_id") and row.get("rel"):
                result[row["qid"].strip()][row["doc_id"].strip()] = int(float(row["rel"]))
    return dict(result)


def evaluate(ranking: list[str], judged: dict[str, int], k: int) -> dict[str, float]:
    ranking = ranking[:k]
    positive = {doc_id for doc_id, rel in judged.items() if rel > 0}
    hits = [1 if doc_id in positive else 0 for doc_id in ranking]
    first = next((index + 1 for index, hit in enumerate(hits) if hit), None)
    dcg = sum(judged.get(doc_id, 0) / math.log2(index + 2) for index, doc_id in enumerate(ranking))
    ideal = sorted(judged.values(), reverse=True)[:k]
    idcg = sum(rel / math.log2(index + 2) for index, rel in enumerate(ideal))
    return {
        "success": float(any(hits)),
        "precision": sum(hits) / k,
        "recall": sum(hits) / len(positive) if positive else 0.0,
        "mrr": 1.0 / first if first else 0.0,
        "ndcg": dcg / idcg if idcg else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config/parent_child_stage4.json")
    parser.add_argument("--topics", type=Path, default=PROJECT_ROOT / "data/stage4/evaluation/topics_ai.csv")
    parser.add_argument("--qrels", type=Path, default=PROJECT_ROOT / "data/stage4/evaluation/qrels_ai.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "experiments/stage4/ai_eval")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    topics = load_topics(args.topics.resolve())
    qrels = load_qrels(args.qrels.resolve())
    engine = ParentChildRetriever(
        config_path=args.config.resolve(),
        device=args.device,
        load_reranker=not args.no_rerank,
    )
    rows = []
    run_rows = []
    for index, topic in enumerate(topics, 1):
        started = time.perf_counter()
        result = engine.search(topic["query"], use_rerank=not args.no_rerank)
        ranking = [str(item["id"]) for item in result["parents"][: args.top_k]]
        metrics = evaluate(ranking, qrels.get(topic["qid"], {}), args.top_k)
        rows.append({"qid": topic["qid"], "query": topic["query"], "latency_seconds": time.perf_counter() - started, **metrics})
        for rank, item in enumerate(result["parents"][: args.top_k], 1):
            run_rows.append({"qid": topic["qid"], "doc_id": item["id"], "rank": rank, "score": item.get("score", item.get("rrf_score", 0.0))})
        print(f"[{index:02d}/{len(topics)}] {topic['qid']} success={metrics['success']:.0f} mrr={metrics['mrr']:.3f}")

    summary = {
        "schema_version": "stage4-ai-evaluation-v1",
        "config": str(args.config.resolve()),
        "topics": len(rows),
        "top_k": args.top_k,
        "rerank": not args.no_rerank,
        **{name: round(statistics.fmean(row[name] for row in rows), 4) for name in ("success", "precision", "recall", "mrr", "ndcg")},
        "latency_mean_seconds": round(statistics.fmean(row["latency_seconds"] for row in rows), 4),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "per_query.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.output_dir / "parents.run.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["qid", "doc_id", "rank", "score"])
        writer.writeheader()
        writer.writerows(run_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
