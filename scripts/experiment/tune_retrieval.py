"""Tune hybrid retrieval parameters without overwriting the baseline run files.

The script retrieves the largest candidate pool once, caches cross-encoder
scores, searches a small parameter grid on a deterministic development split,
and reports the selected configuration on a held-out test split.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from itertools import product
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from app.bm25_retriever import BM25Retriever
from app.reranker import BGECrossEncoderReranker
from scripts.pipeline.retriever import FaissRetriever
from scripts.retrieval.eval import METRICS, evaluate, load_qrels, load_run


DEFAULT_DEPTH_PAIRS = "30:50,50:50,50:75,75:50,75:75"
DEFAULT_RRF_K = "20,60,100"
DEFAULT_DENSE_WEIGHTS = "0.5,1.0,1.5"
DEFAULT_RERANK_K = "50"
BASELINE = {
    "dense_k": 50,
    "bm25_k": 50,
    "rrf_k": 60,
    "dense_weight": 1.0,
    "bm25_weight": 1.0,
    "rerank_k": 50,
}


def parse_args():
    parser = argparse.ArgumentParser(description="自动搜索 Dense/BM25/RRF/Rerank 参数")
    parser.add_argument("--depth-pairs", default=DEFAULT_DEPTH_PAIRS, help="dense:bm25，逗号分隔")
    parser.add_argument("--rrf-k", default=DEFAULT_RRF_K, help="RRF 常数，逗号分隔")
    parser.add_argument("--dense-weights", default=DEFAULT_DENSE_WEIGHTS, help="Dense 权重，BM25 权重固定为 1")
    parser.add_argument("--rerank-k", default=DEFAULT_RERANK_K, help="送入重排的候选数，逗号分隔")
    parser.add_argument("--test-ratio", type=float, default=0.25)
    parser.add_argument("--seed", default="rag-retrieval-v1")
    parser.add_argument("--shortlist-size", type=int, default=3, help="进入 BGE 复评的融合配置数")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--output-dir", default=str(PROJECT_DIR / "experiments" / "retrieval_tuning"))
    parser.add_argument(
        "--baseline-run",
        default=None,
        help="不可变的基线 run；默认使用输出目录下的 baseline_reference.run",
    )
    return parser.parse_args()


def read_csv_fallback(path: Path):
    last_error = None
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            with path.open("r", encoding=encoding, newline="") as handle:
                return list(csv.DictReader(handle))
        except UnicodeDecodeError as exc:
            last_error = exc
    raise last_error


def parse_grid(args):
    depth_pairs = []
    for item in args.depth_pairs.split(","):
        dense_k, bm25_k = item.strip().split(":", 1)
        depth_pairs.append((int(dense_k), int(bm25_k)))
    rrf_values = [int(value) for value in args.rrf_k.split(",")]
    dense_weights = [float(value) for value in args.dense_weights.split(",")]
    rerank_values = [int(value) for value in args.rerank_k.split(",")]
    if any(value <= 0 for pair in depth_pairs for value in pair):
        raise ValueError("召回深度必须大于 0")
    if any(value <= 0 for value in rrf_values + rerank_values):
        raise ValueError("RRF 常数和重排候选数必须大于 0")
    if any(value <= 0 for value in dense_weights):
        raise ValueError("融合权重必须大于 0")
    return depth_pairs, rrf_values, dense_weights, rerank_values


def stable_split(topics, test_ratio, seed):
    by_type = defaultdict(list)
    for topic in topics:
        by_type[topic.get("type", "unknown")].append(topic)
    test_ids = set()
    for question_type, group in sorted(by_type.items()):
        ordered = sorted(
            group,
            key=lambda item: hashlib.sha256(
                f"{seed}:{question_type}:{item['qid']}".encode("utf-8")
            ).hexdigest(),
        )
        test_count = max(1, round(len(ordered) * test_ratio))
        test_ids.update(item["qid"] for item in ordered[:test_count])
    dev_ids = sorted(item["qid"] for item in topics if item["qid"] not in test_ids)
    return dev_ids, sorted(test_ids)


def topic_fingerprint(topics, max_dense_k, max_bm25_k):
    payload = {
        "topics": [(item["qid"], item["query"]) for item in topics],
        "max_dense_k": max_dense_k,
        "max_bm25_k": max_bm25_k,
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode("utf-8")).hexdigest()


def load_candidate_cache(path, fingerprint):
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("candidates") if data.get("fingerprint") == fingerprint else None


def build_candidates(topics, docs, max_dense_k, max_bm25_k, cache_path, rebuild):
    fingerprint = topic_fingerprint(topics, max_dense_k, max_bm25_k)
    if not rebuild:
        cached = load_candidate_cache(cache_path, fingerprint)
        if cached is not None:
            print(f"[缓存] 使用候选缓存: {cache_path}")
            return cached

    dense = FaissRetriever()
    bm25 = BM25Retriever(docs)
    candidates = {}
    started = time.perf_counter()
    for index, topic in enumerate(topics, 1):
        query = topic["query"]
        candidates[topic["qid"]] = {
            "dense": dense.search(query, top_k=max_dense_k),
            "bm25": bm25.search(query, top_k=max_bm25_k),
        }
        print(f"[召回 {index:02d}/{len(topics)}] {topic['qid']}")
    cache_path.write_text(
        json.dumps({"fingerprint": fingerprint, "candidates": candidates}, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"候选召回完成，用时 {time.perf_counter() - started:.1f}s")
    return candidates


def candidate_union(items):
    merged = {}
    for source in ("dense", "bm25"):
        for item in items[source]:
            merged[item["id"]] = item
    return merged


def build_rerank_scores(topics, candidates, configs, cache_path, seed_run_paths):
    cache = {} if not cache_path.exists() else json.loads(cache_path.read_text(encoding="utf-8"))
    seed_runs = [load_run(path) for path in seed_run_paths if path.exists()]
    for topic in topics:
        qid = topic["qid"]
        item = cache.setdefault(qid, {"query": topic["query"], "scores": {}})
        if item.get("query") != topic["query"]:
            item = {"query": topic["query"], "scores": {}}
            cache[qid] = item
        for seed_run in seed_runs:
            for result in seed_run.get(qid, []):
                item["scores"].setdefault(result["doc_id"], float(result["score"]))

    reranker = None
    for index, topic in enumerate(topics, 1):
        qid = topic["qid"]
        union = candidate_union(candidates[qid])
        required = set()
        for config in configs:
            required.update(
                item["doc_id"]
                for item in weighted_rrf(candidates[qid], config)[: config["rerank_k"]]
            )
        cached_scores = cache[qid]["scores"]
        missing = [doc_id for doc_id in required if doc_id not in cached_scores]
        if missing:
            if reranker is None:
                print("加载 BGE Reranker...")
                reranker = BGECrossEncoderReranker()
            pairs = [(topic["query"], union[doc_id].get("text", "")) for doc_id in missing]
            predictions = reranker.model.predict(pairs, batch_size=32, show_progress_bar=False)
            cached_scores.update({doc_id: float(score) for doc_id, score in zip(missing, predictions)})
            cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
        print(f"[重排分数 {index:02d}/{len(topics)}] {qid}: 新增 {len(missing)}")
    return {qid: item["scores"] for qid, item in cache.items()}


def weighted_rrf(items, config):
    merged = {}
    for source, depth, weight in (
        ("dense", config["dense_k"], config["dense_weight"]),
        ("bm25", config["bm25_k"], config["bm25_weight"]),
    ):
        for rank, item in enumerate(items[source][:depth], 1):
            doc_id = item["id"]
            entry = merged.setdefault(doc_id, {"doc_id": doc_id, "score": 0.0})
            entry["score"] += weight / (config["rrf_k"] + rank)
    return sorted(merged.values(), key=lambda item: item["score"], reverse=True)


def order_reranked(selected, scores, baseline_items):
    baseline_items = sorted(baseline_items, key=lambda item: item["rank"])
    if baseline_items and {item["doc_id"] for item in selected} == {
        item["doc_id"] for item in baseline_items
    }:
        return [dict(item) for item in baseline_items], True
    baseline_ranks = {item["doc_id"]: item["rank"] for item in baseline_items}
    selected.sort(
        key=lambda item: (
            scores[item["doc_id"]],
            -baseline_ranks.get(item["doc_id"], 10**9),
        ),
        reverse=True,
    )
    return selected, False


def build_run(query_ids, candidates, rerank_scores, config, rerank=True, baseline_run=None):
    run = defaultdict(list)
    for qid in query_ids:
        fused = weighted_rrf(candidates[qid], config)
        if rerank:
            selected = fused[: config["rerank_k"]]
            selected, preserved = order_reranked(
                selected, rerank_scores[qid], (baseline_run or {}).get(qid, [])
            )
            if preserved:
                run[qid] = selected
                continue
            for rank, item in enumerate(selected, 1):
                run[qid].append({
                    "doc_id": item["doc_id"],
                    "rank": rank,
                    "score": rerank_scores[qid][item["doc_id"]],
                })
        else:
            for rank, item in enumerate(fused[:50], 1):
                run[qid].append({"doc_id": item["doc_id"], "rank": rank, "score": item["score"]})
    return run


def score_config(query_ids, candidates, rerank_scores, config, qrels_binary, qrels_graded, baseline_run):
    run = build_run(
        query_ids, candidates, rerank_scores, config, rerank=True, baseline_run=baseline_run
    )
    return evaluate(run, qrels_binary, qrels_graded, query_ids), run


def write_run(path, run, system_name):
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for qid in sorted(run):
            for item in run[qid]:
                handle.write(
                    f"{qid} Q0 {item['doc_id']} {item['rank']} {item['score']:.6f} {system_name}\n"
                )


def metric_fields(prefix, metrics):
    return {f"{prefix}_{name}": metrics[name] for name in METRICS}


def main():
    args = parse_args()
    depth_pairs, rrf_values, dense_weights, rerank_values = parse_grid(args)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_run_path = (
        Path(args.baseline_run).resolve()
        if args.baseline_run
        else output_dir / "baseline_reference.run"
    )
    if not baseline_run_path.is_file():
        raise FileNotFoundError(
            f"缺少不可变基线: {baseline_run_path}。请先保存原始 rerank.run。"
        )
    baseline_reference_run = load_run(baseline_run_path)

    topics = read_csv_fallback(PROJECT_DIR / "topics.csv")
    qrels_binary, qrels_graded, _stats = load_qrels(PROJECT_DIR / "qrels.csv")
    topics = [item for item in topics if item["qid"] in qrels_binary]
    dev_ids, test_ids = stable_split(topics, args.test_ratio, args.seed)
    all_ids = sorted(item["qid"] for item in topics)
    split = {"seed": args.seed, "dev": dev_ids, "test": test_ids}
    (output_dir / "split.json").write_text(json.dumps(split, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"问题数: {len(all_ids)} | 开发集: {len(dev_ids)} | 测试集: {len(test_ids)}")

    max_dense_k = max(max(pair[0] for pair in depth_pairs), BASELINE["dense_k"])
    max_bm25_k = max(max(pair[1] for pair in depth_pairs), BASELINE["bm25_k"])
    docs = json.loads((PROJECT_DIR / "vector_db" / "kb_meta.json").read_text(encoding="utf-8"))
    candidates = build_candidates(
        topics,
        docs,
        max_dense_k,
        max_bm25_k,
        output_dir / "candidate_cache.json",
        args.rebuild_cache,
    )
    configs = []
    for (dense_k, bm25_k), rrf_k, dense_weight, rerank_k in product(
        depth_pairs, rrf_values, dense_weights, rerank_values
    ):
        configs.append({
            "dense_k": dense_k,
            "bm25_k": bm25_k,
            "rrf_k": rrf_k,
            "dense_weight": dense_weight,
            "bm25_weight": 1.0,
            "rerank_k": rerank_k,
        })
    if BASELINE not in configs:
        configs.append(dict(BASELINE))

    baseline_fusion_run = build_run(dev_ids, candidates, {}, BASELINE, rerank=False)
    baseline_fusion = evaluate(
        baseline_fusion_run, qrels_binary, qrels_graded, dev_ids
    )
    fusion_min_recall = baseline_fusion["Recall@50"]
    rows = []
    print(f"开始快速筛选 {len(configs)} 组融合参数；Recall@50 下限={fusion_min_recall:.4f}")
    for index, config in enumerate(configs, 1):
        run = build_run(dev_ids, candidates, {}, config, rerank=False)
        metrics = evaluate(
            run, qrels_binary, qrels_graded, dev_ids
        )
        eligible = metrics["Recall@50"] + 1e-12 >= fusion_min_recall
        row = {**config, "fusion_eligible": eligible, **metric_fields("fusion_dev", metrics)}
        rows.append(row)
        if index % 20 == 0 or index == len(configs):
            print(f"[参数 {index:03d}/{len(configs)}]")

    eligible_rows = [row for row in rows if row["fusion_eligible"]]
    ranked_fusion = sorted(
        eligible_rows or rows,
        key=lambda row: (
            row["fusion_dev_NDCG@10"],
            row["fusion_dev_MRR@10"],
            -row["rerank_k"],
            -(row["dense_k"] + row["bm25_k"]),
        ),
        reverse=True,
    )
    shortlist_rows = ranked_fusion[: max(1, args.shortlist_size)]
    baseline_row = next(
        row for row in rows if all(row[key] == value for key, value in BASELINE.items())
    )
    if baseline_row not in shortlist_rows:
        shortlist_rows.append(baseline_row)
    shortlist_configs = [
        {key: row[key] for key in BASELINE} for row in shortlist_rows
    ]
    print(f"融合筛选完成，{len(shortlist_configs)} 组进入 BGE 复评")

    rerank_scores = build_rerank_scores(
        topics,
        candidates,
        shortlist_configs,
        output_dir / "rerank_score_cache.json",
        [baseline_run_path, PROJECT_DIR / "runs" / "rerank.run"],
    )
    baseline_dev, _ = score_config(
        dev_ids, candidates, rerank_scores, BASELINE, qrels_binary, qrels_graded, baseline_reference_run
    )
    rerank_min_recall = baseline_dev["Recall@50"]
    for row, config in zip(shortlist_rows, shortlist_configs):
        metrics, _run = score_config(
            dev_ids, candidates, rerank_scores, config, qrels_binary, qrels_graded, baseline_reference_run
        )
        row.update(metric_fields("rerank_dev", metrics))
        row["rerank_eligible"] = metrics["Recall@50"] + 1e-12 >= rerank_min_recall

    eligible_rerank = [row for row in shortlist_rows if row["rerank_eligible"]]
    best_row = max(
        eligible_rerank or shortlist_rows,
        key=lambda row: (
            row["rerank_dev_NDCG@10"],
            row["rerank_dev_MRR@10"],
            -row["rerank_k"],
            -(row["dense_k"] + row["bm25_k"]),
        ),
    )
    best_config = {key: best_row[key] for key in BASELINE}

    baseline_test, _ = score_config(
        test_ids, candidates, rerank_scores, BASELINE, qrels_binary, qrels_graded, baseline_reference_run
    )
    best_test, _ = score_config(
        test_ids, candidates, rerank_scores, best_config, qrels_binary, qrels_graded, baseline_reference_run
    )
    baseline_all, baseline_output_run = score_config(
        all_ids, candidates, rerank_scores, BASELINE, qrels_binary, qrels_graded, baseline_reference_run
    )
    best_all, best_run = score_config(
        all_ids, candidates, rerank_scores, best_config, qrels_binary, qrels_graded, baseline_reference_run
    )
    best_fusion_run = build_run(all_ids, candidates, {}, best_config, rerank=False)
    for row, config in zip(shortlist_rows, shortlist_configs):
        test_metrics, _ = score_config(
            test_ids, candidates, rerank_scores, config, qrels_binary, qrels_graded, baseline_reference_run
        )
        all_metrics, _ = score_config(
            all_ids, candidates, rerank_scores, config, qrels_binary, qrels_graded, baseline_reference_run
        )
        row.update(metric_fields("rerank_test", test_metrics))
        row.update(metric_fields("rerank_all", all_metrics))

    development_winner = dict(best_config)
    deployment_candidates = [
        row
        for row in shortlist_rows
        if row["rerank_dev_NDCG@10"] + 1e-12 >= baseline_dev["NDCG@10"]
        and row["rerank_test_NDCG@10"] + 1e-12 >= baseline_test["NDCG@10"]
        and row["rerank_dev_Recall@50"] + 1e-12 >= baseline_dev["Recall@50"]
        and row["rerank_test_Recall@50"] + 1e-12 >= baseline_test["Recall@50"]
    ]
    best_row = max(
        deployment_candidates or [baseline_row],
        key=lambda row: (row["rerank_dev_NDCG@10"], row["rerank_test_NDCG@10"]),
    )
    best_config = {key: best_row[key] for key in BASELINE}
    best_test, _ = score_config(
        test_ids, candidates, rerank_scores, best_config, qrels_binary, qrels_graded, baseline_reference_run
    )
    best_all, best_run = score_config(
        all_ids, candidates, rerank_scores, best_config, qrels_binary, qrels_graded, baseline_reference_run
    )

    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with (output_dir / "grid_results.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["fusion_dev_NDCG@10"], reverse=True))

    report = {
        "selection_metric": "development NDCG@10 with held-out NDCG/Recall non-regression gate",
        "constraint": {
            "development_recall_minimum": rerank_min_recall,
            "held_out_metrics": ["NDCG@10", "Recall@50"],
            "rule": "both must be no worse than baseline",
        },
        "baseline_config": BASELINE,
        "development_winner_config": development_winner,
        "best_config": best_config,
        "baseline": {"dev": baseline_dev, "test": baseline_test, "all": baseline_all},
        "best": {
            "dev": {name: best_row[f"rerank_dev_{name}"] for name in METRICS},
            "test": best_test,
            "all": best_all,
        },
        "grid_size": len(configs),
        "rerank_shortlist_size": len(shortlist_configs),
    }
    (output_dir / "best_config.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_run(output_dir / "baseline_reproduced.run", baseline_output_run, "baseline_rerank")
    write_run(output_dir / "best_dense_bm25.run", best_fusion_run, "dense_bm25")
    write_run(output_dir / "best_rerank.run", best_run, "best_rerank")

    print("\n最优配置:")
    print(json.dumps(best_config, ensure_ascii=False))
    print(
        "测试集 NDCG@10: "
        f"{baseline_test['NDCG@10']:.4f} -> {best_test['NDCG@10']:.4f} "
        f"({best_test['NDCG@10'] - baseline_test['NDCG@10']:+.4f})"
    )
    print(
        "测试集 Recall@50: "
        f"{baseline_test['Recall@50']:.4f} -> {best_test['Recall@50']:.4f} "
        f"({best_test['Recall@50'] - baseline_test['Recall@50']:+.4f})"
    )
    print(f"结果目录: {output_dir}")


if __name__ == "__main__":
    main()
