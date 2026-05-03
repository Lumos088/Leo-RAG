#!/usr/bin/env python3
"""
步骤4: 各库独立计算评价指标 (Recall@5, MRR@5)
步骤5: 横向对比指标，确定最优 Chunk Size

- Run: 来自 pool 文件，按 dense retrieval score 排序 (rank 1-5)
- Ground truth: 来自 qrels 文件，bge-reranker-base 判定的相关性
- 每个 chunk size 独立计算，不共用 qrels，不跨库
"""

import os
import json
import csv
import math
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
POOL_DIR = os.path.join(BASE_DIR, "data", "pools")
QRELS_DIR = os.path.join(BASE_DIR, "data", "qrels")
CHUNK_SIZES = [300, 400, 500, 600, 700, 800, 900, 1000]


def load_qrels(chunk_size):
    """加载 qrels，返回 {qid: {doc_id: relevance}}"""
    path = os.path.join(QRELS_DIR, f"qrels_{chunk_size}.tsv")
    qrels = defaultdict(dict)
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            qrels[row["qid"]][row["doc_id"]] = int(row["relevance"])
    return qrels


def load_run_from_pool(chunk_size):
    """
    从 pool 文件加载 run，按 dense retrieval score 降序排列
    返回 {qid: [{"doc_id": ..., "rank": ..., "score": ...}, ...]}
    """
    path = os.path.join(POOL_DIR, f"pool_{chunk_size}.json")
    with open(path, "r", encoding="utf-8") as f:
        pool = json.load(f)

    run = {}
    for entry in pool:
        qid = entry["qid"]
        # pool 中 results 已按 dense score 降序
        ranked = []
        for rank, r in enumerate(entry["results"], 1):
            ranked.append({
                "doc_id": r["id"],
                "rank": rank,
                "score": r["score"],
            })
        run[qid] = ranked
    return run


def calc_recall_at_k(run, qrels, k=5):
    """
    Recall@k = |relevant docs in top-k| / |total relevant docs|
    对每个 query 计算后取平均
    """
    recall_sum = 0.0
    num_queries = 0

    for qid, results in run.items():
        if qid not in qrels:
            continue
        num_queries += 1

        # 该 query 的所有相关文档
        all_relevant = {doc_id for doc_id, rel in qrels[qid].items() if rel > 0}
        if not all_relevant:
            continue

        # top-k 中命中的相关文档
        top_k_docs = {r["doc_id"] for r in results if r["rank"] <= k}
        hits = len(top_k_docs & all_relevant)

        recall_sum += hits / len(all_relevant)

    return recall_sum / num_queries if num_queries > 0 else 0.0


def calc_mrr_at_k(run, qrels, k=5):
    """
    MRR@k = 1/|Q| * Σ(1/rank_of_first_relevant)
    对每个 query 找到 top-k 中第一个相关文档的排名倒数
    """
    mrr_sum = 0.0
    num_queries = 0

    for qid, results in run.items():
        if qid not in qrels:
            continue
        num_queries += 1

        relevant_docs = {doc_id for doc_id, rel in qrels[qid].items() if rel > 0}

        for r in sorted(results, key=lambda x: x["rank"]):
            if r["rank"] > k:
                break
            if r["doc_id"] in relevant_docs:
                mrr_sum += 1.0 / r["rank"]
                break

    return mrr_sum / num_queries if num_queries > 0 else 0.0


def main():
    print("=" * 65)
    print("步骤4: 各库独立计算 Recall@5 / MRR@5")
    print("步骤5: 横向对比确定最优 Chunk Size")
    print("=" * 65)

    results = []

    for cs in CHUNK_SIZES:
        # 加载数据
        run = load_run_from_pool(cs)
        qrels = load_qrels(cs)

        # 计算指标
        recall_5 = calc_recall_at_k(run, qrels, k=5)
        mrr_5 = calc_mrr_at_k(run, qrels, k=5)

        # 附加统计
        num_queries = len(run)
        total_rel = sum(
            sum(1 for rel in qrels[qid].values() if rel > 0)
            for qid in qrels
        )
        avg_rel = total_rel / num_queries if num_queries else 0

        results.append({
            "chunk_size": cs,
            "Recall@5": recall_5,
            "MRR@5": mrr_5,
            "queries": num_queries,
            "avg_rel_per_q": avg_rel,
        })

        print(f"  chunk_size={cs}: Recall@5={recall_5:.4f}, MRR@5={mrr_5:.4f}, "
              f"queries={num_queries}, avg_rel/q={avg_rel:.2f}")

    # 横向对比表格
    print("\n" + "=" * 65)
    print("横向对比结果")
    print("=" * 65)
    print(f"{'Chunk Size':<12} {'Recall@5':<12} {'MRR@5':<12} {'Avg Rel/Q':<12}")
    print("-" * 48)
    for r in results:
        print(f"{r['chunk_size']:<12} {r['Recall@5']:<12.4f} {r['MRR@5']:<12.4f} "
              f"{r['avg_rel_per_q']:<12.2f}")
    print("-" * 48)

    # 确定最优 Chunk Size
    print("\n" + "=" * 65)
    print("步骤5: 确定最优 Chunk Size")
    print("=" * 65)

    best_recall = max(results, key=lambda x: x["Recall@5"])
    best_mrr = max(results, key=lambda x: x["MRR@5"])

    print(f"\n  Recall@5 最高: chunk_size={best_recall['chunk_size']} ({best_recall['Recall@5']:.4f})")
    print(f"  MRR@5   最高: chunk_size={best_mrr['chunk_size']} ({best_mrr['MRR@5']:.4f})")

    # 综合判定：两个指标都最高 / 综合最优
    if best_recall["chunk_size"] == best_mrr["chunk_size"]:
        optimal = best_recall["chunk_size"]
        print(f"\n  ★ 最优 Chunk Size = {optimal}  (Recall@5 与 MRR@5 均最高)")
    else:
        # 用加权综合得分 (等权)
        for r in results:
            r["combined"] = r["Recall@5"] + r["MRR@5"]
        best_combined = max(results, key=lambda x: x["combined"])
        print(f"\n  Recall@5 和 MRR@5 最高值不在同一 chunk size，按综合得分判定:")
        for r in sorted(results, key=lambda x: x["combined"], reverse=True):
            print(f"    chunk_size={r['chunk_size']}: Recall@5={r['Recall@5']:.4f} + MRR@5={r['MRR@5']:.4f} = {r['combined']:.4f}")
        optimal = best_combined["chunk_size"]
        print(f"\n  ★ 最优 Chunk Size = {optimal}  (综合得分最高: {best_combined['combined']:.4f})")

    print("\n" + "=" * 65)


if __name__ == "__main__":
    main()
