#!/usr/bin/env python3
"""
步骤1: 用 bge-reranker-base 对 pool 中每个 (query, chunk) 对打分
步骤2: 按 0.5 阈值生成 qrels (≥0.5 → rel=1, <0.5 → rel=0)

输入: data/pools/pool_{500..1000}.json (66 queries × 5 results each)
输出: data/qrels/qrels_{500..1000}.tsv
"""

import os
import json
import csv
import time
import numpy as np
from sentence_transformers import CrossEncoder

# ========== 配置 ==========
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
POOL_DIR = os.path.join(BASE_DIR, "data", "pools")
QRELS_DIR = os.path.join(BASE_DIR, "data", "qrels")

CHUNK_SIZES = [300, 400, 500, 600, 700, 800, 900, 1000]
RERANKER_MODEL = "BAAI/bge-reranker-base"
SCORE_THRESHOLD = 0.5
BATCH_SIZE = 64  # CrossEncoder 批处理大小


def main():
    start_time = time.time()
    print("=" * 60)
    print("QREL 生成: bge-reranker-base 打分 + 0.5 阈值")
    print(f"模型: {RERANKER_MODEL}")
    print(f"阈值: {SCORE_THRESHOLD}")
    print(f"批处理: {BATCH_SIZE}")
    print(f"Pool 文件: {CHUNK_SIZES}")
    print("=" * 60)

    # 1) 加载 reranker 模型（只加载一次，所有 chunk size 共用）
    print("\n[1/3] 加载 CrossEncoder 模型 ...")
    model = CrossEncoder(RERANKER_MODEL)
    print(f"  模型加载成功: {RERANKER_MODEL}")

    # 2) 删除旧 qrel 文件
    print("\n[2/3] 清理旧 qrel 文件 ...")
    os.makedirs(QRELS_DIR, exist_ok=True)
    for cs in CHUNK_SIZES:
        qrel_path = os.path.join(QRELS_DIR, f"qrels_{cs}.tsv")
        if os.path.exists(qrel_path):
            os.remove(qrel_path)
            print(f"  已删除: {qrel_path}")

    # 3) 对每个 chunk size 的 pool 打分并生成 qrel
    print(f"\n[3/3] 逐个 chunk size 打分生成 qrel ...")
    stats_summary = []

    for cs in CHUNK_SIZES:
        cs_start = time.time()
        pool_path = os.path.join(POOL_DIR, f"pool_{cs}.json")
        qrel_path = os.path.join(QRELS_DIR, f"qrels_{cs}.tsv")

        print(f"\n--- chunk_size={cs} ---")

        if not os.path.exists(pool_path):
            print(f"  跳过: 找不到 pool 文件 {pool_path}")
            continue

        # 加载 pool
        with open(pool_path, "r", encoding="utf-8") as f:
            pool = json.load(f)
        print(f"  加载 pool: {len(pool)} 个查询")

        # 收集所有 (query, chunk) 对，同时记录对应的 qid 和 doc_id
        all_pairs = []
        pair_meta = []  # (qid, doc_id)

        for entry in pool:
            qid = entry["qid"]
            query = entry["query"]
            for r in entry["results"]:
                all_pairs.append((query, r["text"]))
                pair_meta.append((qid, r["id"]))

        print(f"  待打分对数: {len(all_pairs)}")

        # 批量打分
        print(f"  执行 CrossEncoder 批量打分 ...")
        scores = model.predict(
            all_pairs,
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
        )
        # CrossEncoder 可能返回单值或数组
        if hasattr(scores, "flatten"):
            scores = scores.flatten()
        scores = [float(s) for s in scores]

        # 按 0.5 阈值生成 qrels
        results = []
        rel_count = 0
        for (qid, doc_id), score in zip(pair_meta, scores):
            relevance = 1 if score >= SCORE_THRESHOLD else 0
            if relevance == 1:
                rel_count += 1
            results.append({
                "qid": qid,
                "doc_id": doc_id,
                "relevance": relevance,
                "score": score,
            })

        # 保存 qrel 文件
        with open(qrel_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["qid", "doc_id", "relevance", "score"])
            for r in results:
                writer.writerow([
                    r["qid"],
                    r["doc_id"],
                    r["relevance"],
                    f"{r['score']:.4f}",
                ])

        # 统计
        total = len(results)
        avg_score = np.mean([r["score"] for r in results])
        score_arr = np.array([r["score"] for r in results])
        cs_time = time.time() - cs_start

        print(f"  保存到: {qrel_path}")
        print(f"  总对数: {total}, 相关(rel=1): {rel_count} ({rel_count/total*100:.1f}%)")
        print(f"  分数统计: min={score_arr.min():.4f}, max={score_arr.max():.4f}, "
              f"mean={avg_score:.4f}, median={np.median(score_arr):.4f}")
        print(f"  耗时: {cs_time:.1f}s")

        stats_summary.append({
            "chunk_size": cs,
            "total": total,
            "rel_count": rel_count,
            "rel_pct": rel_count / total * 100,
            "avg_score": avg_score,
            "time": cs_time,
        })

    # 汇总
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("汇总统计")
    print("=" * 60)
    print(f"{'CS':<6} {'总数':<8} {'相关数':<8} {'相关率':<8} {'平均分':<8} {'耗时(s)':<8}")
    print("-" * 46)
    for s in stats_summary:
        print(f"{s['chunk_size']:<6} {s['total']:<8} {s['rel_count']:<8} "
              f"{s['rel_pct']:<7.1f}% {s['avg_score']:<7.4f} {s['time']:<7.1f}")
    print(f"\n总耗时: {total_time:.1f}s")
    print("所有 qrel 文件生成完成！")


if __name__ == "__main__":
    main()
